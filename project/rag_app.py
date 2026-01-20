import os
import re
import json
import time
import pickle
from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import fitz  # pymupdf
import faiss
import requests
import streamlit as st
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config
# ---------------------------
DATA_DIR = "data"
PDF_DIR = os.path.join(DATA_DIR, "pdf")
INDEX_DIR = os.path.join(DATA_DIR, "index")
FAISS_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

# 会话持久化文件
CONV_PATH = os.path.join(INDEX_DIR, "conversations.json")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
TOP_K = 6
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
MIN_SCORE = 0.18

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """你是面向计算机专业课程的智能答疑助手。
你只能基于【证据】回答，不得编造证据中不存在的内容。
如果证据不足以支持回答，请明确说“资料不足”，并建议用户补充资料或换一种问法。
输出格式必须为：
1) 【回答】用要点分条陈述
2) 【引用】列出你使用到的证据编号与来源（文件名、页码），例如：[1] 文件.pdf p12
"""

# ---------------------------
# Multi-turn retrieval defaults (固定：永远启用、拼接12轮)
# ---------------------------
ENABLE_MULTITURN_DEFAULT = True
MULTITURN_HISTORY_TURNS_DEFAULT = 12
SHOW_RETRIEVAL_QUERY_DEFAULT = False  # 不显示

# ---------------------------
# Topic hint (lightweight)
# ---------------------------
DOMAIN_KEYWORDS = {
    "C/C++ 编程": [
        "c语言", "c 程序", "c程序", "指针", "数组", "函数", "结构体", "宏", "预处理", "编译",
        "printf", "scanf", "malloc", "free", "stdio", "stdlib", "pointer", "array",
        "数组指针", "指针数组", "二维数组", "多维数组", "整型", "浮点", "float", "double", "long double"
    ],
    "计算机网络": [
        "tcp", "udp", "ip", "路由", "arp", "mac", "dns", "拥塞", "滑动窗口", "csma", "以太网",
        "三次握手", "四次挥手"
    ],
    "操作系统": [
        "进程", "线程", "调度", "死锁", "内存", "分页", "段页", "虚拟存储", "文件系统",
        "linux", "shell", "系统调用"
    ],
    "软件工程/测试": [
        "需求", "用例", "测试", "junit", "集成测试", "验收测试", "螺旋模型", "v模型", "sdlc"
    ],
    "人工智能/机器学习": [
        "机器学习", "深度学习", "神经网络", "回归", "分类", "聚类", "损失", "梯度", "lstm", "cnn"
    ],
    "音乐": [
        "古典音乐", "交响乐", "奏鸣曲", "协奏曲", "巴洛克", "浪漫主义", "古典主义",
        "莫扎特", "贝多芬", "巴赫", "柴可夫斯基", "作曲家", "钢琴", "小提琴"
    ],
}


def infer_topic(text: str) -> Tuple[str, int]:
    t = (text or "").lower()
    best_topic = "未知"
    best_score = 0
    for topic, kws in DOMAIN_KEYWORDS.items():
        score = 0
        for k in kws:
            if k and k.lower() in t:
                score += 1
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic, best_score


def infer_query_topic(query: str) -> str:
    topic, score = infer_topic(query)
    return topic if score > 0 else "未知"


def infer_corpus_topic(chunks: List["Chunk"], sample_n: int = 80) -> str:
    if not chunks:
        return "未知"
    n = min(sample_n, len(chunks))
    step = max(1, len(chunks) // n)

    # 兜底：即使 chunks 里出现异常对象也不崩
    sample_text = "\n".join(
        (getattr(chunks[i], "text", "") or "")
        for i in range(0, len(chunks), step)
    )[:200000]

    topic, score = infer_topic(sample_text)
    return topic if score > 0 else "未知"


# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Chunk:
    chunk_id: str
    file_name: str
    page_start: int
    page_end: int
    text: str


def normalize_chunks(raw) -> List[Chunk]:
    """
    把历史版本/异常结构的 chunk 统一转成 Chunk dataclass，
    避免 .text / .file_name 等访问崩溃。
    """
    if not raw:
        return []

    # 已经是新格式
    if isinstance(raw, list) and raw and isinstance(raw[0], Chunk):
        return raw

    out: List[Chunk] = []
    # 允许 raw 不是 list 的情况
    items = raw if isinstance(raw, list) else list(raw)

    for idx, it in enumerate(items):
        # 1) dict 格式（最常见旧格式）
        if isinstance(it, dict):
            out.append(Chunk(
                chunk_id=str(it.get("chunk_id", f"unknown::c{idx:06d}")),
                file_name=str(it.get("file_name", it.get("source", "unknown.pdf"))),
                page_start=int(it.get("page_start", it.get("page", 1)) or 1),
                page_end=int(it.get("page_end", it.get("page_start", it.get("page", 1)) or 1) or 1),
                text=str(it.get("text", it.get("page_content", it.get("content", ""))) or "")
            ))
            continue

        # 2) tuple/list 格式： (chunk_id, file_name, page_start, page_end, text, ...)
        if isinstance(it, (tuple, list)) and len(it) >= 5:
            chunk_id, file_name, page_start, page_end, text = it[:5]
            out.append(Chunk(
                chunk_id=str(chunk_id),
                file_name=str(file_name),
                page_start=int(page_start or 1),
                page_end=int(page_end or page_start or 1),
                text=str(text or "")
            ))
            continue

        # 3) 兜底：有 text/page_content 属性的对象
        text = getattr(it, "text", None) or getattr(it, "page_content", None) or getattr(it, "content", None) or ""
        file_name = getattr(it, "file_name", None) or getattr(it, "source", None) or "unknown.pdf"
        page_start = getattr(it, "page_start", None) or getattr(it, "page", None) or 1
        page_end = getattr(it, "page_end", None) or page_start
        chunk_id = getattr(it, "chunk_id", None) or f"unknown::c{idx:06d}"

        out.append(Chunk(
            chunk_id=str(chunk_id),
            file_name=str(file_name),
            page_start=int(page_start or 1),
            page_end=int(page_end or page_start or 1),
            text=str(text or "")
        ))

    return out


# ---------------------------
# Utility
# ---------------------------
def ensure_dirs():
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)


def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.replace("-\n", "")
    return s.strip()


def extract_pdf_pages(pdf_path: str) -> List[Tuple[int, str]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        text = clean_text(text)
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


def chunk_pages(file_name: str, pages: List[Tuple[int, str]],
                chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    buf = ""
    buf_page_start = None
    buf_page_end = None
    chunk_idx = 0

    def flush_buffer(final: bool = False):
        nonlocal buf, buf_page_start, buf_page_end, chunk_idx
        txt = clean_text(buf)
        if txt:
            chunk_id = f"{file_name}::c{chunk_idx:06d}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                file_name=file_name,
                page_start=buf_page_start or 1,
                page_end=buf_page_end or (buf_page_start or 1),
                text=txt
            ))
            chunk_idx += 1
        if final:
            buf = ""
            buf_page_start = None
            buf_page_end = None
        else:
            buf = txt[-overlap:] if len(txt) > overlap else txt
            buf_page_start = buf_page_end

    for page_no, page_text in pages:
        if buf_page_start is None:
            buf_page_start = page_no
        buf_page_end = page_no

        paras = [p.strip() for p in page_text.split("\n") if p.strip()]
        for p in paras:
            if not buf:
                buf_page_start = page_no
            buf_page_end = page_no

            candidate = (buf + "\n" + p).strip() if buf else p
            if len(candidate) <= chunk_size:
                buf = candidate
            else:
                flush_buffer(final=False)
                buf = p
                buf_page_start = page_no
                buf_page_end = page_no

                while len(buf) > chunk_size:
                    part = buf[:chunk_size]
                    buf = buf[chunk_size:]
                    chunk_id = f"{file_name}::c{chunk_idx:06d}"
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        file_name=file_name,
                        page_start=page_no,
                        page_end=page_no,
                        text=clean_text(part)
                    ))
                    chunk_idx += 1
                    buf = (part[-overlap:] + buf) if overlap > 0 else buf

    if buf.strip():
        flush_buffer(final=True)

    return chunks


def load_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return np.asarray(vecs, dtype="float32")


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def save_index(index: faiss.Index, chunks: List[Chunk]):
    faiss.write_index(index, FAISS_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)


def load_index() -> Tuple[faiss.Index, List[Chunk]]:
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        raw = pickle.load(f)

    chunks = normalize_chunks(raw)

    # 可选：把归一化后的结果写回，避免下次还走兼容分支
    try:
        if isinstance(raw, list) and raw and not isinstance(raw[0], Chunk):
            with open(META_PATH, "wb") as wf:
                pickle.dump(chunks, wf)
    except Exception:
        pass

    return index, chunks


# ---------------------------
# LLM
# ---------------------------
def llm_chat(messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    if not (LLM_API_KEY and LLM_BASE_URL and LLM_MODEL):
        raise RuntimeError("LLM_API_KEY / LLM_BASE_URL / LLM_MODEL 未配置完整。")

    if LLM_BASE_URL.endswith("/v1"):
        url = f"{LLM_BASE_URL}/chat/completions"
    else:
        url = f"{LLM_BASE_URL}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=90)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


# ---------------------------
# Multi-turn retrieval helpers
# ---------------------------
FOLLOWUP_MARKERS = [
    "它", "他", "她", "这", "那", "这个", "那个", "上面", "刚才", "继续", "再说",
    "为什么", "怎么", "那然后", "然后呢", "还有呢", "如何", "能不能"
]


def is_followup_question(q: str) -> bool:
    q = (q or "").strip()
    if len(q) <= 10:
        return True
    return any(m in q for m in FOLLOWUP_MARKERS)


def build_retrieval_query(current_q: str, user_history: List[str], max_turns: int = 4) -> str:
    current_q = (current_q or "").strip()
    if not user_history:
        return current_q
    if not is_followup_question(current_q):
        return current_q
    tail = [h.strip() for h in user_history[-max_turns:] if h.strip()]
    return "；".join(tail + [current_q])


# ---------------------------
# Retrieval + Guards
# ---------------------------
def retrieve(index: faiss.Index, chunks: List[Chunk], embedder: SentenceTransformer,
             query: str, top_k: int = TOP_K) -> List[Tuple[float, Chunk]]:
    qv = embed_texts(embedder, [query], batch_size=1)
    scores, ids = index.search(qv, top_k)
    results = []
    for s, i in zip(scores[0], ids[0]):
        if i < 0:
            continue
        results.append((float(s), chunks[int(i)]))
    return results


def format_evidence(results: List[Tuple[float, Chunk]]) -> str:
    lines = []
    for idx, (score, c) in enumerate(results, start=1):
        src = f"{c.file_name} p{c.page_start}" if c.page_start == c.page_end else f"{c.file_name} p{c.page_start}-{c.page_end}"
        snippet = (c.text or "")[:450].replace("\n", " ")
        lines.append(f"[{idx}] 来源：{src}\n片段：{snippet}")
    return "\n\n".join(lines)


_CH_CONNECTORS = ["和", "与", "及", "以及", "还有", "以及其", "跟", "同", "对比", "比较"]
_STOP_PHRASES = [
    "是什么", "是啥", "是什么呢", "什么", "怎么", "如何", "为什么",
    "能不能", "可以吗", "吗", "呢", "呀", "啊"
]


def extract_keywords(q: str) -> List[str]:
    q = (q or "").strip().lower()

    for p in _STOP_PHRASES:
        q = q.replace(p, " ")

    for c in _CH_CONNECTORS:
        q = q.replace(c, " ")

    q = q.replace("的", " ").replace("了", " ").replace("着", " ")

    q = re.sub(r"[，,。.!！?？;；:/\\()\[\]{}<>\"'“”‘’]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()

    words = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{3,}", q)

    norm_map = {
        "实型": "浮点",
        "整数": "整型",
        "整型数": "整型",
        "浮点数": "浮点",
        "浮点型": "浮点",
    }

    kws = []
    for w in words:
        w = w.strip().lower()
        w = norm_map.get(w, w)
        if w and w not in kws:
            kws.append(w)

    return kws


def _sub_keywords(w: str) -> List[str]:
    w = (w or "").strip()
    if len(w) <= 4:
        return []
    subs = []
    for n in (2, 3, 4):
        for i in range(0, len(w) - n + 1):
            s = w[i:i+n]
            if s and s not in subs:
                subs.append(s)
    return subs


def evidence_has_keywords(results: List[Tuple[float, Chunk]], query: str, min_hits: int = 1) -> bool:
    kws = extract_keywords(query)
    if not kws:
        return True

    evidence_text = "\n".join((c.text or "").lower() for _, c in results)

    hits = 0
    for k in kws:
        if k in evidence_text:
            hits += 1
            continue
        for sub in _sub_keywords(k):
            if sub in evidence_text:
                hits += 1
                break

    return hits >= min_hits


def make_hint_block(query: str, corpus_topic: str) -> str:
    q_topic = infer_query_topic(query)
    if q_topic != "未知" and corpus_topic != "未知" and q_topic != corpus_topic:
        return (
            "\n\n【提示】\n"
            f"- 检测到你的问题更像是「{q_topic}」相关，但当前已导入资料主要是「{corpus_topic}」。\n"
            f"- 建议导入与「{q_topic}」相关的教材/讲义/课件 PDF，或换成更贴合当前资料的问法。"
        )
    return ""


def answer_with_rag(query: str, history: List[Dict[str, str]],
                    results: List[Tuple[float, Chunk]], corpus_topic: str,
                    guard_query: str = "") -> str:
    if (not results) or (max(s for s, _ in results) < MIN_SCORE):
        base = (
            "【回答】\n"
            "- 资料不足：在当前已导入的PDF资料中未检索到足以支撑该问题的明确依据。\n\n"
            "【引用】\n"
            "- 无\n\n"
            "建议：换一种更具体的问法，或导入包含该知识点的讲义/教材章节。"
        )
        return base + make_hint_block(query, corpus_topic)

    guard_q = guard_query.strip() if guard_query else query
    if not evidence_has_keywords(results, guard_q, min_hits=1):
        base = (
            "【回答】\n"
            "- 资料不足：检索到的片段与问题关键词不匹配，无法基于现有PDF给出可靠回答。\n\n"
            "【引用】\n"
            "- 无\n\n"
            "建议：导入包含该知识点的教材/讲义，或换更贴合本资料的问法。"
        )
        return base + make_hint_block(query, corpus_topic)

    evidence = format_evidence(results)
    user_prompt = f"""【问题】
{query}

【证据】
{evidence}
"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_prompt})
    return llm_chat(messages)


def fallback_answer_on_error(err: Exception, results: List[Tuple[float, Chunk]]) -> str:
    lines = [
        "【回答】",
        f"- LLM 调用失败：{err}",
        "- 已返回检索到的资料摘要（未生成扩展解释）。",
        "",
        "【引用】"
    ]
    if not results:
        lines.append("- 无")
        return "\n".join(lines)

    for i, (score, c) in enumerate(results, start=1):
        src = f"{c.file_name} p{c.page_start}" if c.page_start == c.page_end else f"{c.file_name} p{c.page_start}-{c.page_end}"
        lines.append(f"- [{i}] {src}")

    lines.append("")
    for i, (score, c) in enumerate(results, start=1):
        snippet = (c.text or "").strip().replace("\n", " ")
        snippet = snippet[:260] + ("…" if len(snippet) > 260 else "")
        lines.append(f"证据[{i}] 摘要：{snippet}")
    return "\n".join(lines)


# ---------------------------
# Conversation store（多会话 + 自动命名 + 持久化）
# ---------------------------
def _new_conv_id() -> str:
    return f"c{int(time.time()*1000)}"


def _normalize_title(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip(" -—_，,。.!！?？")
    return s


def _make_unique_title(base: str, existing_titles: List[str]) -> str:
    base = _normalize_title(base)
    if not base:
        base = "新对话"
    if base not in existing_titles:
        return base
    k = 2
    while True:
        cand = f"{base} ({k})"
        if cand not in existing_titles:
            return cand
        k += 1


def auto_name_conversation_if_needed(conv_id: str, first_user_query: str):
    store = st.session_state.conv_store
    conv = store["conversations"][conv_id]
    cur_title = (conv.get("title") or "").strip()

    is_default = (not cur_title) or bool(re.fullmatch(r"对话\s*\d+", cur_title)) or cur_title in ("新对话",)
    if not is_default:
        return

    base = _normalize_title(first_user_query)
    if len(base) > 18:
        base = base[:18] + "…"

    existing = [c.get("title", "") for cid, c in store["conversations"].items() if cid != conv_id]
    conv["title"] = _make_unique_title(base, existing)
    conv["updated_at"] = int(time.time())


def load_conversations_from_disk() -> Dict:
    if not os.path.exists(CONV_PATH):
        return {"active_id": "", "conversations": {}}
    try:
        with open(CONV_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "conversations" not in data or not isinstance(data["conversations"], dict):
            return {"active_id": "", "conversations": {}}
        return data
    except Exception:
        return {"active_id": "", "conversations": {}}


def save_conversations_to_disk(store: Dict):
    try:
        with open(CONV_PATH, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def init_conversation_state():
    if "conv_store" in st.session_state and "active_conv_id" in st.session_state:
        return

    store = load_conversations_from_disk()

    if not store.get("conversations"):
        cid = _new_conv_id()
        store["conversations"] = {
            cid: {
                "title": "对话 1",
                "history": [],
                "created_at": int(time.time()),
                "updated_at": int(time.time()),
            }
        }
        store["active_id"] = cid
        save_conversations_to_disk(store)

    if store.get("active_id") not in store["conversations"]:
        store["active_id"] = next(iter(store["conversations"].keys()))

    st.session_state.conv_store = store
    st.session_state.active_conv_id = store["active_id"]


def get_active_history() -> List[Dict[str, str]]:
    cid = st.session_state.active_conv_id
    return st.session_state.conv_store["conversations"][cid]["history"]


def set_active_history(new_history: List[Dict[str, str]]):
    cid = st.session_state.active_conv_id
    st.session_state.conv_store["conversations"][cid]["history"] = new_history
    st.session_state.conv_store["conversations"][cid]["updated_at"] = int(time.time())


def conv_ids_sorted() -> List[str]:
    convs = st.session_state.conv_store["conversations"]
    return sorted(convs.keys(), key=lambda cid: convs[cid].get("updated_at", 0), reverse=True)


def create_new_conversation():
    store = st.session_state.conv_store
    cid = _new_conv_id()
    store["conversations"][cid] = {
        "title": "新对话",
        "history": [],
        "created_at": int(time.time()),
        "updated_at": int(time.time()),
    }
    store["active_id"] = cid
    st.session_state.active_conv_id = cid
    save_conversations_to_disk(store)


def set_active_conversation(cid: str):
    store = st.session_state.conv_store
    if cid not in store["conversations"]:
        return
    st.session_state.active_conv_id = cid
    store["active_id"] = cid
    save_conversations_to_disk(store)


def rename_conversation(cid: str, new_title: str):
    store = st.session_state.conv_store
    new_title = _normalize_title(new_title)
    if not new_title:
        return
    existing = [c.get("title", "") for k, c in store["conversations"].items() if k != cid]
    store["conversations"][cid]["title"] = _make_unique_title(new_title, existing)
    store["conversations"][cid]["updated_at"] = int(time.time())
    save_conversations_to_disk(store)


def delete_conversation(cid: str):
    store = st.session_state.conv_store
    convs = store["conversations"]
    if cid not in convs:
        return
    if len(convs) <= 1:
        convs[cid]["history"] = []
        convs[cid]["title"] = "对话 1"
        convs[cid]["updated_at"] = int(time.time())
        store["active_id"] = cid
        st.session_state.active_conv_id = cid
        save_conversations_to_disk(store)
        return

    del convs[cid]
    new_active = next(iter(convs.keys()))
    store["active_id"] = new_active
    st.session_state.active_conv_id = new_active
    save_conversations_to_disk(store)


# ---------------------------
# Sidebar UI helpers（展开列表 + 单独新聊天按钮）
# ---------------------------
def _fmt_time(ts: int) -> str:
    if not ts:
        return "--"
    return time.strftime("%m-%d %H:%M", time.localtime(ts))


def _group_by_day(conv_ids: List[str], convs: Dict[str, Dict]) -> Dict[str, List[str]]:
    groups = defaultdict(list)
    for cid in conv_ids:
        ts = convs[cid].get("updated_at", 0) or convs[cid].get("created_at", 0) or 0
        day = time.strftime("%Y-%m-%d", time.localtime(ts)) if ts else "未知日期"
        groups[day].append(cid)
    return dict(sorted(groups.items(), key=lambda x: x[0], reverse=True))


def _sidebar_chat_list():
    init_conversation_state()
    store = st.session_state.conv_store
    convs = store["conversations"]
    active = st.session_state.active_conv_id

    st.markdown("## 对话")

    if st.button("➕ 新聊天", use_container_width=True):
        create_new_conversation()
        st.rerun()

    q = st.text_input("搜索对话", value="", placeholder="输入关键词过滤…")
    q_low = (q or "").strip().lower()

    ids = conv_ids_sorted()
    if q_low:
        ids = [cid for cid in ids if q_low in (convs[cid].get("title", "") or "").lower()]

    st.caption(f"共 {len(ids)} 个对话")

    groups = _group_by_day(ids, convs)

    for day, day_ids in groups.items():
        st.markdown(f"**{day}**")
        for cid in day_ids:
            title = convs[cid].get("title", "") or cid
            ts = convs[cid].get("updated_at", 0) or convs[cid].get("created_at", 0) or 0
            prefix = "▶ " if cid == active else ""
            label = f"{prefix}{title}  ·  {_fmt_time(ts)}"

            if st.button(label, key=f"open_{cid}", use_container_width=True):
                set_active_conversation(cid)
                st.rerun()

        st.write("")

    st.divider()
    st.markdown("## 当前对话管理")
    cur_title = convs[active].get("title", "")
    new_title = st.text_input("重命名", value=cur_title, key="rename_input")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("保存名称", use_container_width=True):
            rename_conversation(active, new_title)
            st.rerun()
    with col_b:
        if st.button("删除当前", use_container_width=True):
            delete_conversation(active)
            st.rerun()


# ---------------------------
# Streamlit App
# ---------------------------
def main():
    ensure_dirs()
    st.set_page_config(page_title="PDF+FAISS 课程答疑（RAG）", layout="wide")
    st.title("课程智能答疑（PDF + FAISS + API LLM）")

    with st.sidebar:
        _sidebar_chat_list()

        st.divider()
        st.header("索引管理")
        st.write("将PDF放入：`data/pdf/`")
        rebuild = st.button("重建索引（扫描所有PDF）")

        st.divider()
        st.header("LLM 配置检查")
        st.write(f"BASE_URL: {'已设置' if LLM_BASE_URL else '未设置'}")
        st.write(f"MODEL: {'已设置' if LLM_MODEL else '未设置'}")
        st.write(f"API_KEY: {'已设置' if LLM_API_KEY else '未设置'}")
        st.caption("如未设置，请先配置环境变量：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL")

        # 注意：这里不再显示“多轮检索设置”UI

    if "embedder" not in st.session_state:
        with st.spinner("加载 embedding 模型..."):
            st.session_state.embedder = load_embedder()

    index_exists = os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)

    def do_build():
        pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        if not pdf_files:
            st.error("未找到PDF。请将PDF放入 data/pdf/ 后重试。")
            return False

        all_chunks: List[Chunk] = []
        for pf in tqdm(pdf_files, desc="PDF解析"):
            p = os.path.join(PDF_DIR, pf)
            pages = extract_pdf_pages(p)
            chunks = chunk_pages(pf, pages)
            all_chunks.extend(chunks)

        if not all_chunks:
            st.error("PDF未抽取到有效文本（可能是扫描版）。请更换文本型PDF或后续加OCR。")
            return False

        texts = [c.text for c in all_chunks]
        with st.spinner("向量化中..."):
            vecs = embed_texts(st.session_state.embedder, texts, batch_size=32)

        with st.spinner("建立FAISS索引..."):
            index = build_faiss_index(vecs)
            save_index(index, all_chunks)

        st.success(f"索引构建完成：chunks={len(all_chunks)}，PDF={len(pdf_files)}")
        return True

    if rebuild or (not index_exists):
        if not index_exists:
            st.info("未检测到索引，将自动构建一次。")
        ok = do_build()
        if not ok:
            st.stop()

    if not (os.path.exists(FAISS_PATH) and os.path.exists(META_PATH)):
        st.stop()

    # ---- 加载索引（带旧格式兼容/自动修复）----
    if "index" not in st.session_state or "chunks" not in st.session_state:
        with st.spinner("加载索引..."):
            try:
                st.session_state.index, st.session_state.chunks = load_index()
            except Exception as e:
                st.warning(f"索引加载失败，尝试自动重建：{e}")
                ok = do_build()
                if not ok:
                    st.stop()
                st.session_state.index, st.session_state.chunks = load_index()

        # 极端兜底：如果仍不是 Chunk，就重建
        if st.session_state.chunks and not isinstance(st.session_state.chunks[0], Chunk):
            st.warning("检测到旧版本索引元数据格式不兼容，正在自动重建索引…")
            ok = do_build()
            if not ok:
                st.stop()
            st.session_state.index, st.session_state.chunks = load_index()
            st.rerun()

    if "corpus_topic" not in st.session_state:
        st.session_state.corpus_topic = infer_corpus_topic(st.session_state.chunks)

    history = get_active_history()
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content:
            st.chat_message(role).write(content)

    query = st.chat_input("请输入你的问题（基于已导入PDF资料回答）")
    if query:
        st.chat_message("user").write(query)

        if len(history) == 0:
            auto_name_conversation_if_needed(st.session_state.active_conv_id, query)

        # 固定默认：启用多轮检索，拼接12轮
        enable_multiturn = ENABLE_MULTITURN_DEFAULT
        memory_turns = MULTITURN_HISTORY_TURNS_DEFAULT
        show_retrieval_query = SHOW_RETRIEVAL_QUERY_DEFAULT

        user_questions = [m["content"] for m in history if m.get("role") == "user"]
        retrieval_query = query
        if enable_multiturn:
            retrieval_query = build_retrieval_query(query, user_questions, max_turns=memory_turns)

        if show_retrieval_query and retrieval_query != query:
            st.info(f"本轮实际用于检索的问题：{retrieval_query}")

        t0 = time.time()
        results = retrieve(
            st.session_state.index,
            st.session_state.chunks,
            st.session_state.embedder,
            retrieval_query,
            TOP_K
        )
        t_retr = time.time()

        try:
            # guard 用当前 query（避免多轮拼接误伤）
            ans = answer_with_rag(query, history, results, st.session_state.corpus_topic, guard_query=query)
        except Exception as e:
            ans = fallback_answer_on_error(e, results)

        t_llm = time.time()
        st.chat_message("assistant").write(ans)

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": ans})
        history = history[-40:]
        set_active_history(history)

        st.session_state.conv_store["active_id"] = st.session_state.active_conv_id
        save_conversations_to_disk(st.session_state.conv_store)

        with st.expander("本次检索到的证据（Top-K）", expanded=True):
            cnt = Counter(c.file_name for _, c in results)
            if cnt:
                dist = "；".join([f"{k}×{v}" for k, v in cnt.items()])
                st.write("命中文件分布：", dist)
            for i, (score, c) in enumerate(results, start=1):
                src = f"{c.file_name} p{c.page_start}" if c.page_start == c.page_end else f"{c.file_name} p{c.page_start}-{c.page_end}"
                st.markdown(f"**[{i}] 相似度：{score:.3f}｜{src}｜{c.chunk_id}**")
                st.write(c.text)

        st.caption(f"检索耗时：{(t_retr - t0):.2f}s｜生成耗时：{(t_llm - t_retr):.2f}s")


if __name__ == "__main__":
    main()
