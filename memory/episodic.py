"""第二层记忆：情景记忆（跨会话的历史问答），用 Chroma 存储。

每一轮对话写成一条文档，metadata 带 user_id / session_id / timestamp，
检索时按 user_id 过滤后再用问题做语义相似度匹配，"跨会话"= 同一 user_id 的多个会话。

排序不只看语义相似度，还叠加时间近因性 —— 同样相关时，刚聊过的内容优先于很久以前的：

    final_score = (1 - w) * 相似度 + w * recency_score
    recency_score = exp(-decay_factor * age_hours / 24)

（w = EPISODIC_RECENCY_WEIGHT，默认 0.3；decay_factor 默认 0.1，
即 1 天前约 0.90、1 周前约 0.50、1 个月前约 0.05。）

相似度先在本批候选内做 min-max 归一化再参与加权，原因见 _relevance 的说明 ——
本模型余弦带宽很窄，不归一化的话 [0,1] 的近因性会完全压倒相似度。

三个实现细节值得留意：
- 文档 id 用内容的 md5（确定性 id），Chroma 底层的 add 是 upsert，
  因此重放/重试天然幂等，不需要先查重再写。
- 过滤条件只写单键 {"user_id": ...}。chromadb 要求 where 顶层恰好一个操作符
  （多条件必须套 $and），且 $ne 的实现会让"缺少该字段"的记录也被返回，
  所以这里不用 $ne 排除当前会话，而是多取几条后在 Python 里剔除。
"""

import hashlib
import math
from datetime import datetime

from langchain_core.documents import Document

from config import (
    EPISODIC_COLLECTION,
    EPISODIC_DECAY_FACTOR,
    EPISODIC_RECENCY_WEIGHT,
    EPISODIC_TOP_K,
)
from data_storage import load_vector_store
from observability import logger


def _doc_id(user_id: str, session_id: str, question: str, answer: str) -> str:
    raw = f"{user_id}|{session_id}|{question}|{answer}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _relevance(distance: float) -> float:
    """把 Chroma 的距离折成绝对余弦相似度（0~1）。

    本集合 hnsw.space = l2，且 Chroma 在 l2 空间返回的是**平方**欧氏距离
    （实测 d=1.4822 与手算 ‖q-v‖²=1.4849 一致，而非 1.2186 的普通距离）。
    本项目嵌入为 4096 维单位向量（实测 norm = 1.000000），于是

        ‖q-v‖² = 2 - 2·cos(q, v)   ⟹   cos(q, v) = 1 - d/2

    即 1 - d/2 恰为余弦相似度。负相关截到 0，避免把近因性那一项抵消掉。

    注意不要照搬 langchain 的 _euclidean_relevance_score_fn(1 - d/√2)：它假设 d 是
    普通 L2 距离，套在平方距离上会恒为负、把所有候选压成 0（相似度项失效）。
    也不直接调 similarity_search_with_relevance_scores —— 它同样基于该错误口径，
    且对越界分数只发 warning，warning 里带整批文档、每次调用重复打印，反而污染日志。

    返回值是**绝对**余弦，搜索时还会在候选集内再做一次 min-max —— 本模型对长短
    文本的余弦值被压在很窄的带里（实测"同主题"0.25 与"不同主题"0.20 只差 0.055），
    绝对量纲下 0.7*0.055 = 0.039 的区分度会被 0.3*1.0 = 0.3 的近因性完全淹没。
    """
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def _recency_score(timestamp: str | None, now: datetime) -> float:
    """时间近因性得分，按用户给定的公式：

        recency_score = exp(-decay_factor * age_hours / 24)

    时间戳缺失或无法解析时按"最旧"处理（0 分）——不让损坏的数据靠新鲜度占便宜。
    时钟偏差导致的时间戳在未来时，age_hours 截到 0（最多给 1 分，不做额外奖励）。
    """
    if not timestamp:
        return 0.0
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return 0.0
    age_hours = max(0.0, (now - moment).total_seconds() / 3600.0)
    return math.exp(-EPISODIC_DECAY_FACTOR * age_hours / 24.0)


def add_turn(
    user_id: str,
    session_id: str,
    question: str,
    answer: str,
    timestamp: str | None = None,
) -> None:
    """记一轮问答；没有 user_id 就不写（无法隔离的记忆没有意义）。"""
    if not user_id:
        return

    doc = Document(
        page_content=f"问：{question}\n答：{answer}",
        metadata={
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": timestamp or datetime.now().isoformat(),
        },
    )
    load_vector_store(EPISODIC_COLLECTION).add_documents(
        [doc], ids=[_doc_id(user_id, session_id, question, answer)]
    )


def search(
    user_id: str,
    question: str,
    exclude_session_id: str | None = None,
    k: int = EPISODIC_TOP_K,
) -> list[Document]:
    """取跨会话的相关问答：语义相似度 + 时间近因性加权排序，排除当前会话。

    多取一些候选再重排，否则时间维度没有发挥空间——只按距离取前 k 条的话，
    被近因性提上来的旧记忆根本没机会进入候选集。
    """
    if not user_id:
        return []

    fetch_k = max(k * 4, k + 5)
    scored = load_vector_store(EPISODIC_COLLECTION).similarity_search_with_score(
        question,
        k=fetch_k,
        filter={"user_id": user_id},
    )

    now = datetime.now()
    candidates = []
    for doc, distance in scored:
        if doc.metadata.get("session_id") == exclude_session_id:
            continue
        candidates.append(
            (doc, _relevance(distance), _recency_score(doc.metadata.get("timestamp"), now))
        )

    if not candidates:
        return []

    # 候选集内 min-max：两项都归到 [0,1] 后 w 才真正控制"相似度 vs 时间"的平衡。
    # 只有一个候选、或全部同分时无从区分，统一取 1.0（相似度退化为常数，交给近因性排）。
    cosines = [cosine for _, cosine, _ in candidates]
    lo, hi = min(cosines), max(cosines)
    span = hi - lo

    ranked = []
    for doc, cosine, recency in candidates:
        relevance = (cosine - lo) / span if span > 1e-12 else 1.0
        final = (1.0 - EPISODIC_RECENCY_WEIGHT) * relevance + EPISODIC_RECENCY_WEIGHT * recency
        ranked.append((final, doc, relevance, cosine, recency))

    ranked.sort(key=lambda item: item[0], reverse=True)
    top = ranked[:k]

    if top:
        logger.info(
            "memory.episodic_rank",
            extra={
                "fields": {
                    "user_id": user_id,
                    "candidates": len(ranked),
                    "cos_band": round(span, 4),
                    "top": [
                        {
                            "final": round(final, 4),
                            "relevance": round(relevance, 4),
                            "cosine": round(cosine, 4),
                            "recency": round(recency, 4),
                        }
                        for final, _, relevance, cosine, recency in top
                    ],
                }
            },
        )

    return [doc for _, doc, _, _, _ in top]