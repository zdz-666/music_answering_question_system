"""第二层记忆：情景记忆（跨会话的历史问答），存在 Redis 的向量索引里。

每一轮对话写成一条记忆，带 user_id / session_id / timestamp 三个字段，
检索时先按 user_id 过滤、再用问题做语义相似度匹配，
"跨会话"= 同一 user_id 的多个会话。

排序不只看语义相似度，还叠加时间近因性 —— 同样相关时，刚聊过的内容优先于很久以前的：

    final_score = (1 - w) * 相似度 + w * recency_score
    recency_score = exp(-decay_factor * age_hours / 24)

（w = EPISODIC_RECENCY_WEIGHT，默认 0.3；decay_factor 默认 0.1，
即 1 天前约 0.90、1 周前约 0.50、1 个月前约 0.05。）

相似度先在本批候选内做 min-max 归一化再参与加权，原因见 _relevance 的说明 ——
本模型余弦带宽很窄，不归一化的话 [0,1] 的近因性会完全压倒相似度。

三个实现细节值得留意：
- 文档 id 用内容的 md5（确定性 id），写入是覆盖式的，因此重放/重试天然幂等，
  不需要先查重再写。
- 过滤与排除都交给 RediSearch 在服务端完成（TAG 匹配 + TAG 取反），
  不像 Chroma 那样受 where 语法限制而只能「多取几条再在 Python 里剔除」。
- 写入用的向量直接取 embed_query，与检索侧同一个口径；嵌入对同一文本是纯函数，
  这样还能吃上 config.get_embeddings() 自带的查询缓存。
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
    get_embeddings,
)
from observability import logger

from . import ft_index


def _doc_id(user_id: str, session_id: str, question: str, answer: str) -> str:
    raw = f"{user_id}|{session_id}|{question}|{answer}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _document(hit: dict, user_id: str) -> Document:
    return Document(
        page_content=hit["text"],
        metadata={
            "user_id": user_id,
            # 用原始的 session_id（不是索引里做过滤用的摘要），调用方回传也认得
            "session_id": hit["session"],
            "timestamp": hit["ts"],
        },
    )


def _relevance(distance: float) -> float:
    """把 RediSearch 的 COSINE 距离折成绝对余弦相似度（0~1）。

    索引是按 DISTANCE_METRIC COSINE 建的，KNN 给出的 score 就是 1 - cos(q, v)，
    所以 1 - score 恰为余弦相似度（实测：向量完全相同得 0，正交得 1）。
    负相关（距离 > 1）截到 0，避免把近因性那一项抵消掉。

    返回值是**绝对**余弦，搜索时还会在候选集内再做一次 min-max —— 本模型对长短
    文本的余弦值被压在很窄的带里（实测"同主题"0.25 与"不同主题"0.20 只差 0.055），
    绝对量纲下 0.7*0.055 = 0.039 的区分度会被 0.3*1.0 = 0.3 的近因性完全淹没。
    """
    return max(0.0, min(1.0, 1.0 - distance))


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

    text = f"问：{question}\n答：{answer}"
    ft_index.upsert(
        EPISODIC_COLLECTION,
        doc_id=_doc_id(user_id, session_id, question, answer),
        vector=get_embeddings().embed_query(text),
        text=text,
        user_id=user_id,
        timestamp=timestamp or datetime.now().isoformat(),
        session_id=session_id,
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
    hits = ft_index.search(
        EPISODIC_COLLECTION,
        vector=get_embeddings().embed_query(question),
        user_id=user_id,
        k=fetch_k,
        exclude_session_id=exclude_session_id,
    )

    now = datetime.now()
    candidates = [
        (_document(hit, user_id), _relevance(hit["distance"]), _recency_score(hit["ts"], now))
        for hit in hits
    ]
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