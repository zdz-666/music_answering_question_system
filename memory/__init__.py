"""三层记忆的对外入口。

三层各管一段：
1. 工作记忆（memory.working + session_store）—— 当前会话的最近消息 + 滚动摘要，直接注入 prompt；
2. 情景记忆（memory.episodic）—— 同一 user_id 的跨会话历史问答，按语义相似度检索；
3. 用户画像（memory.profile）—— 从对话里抽取的长期偏好与实体，按语义相似度检索。

三层的落点都在 Redis：工作记忆是 list + hash，后两层是 RediSearch 的向量索引
（统一见 memory/ft_index.py）。放在同一个实例里，过期、容量与备份只有一套策略。

调用方（main.py）只需要两个函数：
- build_context：回答之前取记忆，拼成一段文本交给 rag.get_result；
- after_turn：一轮结束后写记忆（压缩工作记忆 + 写情景记忆 + 更新画像）。

两个函数都逐层 try/except：任何一层失败都只打 warning，绝不影响回答返回。
没有 user_id 时整个记忆层跳过（评测脚本、旧前端、裸 curl 都走这条路）。
"""

from config import MEMORY_SNIPPET_CHARS
from observability import log_step, logger

from . import episodic, profile, working


def _snippet(text: str, limit: int = MEMORY_SNIPPET_CHARS) -> str:
    """单条记忆压缩成一行并截断，避免一条长回答把 prompt 挤爆。"""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[:limit] + "…"


def build_context(user_id: str, session_id: str, question: str) -> str:
    """拼出注入生成 prompt 的记忆段落；没有任何记忆时返回空串。"""
    user_id = (user_id or "").strip()
    if not user_id:
        return ""

    parts = []

    try:
        hits = episodic.search(user_id, question, exclude_session_id=session_id)
        if hits:
            lines = "\n".join(f"- {_snippet(doc.page_content)}" for doc in hits)
            parts.append(f"【情景记忆】用户在过去其它会话中的相关对话：\n{lines}")
    except Exception as exc:
        logger.warning(
            "memory.episodic_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )

    try:
        facts = profile.search(user_id, question)
        if facts:
            lines = "\n".join(f"- {_snippet(doc.page_content)}" for doc in facts)
            parts.append(f"【用户画像】已知关于该用户的长期信息：\n{lines}")
    except Exception as exc:
        logger.warning(
            "memory.profile_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )

    return "\n\n".join(parts)


def after_turn(user_id: str, session_id: str, question: str, answer: str) -> None:
    """一轮对话结束后更新三层记忆。整体不抛异常。"""
    with log_step("memory.after_turn", session_id=session_id) as span:
        compressed = False
        try:
            compressed = working.compress_if_needed(session_id)
        except Exception as exc:
            logger.warning(
                "memory.compress_failed",
                extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
            )

        user_id = (user_id or "").strip()
        if user_id:
            try:
                episodic.add_turn(user_id, session_id, question, answer)
            except Exception as exc:
                logger.warning(
                    "memory.episodic_write_failed",
                    extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
                )

            try:
                profile.extract_and_store(user_id, question, answer)
            except Exception as exc:
                logger.warning(
                    "memory.profile_write_failed",
                    extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
                )

        span.output = {"compressed": compressed}