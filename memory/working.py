"""第一层记忆：工作记忆的摘要压缩。

当前会话的消息由 session_store 存在 SQLite 里。条数超过 COMPRESS_TRIGGER 时，
把最旧一批消息交给 LLM 合并进已有摘要，然后删除这批消息，让 messages 表始终只剩
最近 HISTORY_WINDOW 条 —— 正好与注入 prompt 的窗口对齐，不会出现"摘要与最近几条
之间漏掉一段"的空洞。

压缩走两阶段，LLM 调用绝不放在事务里：
    claim_compression  →  （事务外调 LLM）  →  commit_compression
第二阶段的 commit 在同一个事务内写摘要 + 按 id 删除被覆盖的消息 + 释放认领，
任一步失败都会 release 认领，下一轮对话再重试。
"""

from langchain_core.prompts import ChatPromptTemplate

from config import get_chat_model
from observability import log_step, logger
from session_store import (
    COMPRESS_TRIGGER,
    HISTORY_WINDOW,
    claim_compression,
    commit_compression,
    count_messages,
    get_summary,
    list_oldest_messages,
    release_compression,
)

# 把"已有摘要 + 新增对话"合并成一段滚动摘要
SUMMARY_PROMPT = """你是对话摘要器。请把【已有摘要】与【新增对话】合并成一段简洁的中文摘要，
保留用户的目标、关注点、已确认的事实与结论，去掉寒暄与重复内容。
已有摘要：
{summary}
新增对话：
{transcript}
只输出摘要正文，不要任何前后缀。"""


def _transcript(messages: list[dict]) -> str:
    """把一批消息按时间正序拼成可读文本。"""
    lines = []
    for message in messages:
        speaker = "用户" if message["role"] == "user" else "助手"
        lines.append(f"{speaker}: {message['content']}")
    return "\n".join(lines)


def compress_if_needed(session_id: str) -> bool:
    """消息超阈值时压缩一次；返回是否真的压缩了。任何失败都只告警，不抛异常。"""
    try:
        count = count_messages(session_id)
    except Exception as exc:
        logger.warning(
            "memory.compress_failed",
            extra={"fields": {"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        return False

    if count <= COMPRESS_TRIGGER:
        return False

    # 抢占压缩权：同一会话同一时刻只允许一个 worker 压缩
    if not claim_compression(session_id):
        return False

    try:
        batch = list_oldest_messages(session_id, count - HISTORY_WINDOW)
        if not batch:
            release_compression(session_id)
            return False

        messages = ChatPromptTemplate.from_template(SUMMARY_PROMPT).format_messages(
            summary=get_summary(session_id) or "（无）",
            transcript=_transcript(batch),
        )

        llm = get_chat_model()
        with log_step(
            "memory.compress",
            input={"session_id": session_id, "messages": len(batch)},
        ) as span:
            summary = (llm.invoke(messages).content or "").strip()
            span.output = summary

        if not summary:
            release_compression(session_id)
            return False

        commit_compression(session_id, summary, batch[-1]["id"])
        return True
    except Exception as exc:
        logger.warning(
            "memory.compress_failed",
            extra={"fields": {"session_id": session_id, "error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        release_compression(session_id)
        return False