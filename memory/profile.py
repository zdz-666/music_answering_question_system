"""第三层记忆：用户画像（长期偏好与实体），同样用 Chroma 存储。

每一轮对话结束后由 LLM 抽取"长期稳定"的信息（身份/领域、音乐风格与歌手偏好、
乐器、长期目标等），一条事实一条文档。临时信息（本次具体问题、一次性请求、寒暄）
不抽取。

抽取用 with_structured_output 约束成列表，失败时 fail-open 为空列表，不影响主链路。
文档 id = md5(user_id + 归一化后的事实文本)，同一事实重复出现会 upsert 覆盖而不是
新增，天然去重。
"""

import hashlib
from datetime import datetime
from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field

from config import PROFILE_COLLECTION, PROFILE_TOP_K, get_chat_model
from data_storage import load_vector_store
from observability import log_step, logger

EXTRACT_PROMPT = """你负责维护用户画像。请从下面这一轮对话中抽取关于该用户的【长期稳定】信息。

只抽取这类内容：
- 身份、职业、所在领域或研究方向
- 音乐相关的长期偏好：喜欢的风格、歌手/乐队、乐器、常听的歌单类型
- 明确的长期目标或计划

不要抽取这类内容：本次提问的具体问题、一次性的请求或任务、寒暄与客套话。
同一事实不要拆成多条重复表述；没有值得长期记住的信息时返回空列表。

用户问题：{question}
助手回答：{answer}"""


class ProfileFacts(BaseModel):
    """一轮对话里抽取出的用户长期信息，一条一个原子事实。"""

    facts: List[str] = Field(
        default_factory=list,
        description="关于该用户的长期稳定信息，每条一句完整的中文陈述；没有则为空列表",
    )


def _doc_id(user_id: str, fact: str) -> str:
    return hashlib.md5(f"{user_id}|{fact}".encode("utf-8")).hexdigest()


def extract_and_store(user_id: str, question: str, answer: str) -> int:
    """抽取并写入画像事实，返回写入条数（同一事实重复出现不会新增）。"""
    if not user_id:
        return 0

    executor = get_chat_model().with_structured_output(ProfileFacts)
    prompt_messages = EXTRACT_PROMPT.format(question=question, answer=answer)

    with log_step("memory.profile_extract", input={"question": question}) as span:
        try:
            result = executor.invoke(prompt_messages)
        except Exception as exc:
            # fail-open：抽取失败只是这一轮没更新画像，不影响回答
            logger.warning(
                "memory.profile_extract_failed",
                extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
            )
            result = ProfileFacts()
        span.output = result.facts

    now = datetime.now().isoformat()
    docs, ids = [], []
    for fact in result.facts:
        # 归一化空白，让措辞相同的事实落到同一个 id
        fact = " ".join((fact or "").split())
        if not fact:
            continue
        ids.append(_doc_id(user_id, fact))
        docs.append(
            Document(page_content=fact, metadata={"user_id": user_id, "timestamp": now})
        )

    if not docs:
        return 0

    load_vector_store(PROFILE_COLLECTION).add_documents(docs, ids=ids)
    return len(docs)


def search(user_id: str, question: str, k: int = PROFILE_TOP_K) -> list[Document]:
    """按语义相似度取该用户的画像事实。"""
    if not user_id:
        return []
    return load_vector_store(PROFILE_COLLECTION).similarity_search(
        question, k=k, filter={"user_id": user_id}
    )