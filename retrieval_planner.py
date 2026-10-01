"""检索策略的 LLM 结构化决策：通道规划 + 充分性判断。

与 collection_router.py 同属「LLM 结构化输出」范式：用 with_structured_output(Pydantic)
把模型输出约束成布尔值，避免再去解析自由文本。

- plan_retrieval_channels：这次提问该走知识库、走网络、还是都走（或都不走）。
  取代原先由前端两个勾选框硬控 `use_web_search` / `use_knowledge_base` 的做法。
- judge_retrieval_sufficiency：已检索到的信息够不够；不够时给出新的检索语句，
  供 rag.py 里的补充检索循环再用一轮。
"""

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from config import get_chat_model
from observability import log_step, logger


class SearchPlan(BaseModel):
    # 刻意给 default=True：模型漏字段时按“开启”处理（fail-open），
    # 只会多检索一次，不会静默把通道关掉。
    use_knowledge_base: bool = Field(
        default=True,
        description=(
            "是否需要检索用户的私人知识库。知识库内容为：用户的自我介绍、"
            "他对某些音乐的私人分析、他的歌单。问题涉及这些私人信息时为 true。"
        ),
    )
    use_web_search: bool = Field(
        default=True,
        description=(
            "是否需要联网搜索。问题涉及实时信息、外部公开事实、"
            "知识库之外的音乐作品资料时为 true。"
        ),
    )


def plan_retrieval_channels(question: str) -> SearchPlan:
    """判断本次提问需要走哪些检索通道。

    规划模型不可用时兜底为「两者都开」，与改造前的默认行为一致——
    宁可多检索一次，也不要静默给出无凭据的答案。
    """
    llm = get_chat_model()
    executor = llm.with_structured_output(SearchPlan)
    prompt = """
你是一个检索通道规划器。请判断回答用户提问需要用到哪些检索通道。

可用通道：
- 知识库（knowledge_base）：该用户的私人资料，包括他的自我介绍、他对某些音乐的私人分析、他的歌单。
  问题涉及“我/用户自己的”偏好、经历、私人观点时必须开启。
- 网络搜索（web_search）：公开的外部信息，包括实时新闻、音乐作品的客观资料、人物生平、乐理常识等。

判断规则：
1. 只问用户私人信息 → 只开知识库；
2. 只问外部公开信息 → 只开网络搜索；
3. 既涉及用户私人信息、又需要外部资料补充 → 两者都开；
4. 纯寒暄、纯推理、纯文字创作等不需要任何外部信息 → 两者都关。

用户提问：{question}
请只输出两个布尔字段的取值，不要包含任何解释性文本。
"""
    messages = ChatPromptTemplate.from_template(prompt).format_messages(question=question)

    with log_step("retrieval_planner", input=question) as span:
        try:
            result = executor.invoke(messages)
        except Exception as exc:
            # 规划失败不该打死整条问答：退回“两者都开”，与改造前的默认一致
            logger.warning(
                "retrieval_planner.failed",
                extra={
                    "fields": {"error": f"{type(exc).__name__}: {exc}"[:300]},
                },
            )
            result = SearchPlan()
        span.output = {
            "use_knowledge_base": result.use_knowledge_base,
            "use_web_search": result.use_web_search,
        }
    return result


class RetrievalAssessment(BaseModel):
    # 缺省为 False（= 判定不足）方向偏保守：配合调用方的硬轮数上限与语句去重，
    # 最坏也只是多检索一轮，不会失控。
    is_sufficient: bool = Field(
        default=False,
        description="已检索到的信息是否足以完整、准确地回答用户提问。",
    )
    new_query: str = Field(
        default="",
        description=(
            "当 is_sufficient 为 false 时，给出一条与已用语句不同、更聚焦的补充检索语句；"
            "is_sufficient 为 true 时留空。"
        ),
    )


def judge_retrieval_sufficiency(
    question: str,
    kb_text: str,
    web_text: str,
    channel: str,
    round_index: int,
) -> RetrievalAssessment:
    """判断已检索到的信息是否足够；不够时给出新的检索语句。

    异常不在这里吞掉：调用方（补充检索循环）会捕获后 break，
    这样 `step.error` 仍会留在链路追踪里，而整条问答照常出结果。
    """
    llm = get_chat_model()
    executor = llm.with_structured_output(RetrievalAssessment)
    prompt = """
你是一个检索充分性审查员。请判断下列已检索到的信息，是否足以准确回答用户提问。

用户提问：{question}
本次使用的检索通道：{channel}
已检索到的知识库信息：{kb_text}
已检索到的网络信息：{web_text}

判断规则：
1. 只有当信息能覆盖提问的关键点（关键实体、关键限定条件）时，才判定为充分；
2. 若不足以回答，请给出一条新的检索语句：换角度、更聚焦或更具体，
   不得与用户原问题重复，也不要只是原问题的同义改写；
3. 若信息已经足够，把 new_query 留空。

这是第 {round_index} 轮补充检索，请优先给出最能补齐信息缺口的那一条语句。
请只输出 is_sufficient 与 new_query 两个字段，不要包含任何解释性文本。
"""
    messages = ChatPromptTemplate.from_template(prompt).format_messages(
        question=question,
        channel=channel,
        kb_text=kb_text,
        web_text=web_text,
        round_index=round_index,
    )

    with log_step(
        "retrieval_sufficiency",
        input={"question": question, "channel": channel},
        round_index=round_index,
    ) as span:
        result = executor.invoke(messages)
        span.output = {
            "is_sufficient": result.is_sufficient,
            "new_query": result.new_query,
        }
    return result