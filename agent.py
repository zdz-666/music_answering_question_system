"""问答智能体：把原来的「单次检索 + 生成」改造成 agent_loop（参考 memory_example.py）。

一轮对话的流程：

    system = 音乐助手指令 + 记忆片段（memory.system_section）
    messages = 会话历史 + 本轮用户提问
    while True:
        messages = ContextCompactor.prepare(messages)   # 每轮调用模型前先瘦身
        response = llm.bind_tools(TOOLS).invoke([system] + messages)
        有工具调用？ → 执行工具、把结果作为 tool 消息接回、继续
        没有         → 这就是最终答案，结束
    结束后：把压缩结果写回会话 + memory.learn 学习长期记忆

工具四个：
- knowledge_base_search：复用 rag 的集合路由 + 混合检索 + 重排 + 自反思前的原始 chunk；
- web_search：复用 rag 的联网搜索；
- read_file：读该用户的记忆文件，或读回上下文压缩落盘/归档的原始内容；
- compact：模型自己要一次上下文压缩。

消息全程用 OpenAI 格式的 dict（assistant.tool_calls / role="tool"），这样既能直接交给
LangChain 的 ChatOpenAI.invoke，又能让 ContextCompactor 像示例那样按统一结构处理。
system 不进 messages，每次 invoke 时前置，避免被 compact（整段历史被替换成摘要）时丢掉。

工具处理函数里 import rag 是有意的：rag 反过来要 import 本模块的 run，
放在函数内可以打断 module 级循环依赖。
"""

import json

from langchain_core.messages import AIMessage

from config import CACHE_TTL_ANSWER, get_chat_model
from context_compact import MAX_REACTIVE_RETRIES, ContextCompactor
from observability import log_step, logger
import cache
import file_access
import memory
import session_store

# 单轮请求里最多让模型「思考—调工具」几轮，防止工具调用停不下来
MAX_TOOL_ROUNDS = 8

SYSTEM = """你是一个专业的音乐知识问答助手，名为“乐典”。你的核心职责是准确、专业、清晰地回答用户关于音乐的一切问题。

你可以调用工具来获取外部信息：
- knowledge_base_search：检索本地知识库（音乐理解、歌单、自我介绍等）。当问题涉及具体曲目/作品/歌单/领域知识时使用。
- web_search：联网搜索最新信息。当问题涉及时效性内容、新闻、最新动态时使用。
- read_file：读取记忆文件或上下文压缩归档的内容。当系统提示里的记忆索引不够、需要看某条记忆的正文，或需要回读被压缩替换掉的结果时使用。
- compact：当对话已经很长、上下文快放不下时，可以主动请求压缩历史。
选择工具时给出简洁、精准的检索语句；不需要外部信息时直接凭自身知识作答，不要为了调用而调用。

组织与输出（必须严格遵守，前端会按 Markdown 渲染）：
- 始终使用 Markdown 格式组织回答，不要输出大段无结构的纯文本。
- 小节标题使用 ## 或 ###，不要使用一级标题 #。
- 并列内容用无序列表（- ），有先后顺序的内容用有序列表（1. 2. 3.）。
- 关键术语、曲名、作品名、人名使用 **加粗** 强调，但不要整段加粗。
- 需要横向对比多项内容时，使用 Markdown 表格呈现。
- 引用原文或他人观点时使用 > 引用块。
- 直接输出 Markdown 正文本身，严禁把整个回答包裹在 ``` 代码块中。"""


def _tools() -> list[dict]:
    """OpenAI 格式的工具定义。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "knowledge_base_search",
                "description": "在本地音乐知识库中检索与问题相关的资料（曲目、作品、歌单、领域知识）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "精炼后的检索语句"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "联网搜索最新或时效性强的信息（新闻、最新动态、实时资料）。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "精炼后的搜索关键词"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "读取一个文件的完整内容。可读：本用户的记忆文件（记忆目录下的文件名，"
                    "如 note.md）、以及上下文压缩占位文本里给出的归档路径。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "记忆目录下的文件名，或压缩占位文本里给出的绝对路径",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compact",
                "description": "当对话历史很长、上下文快放不下时，请求压缩历史以腾出空间。",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


TOOLS = _tools()


def _knowledge_base_search(query: str) -> str:
    """检索本地知识库：集合路由 → 混合检索 + 重排 → 文本块。"""
    import rag

    collections = rag.route_collections(query)
    chunks = rag.get_vector_search(query, collections)
    return rag.format_kb_chunks(chunks)


def _web_search(query: str) -> str:
    """联网搜索：改写查询 → Tavily → 带序号的文本块。"""
    import rag

    items = rag.get_web_search(query)
    return rag.format_web_items(items)


def _execute_tool(name: str, args: dict, user_id: str) -> str:
    """按名字分发工具；未知工具与异常都返回可读文本，不打断 loop。"""
    try:
        if name == "knowledge_base_search":
            return _knowledge_base_search(str(args.get("query", "")).strip())
        if name == "web_search":
            return _web_search(str(args.get("query", "")).strip())
        if name == "read_file":
            return file_access.read_file(str(args.get("path", "")).strip(), user_id)
    except Exception as exc:  # noqa: BLE001 - 工具失败要让模型知道，而不是整轮报错
        logger.warning(
            "agent.tool_failed",
            extra={"fields": {"tool": name, "error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        return f"工具 {name} 执行失败：{type(exc).__name__}: {exc}"
    return f"未知工具：{name}"


def _assistant_message(response: AIMessage) -> dict:
    """AIMessage → OpenAI 格式的 assistant 消息（含 tool_calls）。"""
    message = {"role": "assistant", "content": response.content or ""}
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ]
    return message


def _is_prompt_too_long(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "prompt_too_long",
            "too many tokens",
            "context length",
            "maximum context",
            "reduce the length",
        )
    )


def run(
    question: str,
    session_id: str,
    user_id: str = "",
    file_content: str | None = None,
) -> str:
    """跑完一轮问答，返回最终答案文本（Markdown）。

    同步阻塞函数：内部是同步 LLM / 检索 / 网络调用，调用方负责放进线程池。
    """
    context = session_store.load_context(session_id)
    history = context["messages"]
    max_id = context["max_id"]

    user_content = question
    if file_content:
        user_content = f"{question}\n\n用户上传的文件内容：\n{file_content}"
    messages = [*history, {"role": "user", "content": user_content}]

    memory_section = ""
    try:
        memory_section = memory.system_section(user_id, messages)
    except Exception as exc:  # noqa: BLE001 - 记忆失败不影响回答
        logger.warning(
            "agent.memory_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
    system = f"{SYSTEM}\n\n{memory_section}" if memory_section else SYSTEM

    # 只有「没有历史、没有长期记忆、没有上传文件」时，答案才只由 question 决定，
    # 这时按 question 缓存最终答案不会串味；否则同一问题在不同会话/画像下答案不同。
    cache_name = (
        cache.key("answer", cache.digest(question))
        if not history and not memory_section and not file_content
        else None
    )
    if cache_name:
        cached = cache.get_json(cache_name)
        if cached is not None:
            logger.info("answer.cached", extra={"fields": {"question": question}})
            return cached

    compactor = ContextCompactor(user_id)
    llm = get_chat_model().bind_tools(TOOLS)
    reactive_retries = 0
    answer = ""

    with log_step(
        "agent.run", input=question, session_id=session_id, history=len(history)
    ) as span:
        for _ in range(MAX_TOOL_ROUNDS):
            messages = compactor.prepare(messages, question)

            try:
                response = llm.invoke([{"role": "system", "content": system}, *messages])
                reactive_retries = 0
            except Exception as exc:  # noqa: BLE001 - 只有「上下文过长」才压缩重试
                if _is_prompt_too_long(exc) and reactive_retries < MAX_REACTIVE_RETRIES:
                    logger.info("agent.reactive_compact")
                    messages[:] = compactor.reactive_compact(messages, question)
                    reactive_retries += 1
                    continue
                raise

            messages.append(_assistant_message(response))

            if not response.tool_calls:
                answer = (response.content or "").strip()
                break

            compact_requested = False
            for call in response.tool_calls:
                if call["name"] == "compact":
                    output = "已请求在本批工具之后压缩上下文。"
                    compact_requested = True
                else:
                    with log_step("agent.tool", tool=call["name"]) as tool_span:
                        output = _execute_tool(
                            call["name"], call.get("args") or {}, user_id
                        )
                        tool_span.output = output
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": output}
                )

            if compact_requested:
                messages[:] = compactor.compact_history(messages, question)

        span.output = answer

    # 压缩发生了就把摘要写回会话，下一轮上下文更小
    if compactor.last_summary:
        try:
            session_store.save_compaction(session_id, compactor.last_summary, max_id)
        except Exception as exc:  # noqa: BLE001 - 写回失败不影响已经得到的答案
            logger.warning(
                "agent.save_compaction_failed",
                extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
            )

    # 一轮结束后学习长期记忆（抽取 + 必要时合并），失败不影响回答
    try:
        memory.learn(
            user_id,
            [
                {"role": "user", "content": question[:2000]},
                {"role": "assistant", "content": answer},
            ],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "agent.learn_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )

    if cache_name and answer:
        cache.set_json(cache_name, answer, CACHE_TTL_ANSWER)

    return answer