"""记忆召回：回答之前从该用户的记忆目录里挑出与当前提问相关的记录。

流程（参考 memory_example.select_relevant_memories / load_memories）：
1. 取最近的用户提问作为查询；
2. 把目录索引（name + description）连同查询交给 LLM，返回一个 JSON 下标数组；
3. LLM 调用失败（网关抖动、返回不是 JSON）时退化为关键词匹配；
4. 按选中的文件名读正文，拼成 JSON 交给 system prompt，总字符受 RECALL_CHAR_LIMIT 约束。

这里不走向量检索：目录索引本身就是「先粗筛」的一层，具体正文按需读入，
比把整库向量化更省成本，也和 memory_example 的行为一致。
"""

import json
import re

from config import MEMORY_RECALL_CHAR_LIMIT, MEMORY_RECALL_TOP_K, get_chat_model
from observability import log_step, logger

from . import store


def extract_json_array(text: str) -> list:
    """从一段可能夹杂说明文字的回复里抠出第一个合法 JSON 数组。"""
    decoder = json.JSONDecoder()
    for position, character in enumerate(text):
        if character != "[":
            continue
        try:
            value, _ = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            return value
    return []


def message_text(message) -> str:
    """取一条消息的文本；兼容 dict（OpenAI 格式）与带 content 的对象。"""
    content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # 多模态/分块内容：只挑 text 块拼接
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in (None, "text"):
                    parts.append(str(block.get("text", "")))
            else:
                text = getattr(block, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(filter(None, parts))
    return ""


def recent_user_text(messages: list, max_turns: int = 3) -> str:
    """最近 max_turns 条用户提问拼成的查询文本（倒序取完再正序拼）。"""
    turns = []
    for message in reversed(messages):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "type", None)
        if role not in ("user", "human"):
            continue
        text = message_text(message).strip()
        if text:
            turns.append(text)
        if len(turns) == max_turns:
            break
    return "\n".join(reversed(turns))[:4000]


def keyword_memory_selection(records: list[dict], query: str, max_items: int) -> list[str]:
    """关键词回退：按查询词在 name+description 里的命中数排序。"""
    words = set(re.findall(r"[a-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", query.lower()))
    ranked = []
    for record in records:
        catalog_text = f"{record['name']} {record['description']}".lower()
        score = sum(word in catalog_text for word in words)
        if score:
            ranked.append((score, record["filename"]))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [filename for _, filename in ranked[:max_items]]


def select_relevant_memories(
    user_id: str, messages: list, max_items: int = MEMORY_RECALL_TOP_K
) -> list[str]:
    """用 LLM 从目录索引里挑出相关记忆的文件名；失败回退关键词匹配。"""
    records = store.list_memory_files(user_id)
    query = recent_user_text(messages)
    if not records or not query:
        return []

    catalog = "\n".join(
        f"{index}: {' '.join(record['name'].split())} - "
        f"{' '.join(record['description'].split())}"
        for index, record in enumerate(records)
    )
    prompt = (
        "从下面的记忆目录中选出与用户当前请求相关的记录。"
        "只返回一个 JSON 数组，元素是目录下标，例如 [0, 2]；都不相关时返回 []。\n\n"
        f"当前请求：\n{query}\n\n记忆目录：\n{catalog[:12000]}"
    )

    try:
        llm = get_chat_model(temperature=0)
        with log_step(
            "memory.recall_select", user_id=user_id, records=len(records)
        ) as span:
            content = (llm.invoke(prompt).content or "").strip()
            span.output = content
        indices = extract_json_array(content)
        selected = []
        for index in indices:
            if isinstance(index, int) and 0 <= index < len(records):
                filename = records[index]["filename"]
                if filename not in selected:
                    selected.append(filename)
                if len(selected) == max_items:
                    break
        return selected
    except Exception as exc:  # noqa: BLE001 - 召回失败退化为关键词，绝不影响回答
        logger.warning(
            "memory.recall_select_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        return keyword_memory_selection(records, query, max_items)


def load_memories(user_id: str, messages: list) -> str:
    """读回选中的记忆正文，拼成 JSON 字符串（无记忆时返回空串）。"""
    loaded = []
    remaining = MEMORY_RECALL_CHAR_LIMIT
    for filename in select_relevant_memories(user_id, messages):
        content = store.read_memory_file(user_id, filename)
        if not content or remaining <= 0:
            continue
        recalled = content[:remaining]
        loaded.append({"source": filename, "content": recalled})
        remaining -= len(recalled)
    return json.dumps(loaded, ensure_ascii=False, indent=2) if loaded else ""