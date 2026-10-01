"""链路追踪 + 结构化日志：JSON 行输出 + request_id 贯穿全链路 + 分步输入输出/耗时/成本。

RAG 主链路有 6 次 LLM 调用（查询重写、路由、网络查询重写、自反思、生成，
必要时还有评测）与 3 次外部服务调用（Tavily、SiliconFlow 重排、Chroma+BM25 检索），
只有逐步计时与逐步留痕才能定位瓶颈、复现问题。

追踪模型（span）：
- `request_id` 是一次请求的 trace id，由 main.py 的中间件设置一次，业务函数无需层层传参；
- 每个 `log_step` 生成一个 span，带自己的 `span_id` 与 `parent_span_id`，
  层级从嵌套关系自动推导（如 request → rag.total → generate），
  同一日志里出现多次的同名步骤（如多个集合各检索一次）也能靠 span_id 区分；
- 入参与出参都落在 `step.done` 行上（`step.start` 也带一份入参），
  因此单独一行就够用：`grep '"step":"generate"' | tail -1` 能一次看到
  这步吃了什么、吐出什么、花了多久、烧了多少 token；
- log_step 在进入时压入一个用量作用域，退出时把该步的 token 与成本汇总输出，
  并累加到外层步骤，因此嵌套调用（如 rag.total 包住各子步骤）也能拿到整链总量。

输入输出按 `TRACE_TEXT_LIMIT`（默认 500 字符）截断，超长值保留头尾各半并标注原长度，
避免检索文档与生成 prompt 把日志撑爆。用 trace_view.py 可以把一个 request_id 的
所有行还原成可读的调用树。
"""

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler

ROOT_LOGGER_NAME = "music_rag"

# 每段输入/输出预览的最大字符数；超长值保留头尾各半并标注原长度
TRACE_TEXT_LIMIT = int(os.getenv("TRACE_TEXT_LIMIT", "500"))

# 列表类输入/输出（检索命中的文档、路由出的集合名）最多预览几项
TRACE_LIST_ITEMS = 5

# 当前请求 id
_REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

# 当前步骤的用量作用域栈，栈顶即"当前步骤"
_USAGE_SCOPE: ContextVar[tuple] = ContextVar("usage_scope", default=())


class JsonFormatter(logging.Formatter):
    """每条日志一行 JSON，直接可被采集/检索。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created).astimezone().isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "request_id": get_request_id(),
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging() -> logging.Logger:
    """初始化日志器。重复调用不会重复挂 handler（uvicorn --reload 会重载模块）。"""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    # 不向 root 传播，避免被 uvicorn 的默认格式再打印一遍
    logger.propagate = False
    if not any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


logger = setup_logging()


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def set_request_id(request_id: str):
    """返回 token，供上下文退出时还原。"""
    return _REQUEST_ID.set(request_id)


def reset_request_id(token) -> None:
    _REQUEST_ID.reset(token)


def get_request_id() -> str:
    return _REQUEST_ID.get()


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _preview(value, limit: int | None = None):
    """把任意入参/出参压成可放进日志的预览值。

    字符串截断成「头 + 尾」，dict 逐字段截断（保证多字段入参每个字段都能看到），
    列表只留前 TRACE_LIST_ITEMS 项并标注剩余数量。
    """
    limit = TRACE_TEXT_LIMIT if limit is None else limit

    if isinstance(value, str):
        if len(value) <= limit:
            return value
        head = value[: limit // 2]
        tail = value[-(limit - limit // 2) :]
        return f"{head}…[中间省略 {len(value) - limit} 字符]…{tail}"

    if isinstance(value, dict):
        return {key: _preview(item, limit) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        items = [_preview(item, limit) for item in value[:TRACE_LIST_ITEMS]]
        if len(value) > TRACE_LIST_ITEMS:
            items.append(f"…[另有 {len(value) - TRACE_LIST_ITEMS} 项]")
        return items

    if value is None or isinstance(value, (int, float, bool)):
        return value

    # Document 之类的对象：取其文本内容，取不到再退回 repr
    text = getattr(value, "page_content", None)
    if isinstance(text, str):
        return _preview(text, limit)
    return _preview(str(value), limit)


def _text_len(value) -> int:
    """预览前的原始长度，用来判断内容是否被截断过。"""
    if isinstance(value, str):
        return len(value)
    try:
        return len(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return len(str(value))


def _io_fields(prefix: str, value) -> dict:
    """把一段入参/出参转成日志字段：`input` 是预览，`input_chars` 是原始长度。"""
    if value is None:
        return {}
    return {prefix: _preview(value), f"{prefix}_chars": _text_len(value)}


class Span:
    """一次步骤的句柄。

    `log_step(...) as span` 拿到它，把这一步真正吃进去、吐出来的东西挂上去，
    退出时随日志一起输出。入参也可以在进入时用 `log_step(step, input=...)` 直接给。
    """

    def __init__(self, step: str, span_id: str, parent_span_id: str | None):
        self.step = step
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.input = None
        self.output = None


# 需要在步骤之间累加的数值字段
_USAGE_KEYS = ("input_tokens", "output_tokens", "llm_calls", "cost_usd")


def _new_scope(step: str, span_id: str, parent_span_id: str | None) -> dict:
    scope = {key: 0 for key in _USAGE_KEYS}
    scope["cost_usd"] = 0.0
    scope["step"] = step
    scope["span_id"] = span_id
    scope["parent_span_id"] = parent_span_id
    return scope


def _scope_fields(scope: dict) -> dict:
    return {
        "input_tokens": scope["input_tokens"],
        "output_tokens": scope["output_tokens"],
        "llm_calls": scope["llm_calls"],
        "cost_usd": round(scope["cost_usd"], 6),
    }


@contextmanager
def log_step(step: str, input=None, **fields):
    """记录一步的开始与结束，并生成一个 span。

    入参可以在进入时用 `input=` 传入，也可以在块内挂到 `span.input`；
    出参只能在块内挂到 `span.output`（进入时还不知道结果）。
    两行日志都带 span_id / parent_span_id，`step.start` 带入参，
    `step.done` 带耗时 / token / 成本 / 入参 / 出参。
    """
    start = time.perf_counter()
    parent_scope = _USAGE_SCOPE.get()
    parent_span_id = parent_scope[-1]["span_id"] if parent_scope else None

    span = Span(step, uuid.uuid4().hex[:8], parent_span_id)
    span.input = input

    scope = _new_scope(step, span.span_id, parent_span_id)
    token = _USAGE_SCOPE.set(parent_scope + (scope,))

    span_fields = {
        "step": step,
        "span_id": span.span_id,
        "parent_span_id": parent_span_id,
    }
    logger.info(
        "step.start",
        extra={"fields": {**span_fields, **_io_fields("input", span.input), **fields}},
    )

    try:
        yield span
    except Exception:
        _USAGE_SCOPE.reset(token)
        logger.error(
            "step.error",
            extra={
                "fields": {
                    **span_fields,
                    "elapsed_ms": _elapsed_ms(start),
                    **_scope_fields(scope),
                    **_io_fields("input", span.input),
                    **_io_fields("output", span.output),
                    **fields,
                }
            },
            exc_info=True,
        )
        raise

    _USAGE_SCOPE.reset(token)
    # 把本步用量累加到外层步骤，便于拿到整条链路的总量
    parent = _USAGE_SCOPE.get()
    if parent:
        outer = parent[-1]
        for key in _USAGE_KEYS:
            outer[key] += scope[key]

    logger.info(
        "step.done",
        extra={
            "fields": {
                **span_fields,
                "elapsed_ms": _elapsed_ms(start),
                **_scope_fields(scope),
                **_io_fields("input", span.input),
                **_io_fields("output", span.output),
                **fields,
            }
        },
    )


def _extract_usage(response) -> tuple:
    """从 LLMResult 里取 token 用量，兼容新旧两种返回结构。"""
    try:
        message = response.generations[0][0].message
        usage = getattr(message, "usage_metadata", None)
        if usage:
            return (
                usage.get("input_tokens") or 0,
                usage.get("output_tokens") or 0,
                (getattr(message, "response_metadata", None) or {}).get("model_name"),
            )
    except (AttributeError, IndexError, TypeError):
        pass

    # 部分 OpenAI 兼容网关只填 llm_output.token_usage
    llm_output = getattr(response, "llm_output", None) or {}
    usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    if usage:
        return (
            usage.get("prompt_tokens") or 0,
            usage.get("completion_tokens") or 0,
            llm_output.get("model_name"),
        )

    return 0, 0, llm_output.get("model_name")


class TokenUsageCallback(BaseCallbackHandler):
    """挂在 ChatOpenAI 上，每次调用结束后记录 token 用量与折算成本。

    单价由 config.py 传入（每百万 token），未配置时为 0，只记 token 数。
    """

    def __init__(self, input_price_per_mtok: float = 0.0, output_price_per_mtok: float = 0.0):
        self.input_price_per_mtok = input_price_per_mtok
        self.output_price_per_mtok = output_price_per_mtok

    def on_llm_end(self, response, **kwargs) -> None:
        input_tokens, output_tokens, model = _extract_usage(response)
        cost = (
            input_tokens / 1_000_000 * self.input_price_per_mtok
            + output_tokens / 1_000_000 * self.output_price_per_mtok
        )

        scope = _USAGE_SCOPE.get()
        current_step = None
        span_id = None
        if scope:
            top = scope[-1]
            current_step = top["step"]
            span_id = top["span_id"]
            top["input_tokens"] += input_tokens
            top["output_tokens"] += output_tokens
            top["llm_calls"] += 1
            top["cost_usd"] += cost

        logger.info(
            "llm.usage",
            extra={
                "fields": {
                    "step": current_step,
                    "span_id": span_id,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": round(cost, 6),
                }
            },
        )