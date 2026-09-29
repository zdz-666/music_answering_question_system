"""结构化日志：JSON 行输出 + request_id 贯穿全链路 + 分步耗时 + token 成本。

RAG 主链路有 6 次 LLM 调用（查询重写、路由、网络查询重写、自反思、生成，
必要时还有评测）与 3 次外部服务调用（Tavily、SiliconFlow 重排、Chroma+BM25 检索），
只有逐步计时才能定位瓶颈。

设计要点：
- request_id 存在 contextvar 里，由 main.py 的中间件设置一次，业务函数无需层层传参；
- log_step 在进入时压入一个用量作用域，退出时把该步的 token 与成本汇总输出，
  并累加到外层步骤，因此嵌套调用（如 rag.total 包住各子步骤）也能拿到整链总量。
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


# 需要在步骤之间累加的数值字段
_USAGE_KEYS = ("input_tokens", "output_tokens", "llm_calls", "cost_usd")


def _new_scope(step: str) -> dict:
    scope = {key: 0 for key in _USAGE_KEYS}
    scope["cost_usd"] = 0.0
    scope["step"] = step
    return scope


def _scope_fields(scope: dict) -> dict:
    return {
        "input_tokens": scope["input_tokens"],
        "output_tokens": scope["output_tokens"],
        "llm_calls": scope["llm_calls"],
        "cost_usd": round(scope["cost_usd"], 6),
    }


@contextmanager
def log_step(step: str, **fields):
    """记录一步的开始与结束；结束时带上 elapsed_ms 与该步的 token 用量。"""
    start = time.perf_counter()
    scope = _new_scope(step)
    token = _USAGE_SCOPE.set(_USAGE_SCOPE.get() + (scope,))
    logger.info("step.start", extra={"fields": {"step": step, **fields}})

    try:
        yield
    except Exception:
        _USAGE_SCOPE.reset(token)
        logger.error(
            "step.error",
            extra={
                "fields": {
                    "step": step,
                    "elapsed_ms": _elapsed_ms(start),
                    **_scope_fields(scope),
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
                "step": step,
                "elapsed_ms": _elapsed_ms(start),
                **_scope_fields(scope),
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
        if scope:
            top = scope[-1]
            current_step = top["step"]
            top["input_tokens"] += input_tokens
            top["output_tokens"] += output_tokens
            top["llm_calls"] += 1
            top["cost_usd"] += cost

        logger.info(
            "llm.usage",
            extra={
                "fields": {
                    "step": current_step,
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "cost_usd": round(cost, 6),
                }
            },
        )