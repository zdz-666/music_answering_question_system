"""把一个 request_id 的结构化日志还原成一条调用树。

日志是扁平的 JSON 行，每个 step 只知道自己的 span_id 与 parent_span_id；
本脚本按这两个字段把「改写 → 路由 → 检索 → 重排 → 生成」重新拼成树，
并把每步的耗时、token、成本、输入输出放在一起。

用法：
    # 实时看（另开一个窗口发请求）
    python main.py 2>&1 | python trace_view.py

    # 看日志文件里最后一条完整链路
    python trace_view.py app.log

    # 指定 request_id（支持子串，形如 3f2a1c8e）
    python trace_view.py app.log --request-id 3f2a1c8e

    # 列出日志里所有请求
    python trace_view.py app.log --list

    # 关掉输入输出，只看耗时与成本
    python trace_view.py app.log --no-io
"""

import argparse
import json
import re
import sys
from contextlib import ExitStack

# 这些字段由渲染逻辑单独处理，不再当作"步骤附加信息"重复展示
PLUMBING = {
    "event",
    "ts",
    "level",
    "logger",
    "request_id",
    "step",
    "span_id",
    "parent_span_id",
    "elapsed_ms",
    "input_tokens",
    "output_tokens",
    "llm_calls",
    "cost_usd",
    "input",
    "output",
    "input_chars",
    "output_chars",
}

STEP_EVENTS = ("step.start", "step.done", "step.error")


class Span:
    """一步的追踪数据，聚合了它的 step.start / step.done / llm.usage 三类日志行。"""

    __slots__ = (
        "step",
        "span_id",
        "parent_span_id",
        "elapsed_ms",
        "input_tokens",
        "output_tokens",
        "llm_calls",
        "cost_usd",
        "input",
        "output",
        "input_chars",
        "output_chars",
        "extra",
        "error",
        "llm",
        "children",
    )

    def __init__(self, step, span_id, parent_span_id):
        self.step = step
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.elapsed_ms = None
        self.input_tokens = 0
        self.output_tokens = 0
        self.llm_calls = 0
        self.cost_usd = 0.0
        self.input = None
        self.output = None
        self.input_chars = None
        self.output_chars = None
        self.extra = {}
        self.error = None
        self.llm = []
        self.children = []

    def take_extra(self, record) -> None:
        for key, value in record.items():
            if key not in PLUMBING:
                self.extra[key] = value


def read_records(paths) -> list:
    """读入所有 JSON 日志行。

    从第一个 `{` 开始解析，这样能容忍行首的各种前缀：PowerShell 给原生命令的
    stderr 加的 `python.exe : `、容器日志的时间戳、uvicorn 的纯文本 access log 等。
    """
    records = []
    with ExitStack() as stack:
        streams = (
            [stack.enter_context(open(path, encoding="utf-8")) for path in paths]
            if paths
            else [sys.stdin]
        )
        for stream in streams:
            for line in stream:
                start = line.find("{")
                if start == -1:
                    continue
                try:
                    records.append(json.loads(line[start:]))
                except ValueError:
                    continue
    return records


def request_ids(records) -> list:
    """按出现顺序收集所有 request_id。"""
    seen = []
    for record in records:
        rid = record.get("request_id")
        if rid and rid not in seen:
            seen.append(rid)
    return seen


def pick_request_id(records, wanted, all_ids) -> str:
    if wanted:
        if wanted in all_ids:
            return wanted
        matched = [rid for rid in all_ids if wanted in rid]
        if len(matched) == 1:
            return matched[0]
        if not matched:
            raise SystemExit(f"日志里没有匹配 {wanted!r} 的请求，共 {len(all_ids)} 个")
        raise SystemExit(
            f"{wanted!r} 匹配到多个请求，请写全：\n  " + "\n  ".join(matched)
        )

    finished = set()
    for record in records:
        if record.get("event") == "step.done" and record.get("step") == "request":
            finished.add(record.get("request_id"))

    # 优先给最后一条"已结束"的链路；日志末尾那条可能是还在跑的请求，展开也没内容
    for record in reversed(records):
        if (
            record.get("event") == "step.start"
            and record.get("step") == "request"
            and record.get("request_id") in finished
        ):
            return record["request_id"]

    if all_ids:
        return all_ids[-1]
    raise SystemExit("日志里没有任何追踪数据（是不是没配 LOG_LEVEL=INFO？）")


def build_spans(records, request_id) -> list:
    spans = {}
    order = []

    for record in records:
        if record.get("request_id") != request_id:
            continue

        span_id = record.get("span_id")
        if not span_id:
            continue

        span = spans.get(span_id)
        if span is None:
            span = Span(record.get("step"), span_id, record.get("parent_span_id"))
            spans[span_id] = span
            order.append(span_id)

        event = record.get("event")
        if event == "llm.usage":
            span.llm.append(record)
        elif event == "step.start":
            span.input = record.get("input")
            span.input_chars = record.get("input_chars")
            span.take_extra(record)
        elif event == "step.done":
            span.elapsed_ms = record.get("elapsed_ms")
            span.input_tokens = record.get("input_tokens", 0)
            span.output_tokens = record.get("output_tokens", 0)
            span.llm_calls = record.get("llm_calls", 0)
            span.cost_usd = record.get("cost_usd", 0.0)
            span.output = record.get("output")
            span.output_chars = record.get("output_chars")
            if record.get("input") is not None:
                span.input = record["input"]
                span.input_chars = record.get("input_chars")
            span.take_extra(record)
        elif event == "step.error":
            span.elapsed_ms = record.get("elapsed_ms")
            span.error = record.get("exception") or record.get("error")

    roots = []
    for span_id in order:
        span = spans[span_id]
        parent = spans.get(span.parent_span_id) if span.parent_span_id else None
        if parent is None:
            roots.append(span)
        else:
            parent.children.append(span)
    return roots


def _flatten(value) -> str:
    """预览里的换行会把树压垮，这里压成一行。"""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return re.sub(r"\s+", " ", text).strip()


def _header(span) -> str:
    parts = [f"{span.elapsed_ms:.1f}ms" if span.elapsed_ms is not None else "未结束"]
    if span.llm_calls:
        parts.append(f"{span.input_tokens}/{span.output_tokens} tok")
        parts.append(f"${span.cost_usd:.6f}")
        parts.append(f"{span.llm_calls} llm")
    parts.extend(f"{key}={_flatten(value)}" for key, value in span.extra.items())
    if span.error:
        parts.append("✗ 失败")
    return "  ".join(parts)


def render(roots, show_io=True) -> list:
    lines = []
    for root in roots:
        _render_one(root, "", True, lines, show_io, top=True)
    return lines


def _render_one(span, prefix, is_last, lines, show_io, top=False) -> None:
    """根节点顶格，其余节点带连接线；子节点与附属行统一挂在连接线右侧。"""
    connector = "" if top else ("└─ " if is_last else "├─ ")
    lines.append(f"{prefix}{connector}{span.step}  {_header(span)}")

    nested = prefix + ("   " if top or is_last else "│  ")

    if show_io:
        for label, value, chars in (
            ("in ", span.input, span.input_chars),
            ("out", span.output, span.output_chars),
        ):
            if value is None:
                continue
            size = f"[{chars}字符] " if chars else ""
            lines.append(f"{nested}   {label}{size}: {_flatten(value)}")

    for record in span.llm:
        model = record.get("model") or "?"
        lines.append(
            f"{nested}   llm: {model}  "
            f"{record.get('input_tokens')}/{record.get('output_tokens')} tok  "
            f"${record.get('cost_usd', 0):.6f}"
        )

    if span.error:
        for error_line in str(span.error).splitlines():
            lines.append(f"{nested}   ! {error_line}")

    for index, child in enumerate(span.children):
        _render_one(child, nested, index == len(span.children) - 1, lines, show_io)


def list_requests(records) -> list:
    """概览：每个请求一行，显示入口、耗时与总成本。"""
    requests = {}
    for record in records:
        rid = record.get("request_id")
        if not rid:
            continue
        entry = requests.setdefault(
            rid, {"path": "-", "elapsed_ms": None, "cost_usd": 0.0, "step": None}
        )
        if record.get("event") == "step.start" and record.get("step") == "request":
            entry["path"] = f"{record.get('method', '')} {record.get('path', '')}".strip()
        if record.get("step") == "rag.total":
            entry["step"] = record.get("step")
        if record.get("event") == "step.done" and record.get("step") == "request":
            entry["elapsed_ms"] = record.get("elapsed_ms")
        if record.get("event") == "step.done" and record.get("step") == "rag.total":
            entry["cost_usd"] = record.get("cost_usd", 0.0)

    lines = []
    for rid, entry in requests.items():
        elapsed = f"{entry['elapsed_ms']:.1f}ms" if entry["elapsed_ms"] is not None else "未结束"
        lines.append(f"{rid}  {elapsed:>10}  ${entry['cost_usd']:.6f}  {entry['path']}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把结构化日志里的一个 request_id 还原成调用树"
    )
    parser.add_argument("logs", nargs="*", help="日志文件；不传则从 stdin 读")
    parser.add_argument("--request-id", help="目标 request_id（支持子串）")
    parser.add_argument("--list", action="store_true", help="只列出所有请求，不展开")
    parser.add_argument("--no-io", action="store_true", help="不显示步骤的输入输出")
    args = parser.parse_args()

    records = read_records(args.logs)
    all_ids = request_ids(records)
    if not all_ids:
        raise SystemExit("日志里没有任何追踪数据（是不是没配 LOG_LEVEL=INFO？）")

    if args.list:
        print(f"共 {len(all_ids)} 个请求：")
        for line in list_requests(records):
            print("  " + line)
        return

    request_id = pick_request_id(records, args.request_id, all_ids)
    roots = build_spans(records, request_id)

    print(f"request_id = {request_id}")
    for line in render(roots, show_io=not args.no_io):
        print(line)


if __name__ == "__main__":
    main()