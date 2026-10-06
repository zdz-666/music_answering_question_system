"""上传任务消费者（独立进程）。

上传 PDF/DOCX 要逐张调 VLM 识图，几十张图的文档能跑几分钟。原先这些活全在 /api/upload
里同步做完，HTTP 连接得一直挂着；现在由本进程从 Redis 队列取任务慢慢跑，
接口立刻返回 task_id，前端轮询 /api/task/{task_id} 拿结果。

为什么单独一个进程、而不用 FastAPI 的 BackgroundTasks：BackgroundTasks 仍在同一个
进程里跑，照样占着线程池，API 进程重启任务就没了；独立 worker 可以和 API 分头重启，
任务留在 Redis 队列里不丢，也能按需多开几个进程一起消费。

用法：
    python worker.py           # 前台常驻，Ctrl+C 退出（队列里没消费完的任务还在）
    python worker.py --once    # 处理完当前队列里的任务就退出（调试、一次性补跑用）
"""

import argparse
import time
from pathlib import Path

import document_loader
import rag
import tasks
from observability import (
    log_step,
    logger,
    new_request_id,
    reset_request_id,
    set_request_id,
)
from session_store import init_db

# BLPOP 的阻塞上限：够长才不会空转轮询，够短才让 Ctrl+C 及时生效
_CLAIM_TIMEOUT = 5
# Redis 不可用时的重试间隔，避免连不上时空转打爆 CPU
_IDLE_SLEEP = 3


def process(task_id: str, payload: dict) -> dict:
    """跑一个上传任务，返回 {answer, session_id, timestamp}。异常交给调用方处理。"""
    filename = payload.get("filename") or "上传文件"
    with log_step("task.upload", task_id=task_id, file=filename) as span:
        result = rag.answer_with_file(
            filename=filename,
            content=Path(payload["file_path"]).read_bytes(),
            question=payload.get("question"),
            session_id=payload.get("session_id"),
            user_id=payload.get("user_id") or "",
        )
        span.output = {
            "session_id": result["session_id"],
            "chars": len(result["answer"]),
        }
    return result


def handle(task_id: str) -> None:
    """取任务记录 → 执行 → 把结果或错误写回任务记录，最后删掉暂存文件。"""
    record = tasks.load(task_id)
    if record is None:
        # 键已过期（入队太久）或被人为删掉，没有 payload 无法执行
        logger.warning("task.missing", extra={"fields": {"task_id": task_id}})
        return

    payload = record.get("payload") or {}
    # 接回入队时那个 HTTP 请求的 trace id：worker 是另一个进程，contextvar 不会自己传过来，
    # 不接回去的话这次上传的日志在这里就断成两截了。
    token = set_request_id(record.get("request_id") or new_request_id())
    tasks.mark_running(task_id)
    try:
        try:
            result = process(task_id, payload)
        except document_loader.UnsupportedFormatError as exc:
            # 用户选错文件类型，是可预期的输入问题，不算系统故障
            tasks.fail(task_id, str(exc))
            logger.warning(
                "task.unsupported",
                extra={"fields": {"task_id": task_id, "error": str(exc)[:200]}},
            )
        except Exception as exc:  # noqa: BLE001 - 任何失败都要落到任务状态里，别让前端一直转圈
            tasks.fail(task_id, f"文件处理失败: {exc}")
            logger.warning(
                "task.failed",
                extra={"fields": {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"[:300]}},
            )
        else:
            tasks.finish(task_id, result)
            logger.info(
                "task.done",
                extra={"fields": {"task_id": task_id, "chars": len(result["answer"])}},
            )
    finally:
        reset_request_id(token)
        tasks.discard_file(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="上传任务消费者")
    parser.add_argument(
        "--once", action="store_true", help="只处理当前队列里的任务，处理完就退出"
    )
    args = parser.parse_args()

    if not tasks.available():
        logger.warning(
            "worker.no_redis",
            extra={
                "fields": {
                    "detail": "拿不到 Redis 连接，任务进不了队列；上传接口此时走同步处理"
                }
            },
        )

    # 会话存储启动自检（幂等）：worker 会写会话消息，不能等 API 进程先连上 Redis
    init_db()
    logger.info("worker.start", extra={"fields": {"once": args.once}})

    while True:
        task_id = tasks.claim(timeout=_CLAIM_TIMEOUT)
        if task_id is None:
            if args.once:
                break
            if not tasks.available():
                time.sleep(_IDLE_SLEEP)
            continue
        handle(task_id)


if __name__ == "__main__":
    main()