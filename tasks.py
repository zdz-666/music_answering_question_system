"""上传长任务队列（Redis 实现）。

上传 PDF/DOCX 要逐张调 VLM 识图，几十张图的文档能跑几分钟；原先这些活全在 /api/upload
里同步做完，HTTP 连接得一直挂着。现在改成：接口把文件与问题写进队列就立刻返回 task_id，
worker.py 从队列取出来慢慢跑，前端轮询 /api/task/{task_id} 拿结果。

为什么用 Redis 列表：`RPUSH` 入队 + `BLPOP` 出队就是最朴素的 FIFO，不用引额外的消息中间件；
任务记录另存一个 hash，带 TTL 自动回收，不必写清理任务。

几处刻意的取舍：
- **文件不塞进 Redis**：payload 里只放暂存文件的路径。几十 MB 的 PDF 塞进同一个 Redis 实例，
  会把缓存挤掉、AOF 重写变大，而且出问题时排查困难。
- **Redis 不可用时 submit() 返回 None**，调用方退回同步处理。队列是可选设施，
  不能让「没装 Redis」变成「上传不可用」——与缓存层同一条原则。
- **不持久化重试**：worker 崩了任务的状态会停在 running，靠 TTL 自然消失。
  个人项目不做任务重投递，省下的复杂度远比收益大。
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

import cache
from config import TASK_TTL_SECONDS, UPLOAD_SPOOL_DIR
from observability import logger

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"


def _now() -> str:
    return datetime.now().isoformat()


def _task_key(task_id: str) -> str:
    return cache.key("task", task_id)


def _queue_key() -> str:
    return cache.key("queue", "upload")


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _spool_path(task_id: str, filename: str | None) -> Path:
    """暂存文件路径：用 task_id 命名，只保留原扩展名。

    不直接用原文件名：里面可能带目录分隔符或超长字符，拼进路径会跑到目录外面去。
    扩展名要留着 —— document_loader 靠它分发解析器。
    """
    suffix = Path(filename or "").suffix.lower()
    return Path(UPLOAD_SPOOL_DIR) / f"{task_id}{suffix}"


def available() -> bool:
    """Redis 是否可用。worker 用它决定空转时要不要歇一下。"""
    return cache.get_client() is not None


def submit(
    filename: str,
    content: bytes,
    question: str | None,
    session_id: str | None,
    user_id: str,
    request_id: str,
) -> str | None:
    """把一次上传入队，返回 task_id；Redis 不可用时返回 None（调用方退回同步处理）。

    request_id 一起存进任务记录：worker 是另一个进程，把入队时的 trace id 接回去，
    这次上传的日志才能和最初那个 HTTP 请求串成一条链路。
    """
    client = cache.get_client()
    if client is None:
        return None

    task_id = uuid.uuid4().hex
    spool = _spool_path(task_id, filename)
    try:
        spool.parent.mkdir(parents=True, exist_ok=True)
        spool.write_bytes(content)
        client.hset(
            _task_key(task_id),
            mapping={
                "task_id": task_id,
                "status": STATUS_QUEUED,
                "created_at": _now(),
                "request_id": request_id or "-",
                "payload": json.dumps(
                    {
                        "file_path": str(spool),
                        "filename": filename,
                        "question": question,
                        "session_id": session_id,
                        "user_id": user_id,
                    },
                    ensure_ascii=False,
                ),
            },
        )
        client.expire(_task_key(task_id), TASK_TTL_SECONDS)
        client.rpush(_queue_key(), task_id)
    except Exception as exc:  # noqa: BLE001 - 队列不可用要退回同步，不能把上传打死
        logger.warning(
            "task.submit_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        discard_file({"file_path": str(spool)})
        return None

    logger.info(
        "task.queued",
        extra={
            "fields": {
                "task_id": task_id,
                "file": filename,
                "bytes": len(content),
                "has_question": bool(question),
            }
        },
    )
    return task_id


def claim(timeout: int = 5) -> str | None:
    """worker 侧：阻塞取一个 task_id；超时或 Redis 异常返回 None。"""
    client = cache.get_client()
    if client is None:
        return None
    try:
        item = client.blpop([_queue_key()], timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - worker 主循环负责歇一下再试
        logger.warning(
            "task.claim_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        return None
    if not item:
        return None
    return _decode(item[1])


def load(task_id: str) -> dict | None:
    """读任务记录（含 payload）。键不存在或已过期返回 None。"""
    client = cache.get_client()
    if client is None:
        return None
    try:
        raw = client.hgetall(_task_key(task_id))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "task.load_failed",
            extra={"fields": {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        return None
    if not raw:
        return None

    record = {_decode(k): _decode(v) for k, v in raw.items()}
    try:
        record["payload"] = json.loads(record.get("payload") or "{}")
        record["result"] = json.loads(record.get("result") or "{}")
    except ValueError:
        record["payload"] = {}
        record["result"] = {}
    # 结果字段摊平到顶层，轮询接口直接用
    record.update(record.pop("result"))
    return record


def _update(task_id: str, mapping: dict) -> None:
    """写任务状态。每次写入都续一轮 TTL：长任务跑完时键不能已经过期了。"""
    client = cache.get_client()
    if client is None:
        return
    try:
        client.hset(_task_key(task_id), mapping=mapping)
        client.expire(_task_key(task_id), TASK_TTL_SECONDS)
    except Exception as exc:  # noqa: BLE001 - 状态写不进去不该影响任务本身
        logger.warning(
            "task.update_failed",
            extra={"fields": {"task_id": task_id, "error": f"{type(exc).__name__}: {exc}"[:300]}},
        )


def mark_running(task_id: str) -> None:
    _update(task_id, {"status": STATUS_RUNNING, "started_at": _now()})


def finish(task_id: str, result: dict) -> None:
    _update(
        task_id,
        {
            "status": STATUS_DONE,
            "finished_at": _now(),
            "result": json.dumps(result, ensure_ascii=False),
        },
    )


def fail(task_id: str, error: str) -> None:
    _update(
        task_id,
        {"status": STATUS_FAILED, "finished_at": _now(), "error": error[:500]},
    )


def discard_file(payload: dict | None) -> None:
    """删掉暂存文件。任务无论成功失败都要删，否则上传目录只涨不消。"""
    path = (payload or {}).get("file_path")
    if not path:
        return
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(
            "task.spool_cleanup_failed",
            extra={"fields": {"path": path, "error": f"{type(exc).__name__}: {exc}"[:200]}},
        )