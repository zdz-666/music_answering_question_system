"""受白名单约束的文件读取，供 agent 的 read_file 工具使用。

模型能读的文件被严格限制在三处：

- 当前用户自己的记忆目录（`MEMORY_ROOT/<safe_dir(user_id)>/`）；
- 上下文压缩落盘的工具结果目录（`TOOL_RESULTS_DIR/<safe_dir(user_id)>/`）；
- 上下文压缩归档的 transcript 目录（`TRANSCRIPT_DIR/<safe_dir(user_id)>/`）。

三处都按 user_id 分目录，与 `memory.store` / `context_compact` 用的是同一个 `safe_dir` 映射，
因此 A 用户既能读自己的记忆，也能读自己被压缩归档的内容，却读不到 B 用户的任何一份。

后两处的文件名由 uuid / tool_call_id 生成，正是压缩后占位文本（`Full output: <路径>`、
`[N messages archived at <路径>]`）里给出的路径——读回来能让模型找回被压缩替换掉的原始内容。

只放开这几处，是为了不让模型（或其输出被诱导后）读到 `.env`、源码等敏感文件：每次读取都先
`resolve` 再判断是否落在白名单目录内，`..`、绝对路径等穿越写法一律拒绝。读取结果按
`READ_FILE_CHAR_LIMIT` 截断，避免把大文件整段灌进上下文。
"""

from pathlib import Path

from config import READ_FILE_CHAR_LIMIT, TOOL_RESULTS_DIR, TRANSCRIPT_DIR
from memory.store import safe_dir, user_root


def allowed_roots(user_id: str) -> list[Path]:
    """当前请求可读的目录白名单（均已 resolve）。

    归档目录与 context_compact 写入时一样按 user_id 分子目录；记忆目录在有 user_id 时才放开。
    """
    user_key = safe_dir(user_id)
    roots = [
        Path(TOOL_RESULTS_DIR).resolve() / user_key,
        Path(TRANSCRIPT_DIR).resolve() / user_key,
    ]
    if (user_id or "").strip():
        roots.append(user_root(user_id))
    return roots


def resolve_readable(path_str: str, user_id: str) -> Path | None:
    """把模型给的路径解析成可读文件；越界、非文件或不存在时返回 None。

    相对路径按「当前用户记忆目录下的文件名」理解，与 `MEMORY.md` 索引里的链接一致。
    """
    if not path_str:
        return None
    candidate = Path(path_str)
    if not candidate.is_absolute():
        if not (user_id or "").strip():
            return None
        candidate = user_root(user_id) / candidate
    path = candidate.resolve()
    if not any(path.is_relative_to(root) for root in allowed_roots(user_id)):
        return None
    return path if path.is_file() else None


def read_file(path_str: str, user_id: str) -> str:
    """读一个白名单内的文件，返回给模型看的文本；失败时返回说明而不是抛异常。"""
    path = resolve_readable(path_str, user_id)
    if path is None:
        return f"无法读取 {path_str}：它不在可读范围内（记忆目录 / 压缩归档）或不存在。"

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # noqa: BLE001 - 读失败要让模型知道，而不是打断 loop
        return f"读取 {path_str} 失败：{type(exc).__name__}: {exc}"

    if len(text) > READ_FILE_CHAR_LIMIT:
        return (
            text[:READ_FILE_CHAR_LIMIT]
            + f"\n\n[已截断：全文共 {len(text)} 字符，这里只返回前 {READ_FILE_CHAR_LIMIT} 字符]"
        )
    return text