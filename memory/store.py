"""长期记忆的文件式存储（参考 memory_example.py 的 .memory 方案）。

与原先「Redis + RediSearch 向量检索」不同，这里把每条记忆落成一个个 Markdown 文件：
每条记录带 YAML frontmatter（name / description / type），正文是自由文本；目录里再维护
一份 MEMORY.md 作为目录索引，供召回时先用 LLM 挑选、再按需读正文。

目录布局（按 user_id 隔离）：

    {MEMORY_ROOT}/<safe_dir(user_id)>/
        MEMORY.md                目录索引（每条一行）
        {slug}.md                一条记忆
        ...

按 user_id 分目录是为了多用户隔离：A 的记忆永远不会进入 B 的召回候选，也不需要像
向量索引那样在查询侧做过滤。目录名由「slug + user_id 的 md5 前缀」拼成，既保证同一
user_id 稳定映射到同一目录，又保证不同 user_id 不会因为 slug 撞车而共用目录；
memory_path 再做一次 resolve 校验，防止文件名里的 `..` / 绝对路径穿越到别的用户目录。
"""

import hashlib
import re
from pathlib import Path

from config import MEMORY_ROOT

# 允许的记忆类型（与 memory_example 保持一致）
MEMORY_TYPES = ("user", "feedback", "project", "reference")

# 临时性措辞：命中即认为这条内容只与当前会话/当前任务有关，不该长期保存
TEMPORARY_MEMORY_MARKERS = (
    "this session",
    "current session",
    "this turn",
    "current turn",
    "this task",
    "current task",
    "for now",
    "just this time",
    "today only",
    "本次会话",
    "当前会话",
    "这一轮",
    "当前轮次",
    "本次任务",
    "当前任务",
    "暂时",
)

MEMORY_INDEX_NAME = "MEMORY.md"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """拆出 YAML frontmatter 与正文；没有 frontmatter 时返回 ({}, 原文)。"""
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        import yaml

        metadata = yaml.safe_load(parts[1]) or {}
    except Exception:  # noqa: BLE001 - 坏 frontmatter 当作没有，不让整个召回挂掉
        return {}, text
    if not isinstance(metadata, dict):
        return {}, text
    return metadata, parts[2].lstrip()


def memory_slug(name: str) -> str:
    """名字 → 文件名 slug（小写、非字母数字折成连字符）。"""
    slug = re.sub(r"[^\w]+", "-", name.lower()).strip("-_")
    return slug or "memory"


def safe_dir(user_id: str) -> str:
    """user_id → 目录名：slug + md5 前缀。

    只用 slug 会撞车（"张三" 与 "张-三" 同 slug），只用 md5 又不可读，
    两者拼起来兼顾唯一与可排查；md5 也顺带保证目录名里没有路径分隔符。
    """
    digest = hashlib.md5((user_id or "").encode("utf-8")).hexdigest()[:8]
    return f"{memory_slug(user_id or 'user')[:32]}-{digest}"


def user_root(user_id: str) -> Path:
    """某个用户的记忆目录（可能尚不存在）。"""
    return Path(MEMORY_ROOT).resolve() / safe_dir(user_id)


def memory_path(user_id: str, filename: str, allow_index: bool = False) -> Path:
    """把裸文件名解析成用户目录下的绝对路径，并挡住路径穿越。

    - 文件名必须是不含分隔符的裸名（Path.name == filename）；
    - 解析后的路径必须仍在该用户目录内；
    - 除显式 allow_index 外不允许直接指向目录索引。
    """
    if Path(filename).name != filename:
        raise ValueError(f"非法的记忆文件名: {filename}")
    if filename == MEMORY_INDEX_NAME and not allow_index:
        raise ValueError("MEMORY.md 是目录索引，不是记忆记录")

    root = user_root(user_id)
    path = (root / filename).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"记忆路径越界: {filename}")
    return path


def _normalized(value: str) -> str:
    return " ".join((value or "").lower().split())


def should_store_memory(candidate: dict, existing: list[dict]) -> bool:
    """是否该落盘：必须是持久作用域、类型合法、字段齐全，且不是临时信息/已有重复。"""
    if not isinstance(candidate, dict):
        return False
    if candidate.get("scope") != "persistent":
        return False
    if candidate.get("type") not in MEMORY_TYPES:
        return False

    name = str(candidate.get("name", "")).strip()
    description = str(candidate.get("description", "")).strip()
    body = str(candidate.get("body", "")).strip()
    if not name or not description or not body:
        return False

    candidate_text = _normalized(f"{name}\n{description}\n{body}")
    if any(marker in candidate_text for marker in TEMPORARY_MEMORY_MARKERS):
        return False

    slug = memory_slug(name)
    normalized_description = _normalized(description)
    normalized_body = _normalized(body)
    for memory in existing:
        if memory_slug(str(memory.get("name", ""))) == slug:
            return False
        if _normalized(str(memory.get("description", ""))) == normalized_description:
            return False
        if _normalized(str(memory.get("body", ""))) == normalized_body:
            return False
    return True


def memory_document(name: str, mem_type: str, description: str, body: str) -> str:
    """拼出一条记忆的 Markdown 正文（frontmatter + body）。"""
    import yaml

    metadata = yaml.safe_dump(
        {"name": name, "description": description, "type": mem_type},
        sort_keys=False,
        allow_unicode=True,
    ).strip()
    return f"---\n{metadata}\n---\n\n{body.strip()}\n"


def write_memory_file(
    user_id: str, name: str, mem_type: str, description: str, body: str
) -> Path:
    """写一条记忆并重建目录索引；同名记忆按 slug 覆盖。"""
    if not name.strip():
        raise ValueError("记忆 name 不能为空")
    if mem_type not in MEMORY_TYPES:
        raise ValueError(f"未知的记忆类型: {mem_type}")
    if not description.strip() or not body.strip():
        raise ValueError("记忆 description 与 body 不能为空")

    root = user_root(user_id)
    root.mkdir(parents=True, exist_ok=True)
    path = memory_path(user_id, f"{memory_slug(name)}.md")
    path.write_text(memory_document(name, mem_type, description, body), encoding="utf-8")
    rebuild_memory_index(user_id)
    return path


def rebuild_memory_index(user_id: str) -> None:
    """按目录里的记忆文件重建 MEMORY.md 目录索引。"""
    root = user_root(user_id)
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(root.glob("*.md")):
        if path.name == MEMORY_INDEX_NAME:
            continue
        try:
            path = memory_path(user_id, path.name)
        except ValueError:
            continue
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        name = " ".join(str(metadata.get("name") or path.stem).split())
        first_line = next((line for line in body.splitlines() if line.strip()), "")
        description = " ".join(str(metadata.get("description") or first_line).split())
        lines.append(f"- [{name}]({path.name}) - {description}")
    memory_path(user_id, MEMORY_INDEX_NAME, allow_index=True).write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )


def read_memory_index(user_id: str) -> str:
    """读目录索引；不存在时返回空串。"""
    try:
        path = memory_path(user_id, MEMORY_INDEX_NAME, allow_index=True)
    except ValueError:
        return ""
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def read_memory_file(user_id: str, filename: str) -> str | None:
    """读一条记忆的完整内容；文件名非法或不存在时返回 None。"""
    try:
        path = memory_path(user_id, filename)
    except ValueError:
        return None
    return path.read_text(encoding="utf-8") if path.is_file() else None


def list_memory_files(user_id: str) -> list[dict]:
    """列出该用户的所有记忆记录，供召回与合并使用。"""
    root = user_root(user_id)
    records = []
    if not root.exists():
        return records
    for path in sorted(root.glob("*.md")):
        if path.name == MEMORY_INDEX_NAME:
            continue
        try:
            path = memory_path(user_id, path.name)
        except ValueError:
            continue
        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        records.append(
            {
                "filename": path.name,
                "name": str(metadata.get("name") or path.stem),
                "description": str(metadata.get("description") or ""),
                "type": str(metadata.get("type") or "project"),
                "body": body.strip(),
            }
        )
    return records