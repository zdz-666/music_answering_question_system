"""文档解析层：把上传的 PDF / DOCX 统一解析成 ParsedBlock 列表。

为什么单独抽一层：
- 原先 /api/upload 只认 zip 里的 XML，PDF / Word 根本读不出内容；
- 更关键的是图片：PDF 的图片不在文字层里，只调 page.get_text() 会把整张图丢掉。
  这里把图片抽出来，交给视觉模型（VLM）转成中文描述，再当作普通文本块参与问答，
  于是下游（检索、生成）只认 ParsedBlock，完全不感知原始文件格式。

图片管线：尺寸过滤（滤掉 logo / 页码装饰）→ MD5 去重 → VLM 描述 → 描述缓存落盘。
描述缓存按图片 MD5 存 JSON，跨请求复用，避免同一张图反复花钱。
"""

import base64
import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass

import docx
import pymupdf
from langchain_core.messages import HumanMessage
from PIL import Image

from config import IMAGE_MIN_SIZE, UPLOAD_IMAGE_DIR, VLM_MODEL, get_vlm
from observability import log_step, logger

# 目前支持的扩展名；.doc 是旧版二进制格式，解析不了，引导用户另存为 .docx
SUPPORTED_EXTENSIONS = (".pdf", ".docx")

_IMAGE_PROMPT = """请用简体中文描述这张图片的内容，供音乐知识问答检索使用。要求：
1. 若图中含文字（标题、歌词、乐谱记号、表格、说明文字），请逐条转写出来；
2. 描述图中呈现的实体与关系（人物、乐器、作品、曲式结构、图表趋势等）；
3. 只输出描述本身，不要加“这张图片是……”之类的前缀，不要使用 Markdown 标题。"""

# 描述拿不到时的降级文本；这类占位**不进缓存**，等 VLM 配置好或服务恢复后下次仍会重试
_FALLBACK_UNCONFIGURED = "[图片：未配置 VLM_MODEL，未生成图片描述]"
_FALLBACK_FAILED = "[图片：描述生成失败]"

_EXT_BY_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
}


class UnsupportedFormatError(ValueError):
    """上传了不支持的文件格式，调用方应回 400 而不是 500。"""


@dataclass
class ParsedBlock:
    """解析出的最小内容单元：一段正文，或一张图的文字描述。

    下游只关心 kind 与 text；page / image_path / source 是给日志与调试用的溯源信息。
    """

    kind: str  # "text" | "image"
    text: str
    page: int | None = None
    image_path: str | None = None
    source: str | None = None


def parse_document(filename: str, content: bytes) -> list[ParsedBlock]:
    """按扩展名分发到对应解析器。同步阻塞（读文件 + 逐图调 VLM），调用方需放线程池。"""
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix == ".pdf":
        return _parse_pdf(content)
    if suffix == ".docx":
        return _parse_docx(content)
    if suffix == ".doc":
        raise UnsupportedFormatError("不支持旧版 .doc，请用 Word 另存为 .docx 后再上传")
    raise UnsupportedFormatError(
        f"不支持的文件格式 {suffix or '(无扩展名)'}，目前仅支持 .pdf / .docx"
    )


def blocks_to_text(blocks: list[ParsedBlock]) -> str:
    """拼成给 Prompt 用的整段文本。图片描述单独成段，便于后续按段切块入库。"""
    parts = []
    for block in blocks:
        if block.kind == "text":
            parts.append(block.text)
            continue
        where = f"第 {block.page} 页" if block.page else "文中"
        parts.append(f"【图片内容（{where}）】{block.text}")
    return "\n\n".join(part for part in parts if part and part.strip())


class _ImagePipeline:
    """图片过滤 / 去重 / 描述 / 落盘，PDF 与 DOCX 共用。

    - `_cache`：MD5 → 描述，落盘持久化，跨请求复用；
    - `_seen`：本次解析内已出现过的 MD5，同一张图重复出现不再产出新块。
    """

    def __init__(self):
        self._cache = _load_cache()
        self._seen: set[str] = set()

    def build(self, data: bytes, page: int | None, width: int, height: int) -> ParsedBlock | None:
        if min(width, height) < IMAGE_MIN_SIZE:
            logger.info(
                "image.skip",
                extra={"fields": {"reason": "too_small", "width": width, "height": height}},
            )
            return None

        digest = hashlib.md5(data).hexdigest()
        if digest in self._seen:
            return None
        self._seen.add(digest)

        description = self._cache.get(digest)
        if description is None:
            description = _describe_with_vlm(data)
            if description is not None:
                # 只有成功的描述才写缓存，占位文本下次仍会重试
                self._cache[digest] = description
                _save_cache(self._cache)
        if description is None:
            description = _FALLBACK_UNCONFIGURED if not VLM_MODEL else _FALLBACK_FAILED

        return ParsedBlock(
            kind="image",
            text=description,
            page=page,
            image_path=_save_image(data, digest),
        )


def _parse_pdf(content: bytes) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    pipeline = _ImagePipeline()

    with pymupdf.open(stream=content, filetype="pdf") as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text().strip()
            images = _pdf_page_images(document, page)

            if not text and images:
                # 扫描页：没有文字层，整页渲染一次交给 VLM，比逐张零散抽图更完整
                pixmap = page.get_pixmap(dpi=150)
                block = pipeline.build(
                    pixmap.tobytes("png"), page_number, pixmap.width, pixmap.height
                )
                if block:
                    block.source = "扫描页"
                    blocks.append(block)
                continue

            if text:
                blocks.append(ParsedBlock(kind="text", text=text, page=page_number))

            for data, width, height in images:
                block = pipeline.build(data, page_number, width, height)
                if block:
                    blocks.append(block)

    return blocks


def _pdf_page_images(document, page) -> list[tuple[bytes, int, int]]:
    """取出本页内嵌图片的原始字节与像素尺寸，单张失败不影响整页。"""
    images = []
    for info in page.get_images(full=True):
        xref = info[0]
        try:
            raw = document.extract_image(xref)
        except Exception as exc:  # 个别损坏对象不该让整篇解析失败
            logger.warning(
                "image.extract_failed",
                extra={"fields": {"xref": xref, "error": f"{type(exc).__name__}: {exc}"}},
            )
            continue
        images.append((raw["image"], raw["width"], raw["height"]))
    return images


def _parse_docx(content: bytes) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []

    document = docx.Document(io.BytesIO(content))
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            blocks.append(ParsedBlock(kind="text", text=text))

    # docx 本质是 zip，图片都在 word/media/ 下，直接当压缩包读，不必依赖 python-docx 的关系解析
    pipeline = _ImagePipeline()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            if not name.startswith("word/media/"):
                continue
            data = archive.read(name)
            size = _image_size(data)
            if size is None:
                continue
            block = pipeline.build(data, None, size[0], size[1])
            if block:
                blocks.append(block)

    return blocks


def _image_size(data: bytes) -> tuple[int, int] | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            return image.size
    except Exception:
        return None


def _describe_with_vlm(data: bytes) -> str | None:
    """调 VLM 把图片转成中文描述；未配置或调用失败返回 None，由调用方降级为占位文本。"""
    if not VLM_MODEL:
        logger.info("vlm.skipped", extra={"fields": {"reason": "VLM_MODEL 未配置"}})
        return None

    message = HumanMessage(
        content=[
            {"type": "text", "text": _IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": _data_url(data)}},
        ]
    )
    try:
        with log_step("vlm.describe", input={"image_bytes": len(data)}) as span:
            result = get_vlm().invoke([message])
            span.output = result.content
        description = result.content.strip()
        return description or None
    except Exception as exc:
        logger.error(
            "vlm.describe_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"[:300]}},
        )
        return None


def _data_url(data: bytes) -> str:
    return f"data:{_guess_mime(data)};base64,{base64.b64encode(data).decode()}"


def _guess_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"GIF8":
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _save_image(data: bytes, digest: str) -> str | None:
    """按内容哈希落盘，天然去重；落盘失败不影响问答。"""
    if not UPLOAD_IMAGE_DIR:
        return None
    try:
        os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)
        path = os.path.join(UPLOAD_IMAGE_DIR, digest + _EXT_BY_MIME.get(_guess_mime(data), ".png"))
        if not os.path.exists(path):
            with open(path, "wb") as handle:
                handle.write(data)
        return path
    except OSError as exc:
        logger.warning(
            "image.save_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}},
        )
        return None


def _cache_path() -> str:
    return os.path.join(UPLOAD_IMAGE_DIR, "descriptions.json")


def _load_cache() -> dict:
    if not UPLOAD_IMAGE_DIR:
        return {}
    path = _cache_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        logger.warning(
            "image.cache_load_failed",
            extra={"fields": {"path": path, "error": f"{type(exc).__name__}: {exc}"}},
        )
        return {}


def _save_cache(cache: dict) -> None:
    if not UPLOAD_IMAGE_DIR:
        return
    try:
        os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)
        with open(_cache_path(), "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=2)
    except OSError as exc:
        logger.warning(
            "image.cache_save_failed",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}},
        )