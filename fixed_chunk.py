"""按固定长度切块（带重叠），与 dynamic_chunk 的语义切块互为对照。

`SemanticChunker` 靠相邻句子的余弦相似度找语义边界，块长不固定、长短差异大；
这里完全不看语义，按字符数等步长滑动窗口切：

    chunk_size   每块的目标长度上限
    overlap_size 相邻两块重叠的字符数
    step         = chunk_size - overlap_size，窗口每次前进的步长

两边都提供 `chunk_document(text) -> list[Document]`，可以互换使用：
rgb_eval.py 用 `--chunker` 切换，用来对比两种切分方式对检索效果的影响。
"""

from langchain_core.documents import Document


class FixedChunker:
    def __init__(self, overlap_size=0, chunk_size=300):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size 必须为正数，当前为 {chunk_size}")
        if overlap_size < 0:
            raise ValueError(f"overlap_size 不能为负数，当前为 {overlap_size}")
        if overlap_size >= chunk_size:
            # step 会变成 0 或负数，滑动窗口无法前进
            raise ValueError(
                f"overlap_size（{overlap_size}）必须小于 chunk_size（{chunk_size}）"
            )

        self.overlap_size = overlap_size
        self.chunk_size = chunk_size

    def chunk_document(self, text):
        """等步长滑窗切分，返回 List[Document]。"""
        if not text:
            return []

        step = self.chunk_size - self.overlap_size
        chunks = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size]
            # 末块整段都落在上一块的重叠范围内时是纯重复内容，到此为止
            if chunks and len(piece) <= self.overlap_size:
                break
            chunks.append(piece)

        return [Document(page_content=chunk) for chunk in chunks]