from langchain_core.documents import Document
import torch
import numpy as np
import re

from config import get_embeddings

class SemanticChunker:
    def __init__(self, overlap_size, max_chunk_size):
        self.embeddings_model = get_embeddings()
        self.overlap_size = overlap_size
        self.max_chunk_size = max_chunk_size
    
    def split_to_sentences(self, text):
        sentences = re.split(r'([。！？.?!]\s*)', text)
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences):
            # 有标点符号，合并句子内容和标点
                result.append(sentences[i] + sentences[i+1])
                i += 2
            else:
            # 没有标点符号，只取句子内容
                result.append(sentences[i])
                i += 1

        #print(f"已分割为句子: {result}")

        return [s.strip() for s in result if s.strip()]

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def get_sentence_embeddings(self, sentences):
        # 原来逐句调 embed_query，一句一次 HTTP 往返，长文本切块要上百次请求
        # （实测一篇 115 句的文本光切块就 149 秒）。embed_documents 一次批量提交、
        # 按输入顺序返回向量，相似度计算的语义不变，耗时降到秒级。
        return self.embeddings_model.embed_documents(sentences)

    def get_boundaries(self, text):
        sentences = self.split_to_sentences(text)
        embeddings = self.get_sentence_embeddings(sentences)
        boundaries = []
        for i in range(len(embeddings)-1):
            similarity = self.cosine_similarity(embeddings[i], embeddings[i+1])
            #print(f"句子对 {sentences[i]} 和 {sentences[i+1]} 的相似度: {similarity}")
            if similarity < 0.7:
                boundary_pos = sum(len(s) for s in sentences[:i+1]) 
                boundaries.append(boundary_pos)

        return boundaries

    def adjust_boundaries(self, text, boundaries):
        """语义段超过 max_chunk_size 时按固定步长切到底，保证每块都不超上限。

        语义边界是由句子间相似度决定的，连续句子都相似时切出来的段可能很长。
        原来只补切一刀（`sub_start = sub_end` 之后没有再切），剩下的部分仍是一块
        远超上限的长文本——实测 5690 字的段只被切成 300 + 5390 两块。
        """
        all_boundaries = [0] + boundaries + [len(text)]
        adjust_boundaries = []
        for i in range(len(all_boundaries)-1):
            start = all_boundaries[i]
            end = all_boundaries[i+1]
            # 每满一个 max_chunk_size 补一个切点，末段不足一步长时由 end 收尾
            while end - start > self.max_chunk_size:
                start = start + self.max_chunk_size
                adjust_boundaries.append(start)
            adjust_boundaries.append(end)

        return [b for b in adjust_boundaries if b < len(text)]

    def create_chunks_with_overlap(self, text, boundaries):
        chunks = []
        boundaries = [0] + boundaries + [len(text)]
        for i in range(len(boundaries)-1):
            start = boundaries[i]
            end = boundaries[i+1]
            
            overlap_start = max(start - self.overlap_size, 0)
            overlap_end = min(end + self.overlap_size, len(text))

            chunk_text = text[overlap_start:overlap_end]
            chunks.append(chunk_text)

        return chunks

    def chunk_document(self, text):
        boundaries = self.get_boundaries(text)
        adjust_boundaries = self.adjust_boundaries(text, boundaries)
        chunks = self.create_chunks_with_overlap(text, adjust_boundaries)
        documents = []
        for chunk_text in chunks:
            doc = Document(
                page_content=chunk_text,
            )
            documents.append(doc)  

        return documents

