from langchain_core.documents import Document
import torch
from langchain_openai import OpenAIEmbeddings
import numpy as np
import re

class SemanticChunker:
    def __init__(self, overlap_size, max_chunk_size):
        self.embeddings_model = OpenAIEmbeddings(
            model="",
            api_key="",
            base_url=""
        )
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
        embeddings = []
        for sentence in sentences:
            embedding = self.embeddings_model.embed_query(sentence)
            #print(f"已嵌入句子: {sentence}")
            embeddings.append(embedding)
        return embeddings

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
        all_boundaries = [0] + boundaries + [len(text)]
        adjust_boundaries = []
        for i in range(len(all_boundaries)-1):
            start = all_boundaries[i]
            end = all_boundaries[i+1]
            if end - start > self.max_chunk_size:
                sub_start = start
                sub_end = min(end, sub_start + self.max_chunk_size)
                adjust_boundaries.append(sub_end)
                sub_start = sub_end
            else:
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


'''
#测试用
chunker = SemanticChunker(overlap_size=0, max_chunk_size=500)
docs = """人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，旨在创造能够执行通常需要人类智能的任务的机器。这些任务包括视觉感知、语音识别、决策制定和语言翻译。机器学习是人工智能的一个子领域，它使计算机能够在没有明确编程的情况下学习。机器学习算法从数据中构建数学模型，用于进行预测或决策。深度学习是机器学习的一个子集，它使用多层神经网络。与传统的机器学习方法相比，深度学习在图像识别、自然语言处理等领域表现出色。然而，深度学习模型需要大量的计算资源和数据。没有足够的数据，深度学习模型可能无法达到预期的性能。张德志是大帅哥。"""
result = chunker.chunk_document(docs)
print(result)
'''