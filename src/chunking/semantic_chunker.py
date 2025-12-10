import os
import json
import math
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk import sent_tokenize
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

def token_count(text: str) -> int:
    return len(text.split())

class SemanticChunker:
    def __init__(self, model_name: str='all-MiniLM-L6-v2', buffer_size: int=5, theta: float=0.75,
                 token_limit: int=1024, overlap: int=128):
        self.model = SentenceTransformer(model_name)
        self.buffer_size = buffer_size
        self.theta = theta
        self.token_limit = token_limit
        self.overlap = overlap

    def split_sentences(self, text: str) -> List[str]:
        return sent_tokenize(text)

    def buffer_merge(self, sentences: List[str]) -> List[str]:
        merged = []
        n = len(sentences)
        b = self.buffer_size
        for i in range(n):
            left = max(0, i - b)
            right = min(n, i + b + 1)
            merged.append(' '.join(sentences[left:right]))
        seen = set(); out = []
        for s in merged:
            if s not in seen:
                out.append(s); seen.add(s)
        return out

    def embed_sentences(self, sentences: List[str]):
        return self.model.encode(sentences, show_progress_bar=False)

    def semantic_grouping(self, sentences: List[str]) -> List[str]:
        merged = self.buffer_merge(sentences)
        if not merged:
            return []
        embeddings = self.embed_sentences(merged)
        chunks = []
        current = merged[0]
        for i in range(len(merged)-1):
            d = 1 - cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            if d < (1 - self.theta):
                current += ' ' + merged[i+1]
            else:
                chunks.append(current)
                current = merged[i+1]
        chunks.append(current)
        final = []
        for c in chunks:
            if token_count(c) <= self.token_limit:
                final.append(c)
            else:
                words = c.split()
                i = 0
                L = len(words)
                while i < L:
                    end = min(i + self.token_limit, L)
                    sub = ' '.join(words[i:end])
                    final.append(sub)
                    if end == L:
                        break
                    i = max(0, end - self.overlap)
        return final

    def process_document(self, text: str) -> List[Dict]:
        sentences = self.split_sentences(text)
        groups = self.semantic_grouping(sentences)
        chunk_embs = self.embed_sentences(groups)
        out = []
        for i,g in enumerate(groups):
            out.append({'id': f'chunk_{i}', 'text': g, 'embedding': chunk_embs[i].tolist()})
        return out

    def save_chunks(self, chunks: List[Dict], out_path: str):
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)




