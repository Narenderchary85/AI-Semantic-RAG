import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class GlobalSearcher:
    def __init__(self, community_summaries, community_embeddings, config):
        # community_summaries: dict cid->summary_text
        # community_embeddings: dict cid->vector
        self.comm_summ = community_summaries
        self.comm_emb = community_embeddings
        self.config = config

    def topk_communities(self, q_emb, top_k=3):
        scores = []
        for cid, emb in self.comm_emb.items():
            s = cosine_similarity([q_emb], [emb])[0][0]
            scores.append((cid, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def global_graph_rag_search(self, q_emb, chunk_texts, chunk_embeddings, top_k=3):
        # 1. find top-K communities relevant to Q
        comm_scores = self.topk_communities(q_emb, top_k=top_k)
        points = []
        for cid, cs in comm_scores:
            chunks = [cid_item for cid_item in chunk_texts.keys() if cid_item.startswith('chunk_')]
            # score points (sub-pieces) inside each chunk by similarity
            for cid_chunk in chunks:
                emb = chunk_embeddings.get(cid_chunk)
                if emb is None:
                    continue
                score = cosine_similarity([q_emb], [emb])[0][0]
                points.append((cid_chunk, score))
        points.sort(key=lambda x: x[1], reverse=True)
        return points[:top_k]