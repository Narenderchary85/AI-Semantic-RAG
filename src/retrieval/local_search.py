import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class LocalSearcher:
    def __init__(self, kg, chunk_embeddings, config):
        self.kg = kg
        self.chunk_embeddings = chunk_embeddings
        self.config = config

    def query_entity_similarity(self, q_emb, top_n=50):
        scores = []
        for entity, chunks in self.kg.entity_chunks.items():
            vecs = [self.chunk_embeddings[cid] for cid in chunks if cid in self.chunk_embeddings]
            if not vecs:
                continue
            mean_vec = np.mean(vecs, axis=0)
            sim = cosine_similarity([q_emb], [mean_vec])[0][0]
            scores.append((entity, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]

    def local_graph_rag_search(self, q_emb, history_emb=None, top_k=5):
        tau_e = self.config['tau_e']
        tau_d = self.config['tau_d']
        entity_scores = self.query_entity_similarity(q_emb, top_n=200)
        candidates = [e for e,s in entity_scores if s >= tau_e]
        results = []
        for e in candidates:
            for cid in self.kg.entity_chunks.get(e, []):
                if cid not in self.chunk_embeddings:
                    continue
                sim_chunk = cosine_similarity([q_emb], [self.chunk_embeddings[cid]])[0][0]
                if sim_chunk >= tau_d:
                    results.append({'entity': e, 'chunk_id': cid, 'score': sim_chunk})
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
