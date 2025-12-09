import networkx as nx
import pickle
from collections import defaultdict

class KnowledgeGraph:
    def __init__(self):
        self.G = nx.Graph()
        self.entity_chunks = defaultdict(set)

    def add_chunk_entities(self, chunk_id: str, entities, relations):
        for e in entities:
            node = e['text']
            if not self.G.has_node(node):
                self.G.add_node(node, label=e.get('label',''))
            self.entity_chunks[node].add(chunk_id)
        for r in relations:
            s = r['source']; t = r['target']
            if not self.G.has_edge(s,t):
                self.G.add_edge(s,t, type=r.get('type','co-occur'), weight=1)
            else:
                self.G[s][t]['weight'] += 1

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'graph': self.G, 'entity_chunks': self.entity_chunks}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.G = data['graph']
            self.entity_chunks = data['entity_chunks']
