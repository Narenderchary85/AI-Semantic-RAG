import pickle
import json
import yaml
import numpy as np
from pathlib import Path
from src.retrieval.local_search import LocalSearcher
from src.retrieval.global_search import GlobalSearcher
from src.chunking.semantic_chunker import SemanticChunker
from src.graph.graph_builder import KnowledgeGraph

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
CONFIG = yaml.safe_load(open(BASE_DIR / "config.yaml"))

def load_data():
    chunks = json.load(open(DATA_DIR / "chunks.json"))
    kgdata = pickle.load(open(DATA_DIR / "knowledge_graph.pkl", "rb"))
    commdata = pickle.load(open(DATA_DIR / "community_summaries.pkl", "rb"))
    chunk_emb = pickle.load(open(DATA_DIR / "chunk_embeddings.pkl", "rb"))

    kg = KnowledgeGraph()
    kg.G = kgdata["graph"]
    kg.entity_chunks = kgdata["entity_chunks"]

    return chunks, kg, commdata, chunk_emb

def test_local_graph_rag_search():
    chunks, kg, _, chunk_emb = load_data()
    chunker = SemanticChunker(CONFIG["embedding_model"])
    query_emb = chunker.model.encode("Ambedkar views on caste system")

    localer = LocalSearcher(kg, chunk_emb, CONFIG)
    results = localer.local_graph_rag_search(query_emb, top_k=3)

    assert len(results) > 0
    assert "chunk_id" in results[0]

def test_global_graph_rag_search():
    chunks, _, commdata, chunk_emb = load_data()
    chunker = SemanticChunker(CONFIG["embedding_model"])
    q_emb = chunker.model.encode("Indian constitution drafting")

    globaler = GlobalSearcher(
        commdata["summaries"],
        commdata["emb"],
        CONFIG
    )

    chunk_texts = {c["id"]: c["text"] for c in chunks}
    results = globaler.global_graph_rag_search(q_emb, chunk_texts, chunk_emb, top_k=3)

    assert len(results) > 0
