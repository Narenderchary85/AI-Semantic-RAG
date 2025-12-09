import argparse
import os
import json
from pathlib import Path
import pdfplumber
from src.chunking.semantic_chunker import SemanticChunker
from src.graph.entity_extractor import extract_entities_and_relations
from src.graph.graph_builder import KnowledgeGraph
from src.graph.community_detector import detect_communities
from src.graph.summarizer import summarize_community
from src.llm.llm_client import OllamaClient
from src.llm.prompt_templates import TEMPLATE
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import yaml

BASE_DIR = Path(__file__).resolve().parents[2]  # project root
CONFIG_PATH = BASE_DIR / "config.yaml"

CONFIG = yaml.safe_load(CONFIG_PATH.open())

# helper: read pdf to text
def read_pdf(path):
    text = ''
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text += p.extract_text() + '\n'
    return text

# build pipeline

def build_pipeline(pdf_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = read_pdf(pdf_path)
    chunker = SemanticChunker(model_name=CONFIG['embedding_model'], buffer_size=CONFIG['buffer_size'],
                              theta=CONFIG['cosine_threshold'], token_limit=CONFIG['token_limit'],
                              overlap=CONFIG['subchunk_overlap'])
    print('Creating semantic chunks...')
    chunks = chunker.process_document(text)
    # save chunk_texts and embeddings
    chunk_texts = {c['id']: c['text'] for c in chunks}
    chunk_embeddings = {c['id']: np.array(c['embedding']) for c in chunks}
    with open(out_dir / 'chunks.json','w',encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f'Saved {len(chunks)} chunks')

    # build KG
    kg = KnowledgeGraph()
    for c in chunks:
        ents, rels = extract_entities_and_relations(c['text'])
        kg.add_chunk_entities(c['id'], ents, rels)
    kg.save(out_dir / 'knowledge_graph.pkl')
    print('Knowledge graph built with nodes:', kg.G.number_of_nodes(), 'edges:', kg.G.number_of_edges())

    # community detection
    partition, comm_to_nodes = detect_communities(kg.G)
    print('Detected', len(comm_to_nodes), 'communities')

    # community summaries and embeddings (summaries via ollama)
    ollama = OllamaClient(CONFIG.get('ollama_model','mistral:latest'))
    community_summaries = {}
    community_embeddings = {}
    for cid, nodes in comm_to_nodes.items():
        summary = summarize_community(nodes, kg.entity_chunks, chunk_texts)
        community_summaries[cid] = summary
        # embed summary using same chunker
        emb = chunker.model.encode(summary)
        community_embeddings[cid] = emb
    with open(out_dir / 'community_summaries.pkl','wb') as f:
        pickle.dump({'summaries': community_summaries, 'emb': community_embeddings}, f)
    # save chunk embeddings binary
    with open(out_dir / 'chunk_embeddings.pkl','wb') as f:
        pickle.dump({k: v for k,v in chunk_embeddings.items()}, f)
    print('Saved community summaries and chunk embeddings')


def answer_query(query, data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / 'chunks.json','r',encoding='utf-8') as f:
        chunks = json.load(f)
    with open(data_dir / 'knowledge_graph.pkl','rb') as f:
        kgdata = pickle.load(f)
    kg = KnowledgeGraph(); kg.G = kgdata['graph']; kg.entity_chunks = kgdata['entity_chunks']
    chunk_texts = {c['id']: c['text'] for c in chunks}
    chunk_embeddings = pickle.load(open(data_dir / 'chunk_embeddings.pkl','rb'))
    commdata = pickle.load(open(data_dir / 'community_summaries.pkl','rb'))
    community_summaries = commdata['summaries']
    community_embeddings = commdata['emb']

    chunker = SemanticChunker(model_name=CONFIG['embedding_model'])
    q_emb = chunker.model.encode(query)

    # Local search
    from src.retrieval.local_search import LocalSearcher
    localer = LocalSearcher(kg, chunk_embeddings, CONFIG)
    local_results = localer.local_graph_rag_search(q_emb, top_k=CONFIG['local_top_k'])

    # Global search
    from src.retrieval.global_search import GlobalSearcher
    globaler = GlobalSearcher(community_summaries, community_embeddings, CONFIG)
    global_results = globaler.global_graph_rag_search(q_emb, chunk_texts, chunk_embeddings, top_k=CONFIG['global_top_k'])

    # Build context
    context_items = []
    cited_chunks = set()
    for r in local_results:
        cid = r['chunk_id']; cited_chunks.add(cid)
        context_items.append(f"[Local chunk {cid}]: {chunk_texts[cid][:500]}")
    for pid, score in global_results:
        if pid in chunk_texts:
            cited_chunks.add(pid)
            context_items.append(
                f"[Global chunk {pid}]: {chunk_texts[pid][:500]}")
    # include top community summaries
    for cid, summ in list(community_summaries.items())[:CONFIG['global_top_k']]:
        context_items.append(f"[Community {cid} summary]: {summ[:800]}")

    context = '\n\n'.join(context_items)
    prompt = TEMPLATE.format(context=context, question=query)
    ollama = OllamaClient(CONFIG.get('ollama_model','mistral:latest'))
    ans = ollama.generate(prompt, max_tokens=300)
    print('\n=== Answer ===\n')
    print(ans)
    print('\n=== Cited chunks ===\n')
    print('\n'.join(sorted(list(cited_chunks))))

# Add this function near other helpers
def load_artifacts(data_dir):
    data_dir = Path(data_dir)
    # load chunks (list of dicts)
    with open(data_dir / 'chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    # load KG
    with open(data_dir / 'knowledge_graph.pkl', 'rb') as f:
        kgdata = pickle.load(f)
    kg = KnowledgeGraph()
    kg.G = kgdata['graph']
    kg.entity_chunks = kgdata['entity_chunks']
    # load chunk embeddings and community data
    chunk_embeddings = pickle.load(open(data_dir / 'chunk_embeddings.pkl', 'rb'))
    commdata = pickle.load(open(data_dir / 'community_summaries.pkl', 'rb'))
    community_summaries = commdata['summaries']
    community_embeddings = commdata['emb']
    # create a chunk_texts dict once
    chunk_texts = {c['id']: c['text'] for c in chunks}

    # create and return the shared components
    return {
        'chunks': chunks,
        'chunk_texts': chunk_texts,
        'kg': kg,
        'chunk_embeddings': chunk_embeddings,
        'community_summaries': community_summaries,
        'community_embeddings': community_embeddings
    }

def query(data_dir):
    print("\n AmbedkarGPT Interactive Mode")
    print("Type your question and press Enter")
    print("Press 1 or type 'exit' to quit\n")

    # Load artifacts once
    art = load_artifacts(data_dir)
    # Init the chunker ONCE (reuses sentence-transformers model in memory)
    chunker = SemanticChunker(model_name=CONFIG['embedding_model'])
    # Create searcher instances once
    from src.retrieval.local_search import LocalSearcher
    from src.retrieval.global_search import GlobalSearcher
    localer = LocalSearcher(art['kg'], art['chunk_embeddings'], CONFIG)
    globaler = GlobalSearcher(art['community_summaries'], art['community_embeddings'], CONFIG)

    ollama_client = OllamaClient(CONFIG.get('ollama_model','mistral:latest'))

    while True:
        q = input("Ask a question ➜ ").strip()
        if q in {"1", "exit", "quit"}:
            print("Exiting AmbedkarGPT. Goodbye!")
            break
        if not q:
            print("Please enter a valid question.\n")
            continue

        # Only compute the query embedding (fast) — do NOT reload the model
        q_emb = chunker.model.encode(q)

        # Local & Global search (use existing searchers / artifacts)
        local_results = localer.local_graph_rag_search(q_emb, top_k=CONFIG['local_top_k'])
        global_results = globaler.global_graph_rag_search(q_emb, art['chunk_texts'], art['chunk_embeddings'], top_k=CONFIG['global_top_k'])

        # Build context: keep it small — top few chunks only
        context_items = []
        cited_chunks = set()
        for r in local_results[:3]:   # only top 3 local chunks
            cid = r['chunk_id']; cited_chunks.add(cid)
            context_items.append(f"[Local chunk {cid}]: {art['chunk_texts'][cid][:300]}")
        for pid, score in global_results[:2]:  # top 2 global
            if pid in art['chunk_texts']:
                cited_chunks.add(pid)
                context_items.append(f"[Global chunk {pid}]: {art['chunk_texts'][pid][:300]}")
        # include top community summaries (limit size)
        comm_keys = list(art['community_summaries'].keys())[:CONFIG['global_top_k']]
        for cid in comm_keys:
            summ = art['community_summaries'][cid]
            context_items.append(f"[Community {cid} summary]: {summ[:500]}")

        context = '\n\n'.join(context_items)
        prompt = TEMPLATE.format(context=context, question=q)

        # Call the LLM (this is still the main time consumer)
        ans = ollama_client.generate(prompt, max_tokens=CONFIG.get('max_answer_tokens', 150))
        print('\n=== Answer ===\n')
        print(ans)
        print('\n=== Cited chunks ===\n')
        print('\n'.join(sorted(list(cited_chunks))))
        print('\n---\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true', help='build chunks and KG')
    parser.add_argument('--pdf', type=str, default='data/Ambedkar_book.pdf')
    parser.add_argument('--out', type=str, default='data/processed')
    parser.add_argument('--query', type=str, help='query to ask')
    args = parser.parse_args()
    if args.build:
        build_pipeline(args.pdf, args.out)
    if args.query:
        answer_query(args.query, args.out)
    else:
        query(args.out)
