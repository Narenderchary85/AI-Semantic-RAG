AmbedkarGPT — SemRAG implementation


This repository contains a runnable, local implementation of SemRAG specialized to answer
questions about Dr. B.R. Ambedkar's works (Ambedkar_book.pdf).


Structure (flat view):
- config.yaml # hyperparameters
- requirements.txt
- data/Ambedkar_book.pdf # place your PDF here
- src/pipeline/ambedkargpt.py # main pipeline (run this)
- src/chunking/semantic_chunker.py
- src/graph/entity_extractor.py
- src/graph/graph_builder.py
- src/graph/community_detector.py
- src/retrieval/local_search.py
- src/retrieval/global_search.py
- src/llm/ollama_client.py
- src/llm/prompt_templates.py
- notebooks/demo.ipynb # optional notebook for visualization


Run (example):
1. Create venv: python -m venv .venv && source .venv/bin/activate
2. Install: pip install -r requirements.txt
3. Install spacy model: python -m spacy download en_core_web_sm
4. Start ollama and load a local model (e.g., "llama3" or "mistral") via ollama instructions
(see https://ollama.ai docs) and ensure `ollama` CLI is available.
5. Place Ambedkar_book.pdf into data/
6. Run pipeline once to build chunks & graph: python src/pipeline/ambedkargpt.py --build
7. Run interactive server/demo: python src/pipeline/ambedkargpt.py --query "Who is Ambedkar?"


Note: This implementation intentionally focuses on clarity and reproducibility. The code
is modular and follows the SemRAG paper's algorithms (semantic chunking, buffer merging,
KG construction, local/global retrieval, community summaries, and LLM prompt integration).