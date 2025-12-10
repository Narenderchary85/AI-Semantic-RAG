Structure:
- config.yaml 
- requirements.txt
- data/Ambedkar_book.pdf 
- src/pipeline/ambedkargpt.py 
- src/chunking/semantic_chunker.py
- src/graph/entity_extractor.py
- src/graph/graph_builder.py
- src/graph/community_detector.py
- src/retrieval/local_search.py
- src/retrieval/global_search.py
- src/llm/ollama_client.py
- src/llm/prompt_templates.py

## Installation and Setup

1. **Create venv:**
   ```bash
   python -m venv .venv 
   
2.  **Install:**
     ```bash
      pip install -r requirements.txt

3.  **Install spacy model:**
     ```bash
     python -m spacy download en_core_web_sm

4. **Start ollama and load a local model:**
     ```bash
        (e.g., "llama3" or "mistral") via ollama instructions
        (see https://ollama.ai docs) and ensure `ollama` CLI is available.
     
5. **Run pipeline once to build chunks & graph:**
     ```bash
          python -m src.pipeline.ambedkargpt.py --build

6. **Run For Query:**
     ```bash
      python -m src.pipeline.ambedkargpt.py --query "Who is Annihilation in Caste?"

