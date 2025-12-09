from src.llm.llm_client import OllamaClient

ollama = OllamaClient()

MAX_CHARS = 3000  # prevent Windows overflow

community_text = "Provide your community text here or pass it as an argument"
safe_text = community_text[:MAX_CHARS]

def summarize_community(nodes, entity_chunks, chunk_texts, max_chars=3000):
    texts = []
    for n in nodes:
        for cid in entity_chunks.get(n, []):
            texts.append(chunk_texts.get(cid, ""))

    text_blob = " ".join(texts)[:max_chars]

    prompt = f"""
Summarize the following content in 5–6 lines, focusing on key themes:

{text_blob}
"""
    return ollama.generate(prompt, max_tokens=200)