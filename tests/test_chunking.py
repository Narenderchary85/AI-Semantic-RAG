from src.chunking.semantic_chunker import SemanticChunker
from pathlib import Path
import pdfplumber
import yaml

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(open(BASE_DIR / "config.yaml"))

def load_pdf_text():
    pdf_path = BASE_DIR / "data" / "Ambedkar_book.pdf"
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages[:5]:  
            text += p.extract_text() + "\n"
    return text

def test_semantic_chunking_produces_chunks():
    text = load_pdf_text()
    chunker = SemanticChunker(
        model_name=CONFIG['embedding_model'],
        buffer_size=CONFIG['buffer_size'],
        theta=CONFIG['cosine_threshold']
    )

    chunks = chunker.process_document(text)

    assert len(chunks) > 0, "Semantic chunking returned no chunks"

def test_chunks_have_embeddings():
    text = load_pdf_text()
    chunker = SemanticChunker(model_name=CONFIG['embedding_model'])

    chunks = chunker.process_document(text)

    for c in chunks[:5]:
        assert "embedding" in c
        assert len(c["embedding"]) > 0
