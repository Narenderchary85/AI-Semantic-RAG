from src.chunking.semantic_chunker import SemanticChunker

def test_chunking_basic():
    s = "This is sentence one. This is sentence two that continues the idea. A new topic starts here."
    sc = SemanticChunker(buffer_size=1, theta=0.7)
    chunks = sc.process_document(s)
    assert len(chunks) >= 1