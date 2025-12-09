from src.pipeline.ambedkargpt import answer_query
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"

def test_full_pipeline_answer():
    question = "What were Dr. Ambedkar's ideas on caste?"

    try:
        answer_query(question, DATA_DIR)
        success = True
    except Exception as e:
        success = False
        print(e)

    assert success, "End-to-end SemRAG pipeline failed"
