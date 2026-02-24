import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")

class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""
    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must present in the retrieval context")
    reference_answer: str = Field(description="Reference answer for this question")
    category: str = Field(description="Question category (like direct_fact, temporal, comparative)")

def load_tests() -> list[TestQuestion]:
    """
    Load all test questions from the JSONL file

    Args:
        None

    Returns:
        list[pydantic]: List of test questions in the pydantic object class TestQuestion 
    """
    tests = []
    with open(file=TEST_FILE, mode="r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))

    return tests