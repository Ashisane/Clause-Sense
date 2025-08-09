from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]
