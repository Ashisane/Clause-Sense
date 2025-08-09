from fastapi import FastAPI, Header, HTTPException
from app.models.request_model import QueryRequest
from app.models.response_model import QueryResponse

app = FastAPI()

API_KEY = "test_key"  # later from .env

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(payload: QueryRequest, authorization: str = Header(...)):
    # Check API Key
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API Key")

    # For now: dummy pipeline returns "sample" answers for testing
    answers = [f"Sample answer for: {q}" for q in payload.questions]

    return QueryResponse(answers=answers)
