# main.py
import os
import requests
import tempfile
import logging
from typing import List
import fitz  # PyMuPDF
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# -----------------------
# Configuration & Logging
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("clause_sense")

# -------------
# App + models
# -------------
app = FastAPI()
groq_client = Groq(api_key=GROQ_API_KEY)

# Request schema
class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

# -----------------------
# Text extraction Helpers
# -----------------------
def extract_text_from_pdf(url: str) -> str:
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    tmp_file.write(r.content)
    tmp_file.close()
    doc = fitz.open(tmp_file.name)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)

# -----------------------
# Chunking (1000 / 200)
# -----------------------
def chunk_document(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [c.strip().replace("\r", " ") for c in chunks if c.strip()]

# -----------------------
# Groq query with retry
# -----------------------
def groq_chat_with_retry(prompt_text: str, model: str = "llama3-70b-8192", retries: int = 3, backoff: float = 1.5) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_text}],
                model=model,
                temperature=0.1
            )
            choice = resp.choices[0]
            text = getattr(choice.message, "content", None) if hasattr(choice, "message") else None
            if not text:
                raise RuntimeError("No content from Groq response.")
            return text.strip()
        except Exception as e:
            last_err = e
            logger.warning(f"Groq attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                import time
                time.sleep(backoff ** attempt)
    raise last_err

# -----------------------
# Main LLM prompt (kept unchanged)
# -----------------------
PROMPT_TEMPLATE = """
You are an insurance policy QA assistant. To answer the question you need to follow three rules.
IMPORTANT - Give a compiled answer consisting of only the important information in form of sentence in same wording as of the context provided using the 3 rules mentioned below:

1) Read what is asked and then strictly use only the context to answer the question.
2) Keep the numbers/legal terms intact in the answer.
3) Use this format to generate the sentence - 1) Answer the direct question. 2) Mention important surrounding conditions that are related to the answer that are in the clause. 3) Provide source to your explanation if there exists any eg-"clause number or policy name"

Context:
{context}

Question:
{question}

Answer:
"""

# -----------------------
# Endpoint implementation
# -----------------------
@app.post("/hackrx/run")
def hackrx_run(req: HackRxRequest):
    # Extract document text
    doc_text = extract_text_from_pdf(req.documents)

    # Chunk text
    chunks = chunk_document(doc_text, chunk_size=1000, chunk_overlap=200)

    answers = []
    for question in req.questions:
        # For now: just take first few chunks (no embedding retrieval)
        # Can be replaced with better retrieval later
        context_text = "\n".join(chunks[:8])

        prompt_text = PROMPT_TEMPLATE.format(context=context_text, question=question)

        try:
            llm_answer = groq_chat_with_retry(prompt_text)
        except Exception as e:
            logger.exception(f"LLM failed for question: {question}")
            llm_answer = "Error: LLM backend failure."

        answers.append(llm_answer)

    return {"answers": answers}
