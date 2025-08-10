# main.py
import os
import requests
import tempfile
import time
import math
import re
import logging
from typing import List, Tuple, Set

import fitz  # PyMuPDF
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import faiss
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
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
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
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)

# -----------------------
# Chunking (1000 / 200)
# -----------------------
def chunk_document(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Use langchain splitter (character-based) with desired chunk size/overlap.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    # Cleanup whitespace
    chunks = [c.strip().replace("\r", " ") for c in chunks if c and c.strip()]
    return chunks

# -----------------------
# FAISS index helpers
# -----------------------
def create_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatL2, object]:
    """
    Create FAISS index and return (index, embeddings_array)
    """
    # encode in batches to avoid OOM
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# -----------------------
# Small NLP helpers
# -----------------------
_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def text_tokens(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]

def keyword_overlap_score(question: str, chunk: str) -> float:
    q_tokens = set(text_tokens(question))
    c_tokens = set(text_tokens(chunk))
    if not q_tokens:
        return 0.0
    overlap = q_tokens.intersection(c_tokens)
    # simple normalized overlap
    return len(overlap) / len(q_tokens)

def contains_any_keywords(question: str, chunk: str) -> bool:
    q_tokens = set(text_tokens(question))
    c_tokens = set(text_tokens(chunk))
    # require at least one non-trivial token match
    return len(q_tokens.intersection(c_tokens)) > 0

# -----------------------
# Clause stitching helper
# -----------------------
def stitch_if_truncated(chunks: List[str], idx_list: List[int]) -> str:
    """
    Given a list of selected chunk indices, produce a stitched context:
    - If a chunk ends without sentence-ending punctuation, append the next chunk.
    - Deduplicate repeated text.
    """
    used = set()
    out = []
    for idx in idx_list:
        if idx in used or idx < 0 or idx >= len(chunks):
            continue
        c = chunks[idx].strip()
        used.add(idx)

        # if chunk looks cut at end (no . ? ! ;), try to append next chunk
        if c and c[-1] not in ".?!;:" and idx + 1 < len(chunks):
            # append next chunk (one-level stitching)
            c_next = chunks[idx + 1].strip()
            stitched = c + " " + c_next
            out.append(stitched)
            used.add(idx + 1)
        else:
            out.append(c)
    # dedupe contiguous duplicates
    result_text = "\n\n".join(out)
    return result_text

# -----------------------
# Retrieval: hybrid flow
# -----------------------
def retrieve_relevant_chunks(question: str, chunks: List[str], index: faiss.IndexFlatL2, embeddings, base_k: int = 8) -> Tuple[List[str], List[int]]:
    """
    1. Determine dynamic k based on question
    2. Query FAISS top-k
    3. Build candidate set including any chunks that literally contain question tokens (context-boost)
    4. Re-rank candidates by combined score: alpha * cosine + beta * keyword_overlap
    5. Return top N chunks and their indices
    """

    # dynamic k: if question likely needs definitions/limits, increase k
    question_lower = question.lower()
    high_risk_terms = ["waiting period", "define", "what is", "limit", "sub-limit", "room rent", "icuc", "icu", "covered", "coverage"]
    dynamic_k = base_k
    for t in high_risk_terms:
        if t in question_lower:
            dynamic_k = max(dynamic_k, 12)

    # embeddings for question
    q_emb = embedding_model.encode([question], convert_to_numpy=True)

    # FAISS search
    distances, indices = index.search(q_emb, dynamic_k)
    faiss_indices = [int(i) for i in indices[0] if i != -1]

    # Candidate set start with FAISS results
    candidate_idxs: Set[int] = set(faiss_indices)

    # Add any exact token-match chunks (context boost)
    q_tokens = set(text_tokens(question))
    # avoid tiny tokens set
    if len(q_tokens) >= 1:
        for i, ch in enumerate(chunks):
            # simple containment check (case-insensitive) on important tokens
            if any(tok in ch.lower() for tok in q_tokens):
                candidate_idxs.add(i)

    # Build scored list
    candidates = list(candidate_idxs)
    chunk_embeddings = embeddings[candidates] if len(candidates) > 0 else None

    scores = []
    if candidates:
        # cosine similarities using util.cos_sim (batch)
        # compute q_emb dot with each candidate embedding
        cos_sims = util.cos_sim(q_emb, chunk_embeddings)[0].cpu().numpy()
        for idx_pos, idx in enumerate(candidates):
            chunk = chunks[idx]
            cos = float(cos_sims[idx_pos])
            kw = keyword_overlap_score(question, chunk)
            # combined score weights (tuneable)
            score = (0.7 * cos) + (0.3 * kw)
            scores.append((idx, score, cos, kw))
    else:
        scores = []

    # sort by combined score descending
    scores.sort(key=lambda x: x[1], reverse=True)

    # pick top N (we'll send top 6-8 chunks as context but keep indices)
    top_n = min(8, len(scores))
    top_indices = [s[0] for s in scores[:top_n]]
    top_chunks = [chunks[i] for i in top_indices]

    logger.info("Retrieval debug - question: %s", question)
    logger.info("FAISS indices (k=%d): %s", dynamic_k, faiss_indices)
    logger.info("Candidate indices after boost: %s", sorted(list(candidate_idxs))[:30])
    logger.info("Top selected indices: %s", top_indices)

    return top_chunks, top_indices

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
            # API returns ChatCompletionMessage objects; access .message.content if present
            # The Groq SDK may differ slightly; handle common possibilities robustly.
            choice = resp.choices[0]
            # Try several common access patterns
            text = None
            if hasattr(choice, "message") and isinstance(choice.message, dict):
                text = choice.message.get("content")
            elif hasattr(choice, "message") and hasattr(choice.message, "content"):
                text = choice.message.content
            elif isinstance(choice, dict):
                # fallback dict
                text = choice.get("message", {}).get("content") or choice.get("text")
            else:
                # last fallback: str casting
                text = str(choice)

            if not text:
                # try resp.choices[0].message["content"]
                try:
                    text = resp.choices[0].message["content"]
                except Exception:
                    pass

            if not text:
                raise RuntimeError("Could not extract content from Groq response.")

            return text.strip()
        except Exception as e:
            last_err = e
            logger.warning("Groq attempt %d/%d failed: %s", attempt, retries, repr(e))
            # on fatal http/client errors don't retry: but we'll do simple backoff for transient
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                break
    # final raise with the last error
    logger.error("Groq failed after %d attempts: %s", retries, repr(last_err))
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
    # 1. Extract doc text
    doc_text = extract_text_from_pdf(req.documents)

    # 2. Chunk document
    chunks = chunk_document(doc_text, chunk_size=1000, chunk_overlap=200)

    # 3. Index creation (in-memory)
    index, embeddings = create_faiss_index(chunks)

    answers = []
    selection_debug = []

    for question in req.questions:
        # 4. Retrieve top chunks + indices
        top_chunks, top_indices = retrieve_relevant_chunks(question, chunks, index, embeddings, base_k=8)

        # 5. Stitch if a chunk is truncated (stitch adjacent indices)
        # Ensure indices sorted and unique
        top_indices_sorted = sorted(set(top_indices))
        context_text = stitch_if_truncated(chunks, top_indices_sorted)

        # 6. Build final prompt (keeps existing prompt intact)
        prompt_text = PROMPT_TEMPLATE.format(context=context_text, question=question)

        # 7. Query Groq with retry wrapper
        try:
            llm_answer = groq_chat_with_retry(prompt_text, model="llama3-70b-8192", retries=3, backoff=1.5)
        except Exception as e:
            # surface graceful failure
            logger.exception("LLM failed for question: %s", question)
            llm_answer = "Error: LLM backend failure."

        answers.append(llm_answer)
        # store debug info
        selection_debug.append({
            "question": question,
            "selected_chunk_indices": top_indices_sorted,
            "selected_chunk_count": len(top_indices_sorted),
            "context_preview": context_text[:1200]  # only preview
        })

    # return answers and debug info (debug can be removed for final submission)
    return {"answers": answers, "debug": selection_debug}
