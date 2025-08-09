import os
import requests
import tempfile
import fitz  # PyMuPDF
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI
app = FastAPI()

# Embedding model (free + fast)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# LLM Client (Groq now, GPT-4o later if needed)
groq_client = Groq(api_key=GROQ_API_KEY)

# Request schema
class HackRxRequest(BaseModel):
    documents: str
    questions: list[str]

# Download PDF & extract text
def extract_text_from_pdf(url):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    r = requests.get(url)
    tmp_file.write(r.content)
    tmp_file.close()
    doc = fitz.open(tmp_file.name)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text

# Create FAISS index
def create_faiss_index(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

# Retrieve and re-rank chunks
def retrieve_top_chunks(question, chunks, index, embeddings, k=8):
    q_emb = embedding_model.encode([question], convert_to_numpy=True)
    distances, indices = index.search(q_emb, k)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Re-rank by cosine similarity
    scored_chunks = [(chunk, util.cos_sim(q_emb, embedding_model.encode([chunk], convert_to_numpy=True))[0][0].item()) for chunk in retrieved_chunks]
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in scored_chunks]

# Query LLaMA (Groq)
def query_llm(context, question):
    prompt = f"""
You are an insurance policy QA assistant.
You must answer strictly and only using the provided context.
- Do NOT guess or add extra information not present in the context.
- Extract the answer exactly as written in the document, making minimal edits only for grammar or readability.
- Preserve numbers, dates, and legal terms exactly as they appear.
- If the answer is not clearly and explicitly stated in the context, respond exactly with: "Not mentioned in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    chat_completion = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0  # zero temp = max accuracy
    )
    return chat_completion.choices[0].message.content.strip()

# Main API endpoint
@app.post("/hackrx/run")
def hackrx_run(req: HackRxRequest):
    # Step 1: Extract text from PDF
    doc_text = extract_text_from_pdf(req.documents)

    # Step 2: Chunk text (small chunks, high overlap)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = splitter.split_text(doc_text)

    # Step 3: Create FAISS index
    index, embeddings = create_faiss_index(chunks)

    # Step 4: Process each question
    answers = []
    for q in req.questions:
        top_chunks = retrieve_top_chunks(q, chunks, index, embeddings, k=8)
        context = "\n".join(top_chunks)
        ans = query_llm(context, q)

        # Step 5: Double-check if "Not mentioned" might be false
        if ans.strip().lower() == "not mentioned in the document.":
            # Try with slightly larger context
            alt_context = "\n".join(top_chunks[:5])
            alt_ans = query_llm(alt_context, q)
            if alt_ans.strip().lower() != "not mentioned in the document.":
                ans = alt_ans

        answers.append(ans)

    # Step 6: Return answers
    return {"answers": answers}
