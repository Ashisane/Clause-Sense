import faiss
from app.services.embeddings import embed_texts

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

class InMemoryRetriever:
    def __init__(self, chunks):
        self.chunks = chunks
        embeddings = embed_texts(chunks)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        query_vec = embed_texts([query])
        distances, indices = self.index.search(query_vec, top_k)
        return [self.chunks[i] for i in indices[0]]
