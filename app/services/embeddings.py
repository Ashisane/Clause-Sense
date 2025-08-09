from sentence_transformers import SentenceTransformer

# Free test model
model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return model.encode(texts, convert_to_tensor=False)
