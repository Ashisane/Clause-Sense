import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_question(question, context):
    prompt = f"""
You are an assistant that answers questions from policy documents.
Context:
{context}

Question:
{question}

Answer clearly and concisely based only on the context above.
    """
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a precise insurance policy Q&A assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama3-70b-8192",  # free tier on Groq
        temperature=0,
        max_tokens=512
    )
    return completion.choices[0].message["content"].strip()
