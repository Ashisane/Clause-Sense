import requests
from io import BytesIO
from PyPDF2 import PdfReader

def load_pdf_from_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    pdf_bytes = BytesIO(response.content)
    reader = PdfReader(pdf_bytes)

    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text
