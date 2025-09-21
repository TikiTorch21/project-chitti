from chunking import *
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os
import warnings
from sentence_transformers import SentenceTransformer
import pymupdf

warnings.filterwarnings("ignore", message=".*resource_tracker.*")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")




def get_embeddings(chunks, batch_size=8):
    return model.encode(chunks, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)

def extract_text(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF using pymupdf
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype='pdf')
    return "\n\n".join(page.get_text("text") for page in doc)



PDF_PATH = Path('/Users/prateekM/Desktop/1_Projects/Project Chitti/data/raw/test pdfs/basketball_pdf.pdf')
with open(PDF_PATH, "rb") as f:
    pdf_bytes = f.read()
pdf_text = extract_text(pdf_bytes=pdf_bytes)
embeddings = get_embeddings(chunk_text(pdf_text, goal="exact_size", chunk_size=100, overlap=20))
print(embeddings.shape)   # e.g. torch.Size([3, 4096])