#from thefuzz import fuzz
import pymupdf
import time
from pypdf import PdfReader
from pathlib import Path
import re

def extract_text(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF using pymupdf
    """
    doc = pymupdf.open(stream=pdf_bytes, filetype='pdf')
    return "\n\n".join(page.get_text("text") for page in doc)

def clean_pdf_text(text: str) -> str:
    """
    Clean extracted PDF text for RAG preprocessing.
    
    Steps:
    1. Normalize whitespace
    2. Remove extra line breaks while keeping paragraph breaks
    3. Fix broken hyphenated words across lines
    4. Remove weird non-ASCII characters
    """
    # 1. Replace multiple spaces/tabs with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # 2. Fix hyphenated line breaks (e.g., "exam-\nple" -> "example")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # 3. Replace line breaks within paragraphs with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. Normalize multiple newlines into just two (for paragraphs)
    text = re.sub(r"\n{2,}", "\n\n", text)

    # 5. Strip weird characters (keep basic punctuation)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # 6. Final strip
    text = text.strip()

    return text
