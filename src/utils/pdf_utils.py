#from thefuzz import fuzz
import pymupdf
import time
from pypdf import PdfReader
from pathlib import Path
import re

def extract_pdf_text(file_path, extraction_func):
    
    """
    Extract text from a PDF file using a specified extraction function.

    Args:
        file_path (str): Path to the PDF file.
        extraction_func (callable): Function that extracts text from the PDF. It should take 
                                    a file path as input and return an iterable of dictionaries 
                                    containing 'text' and 'extraction_time' keys.

    Returns:
        list: A list of dictionaries, each containing:
              - 'path': The file path of the PDF.
              - 'pageNumber': The page number of the extracted text.
              - 'pdfId': A unique identifier for the page in the format "{file_name} ~ {page_no}".
              - 'pageContent': The extracted text content of the page.
              - 'extractionTimeSeconds': The time taken to extract text from the page in seconds.
    """

    extracted_data = []


    for page_no, page_content in enumerate(extraction_func(file_path)):
        extracted_data.append(
            {
                'path': file_path,
                'pageNumber': page_no,
                'pdfId': f"{str(file_path).split('/')[-1]} ~ {page_no}",
                'pageContent': page_content['text'],
                'extractionTimeSeconds': page_content['extraction_time']
            }
        )

    return extracted_data

def pymupdf_extract_page_text(pdf):
    """
    Function to extract text from a PDF using pymupdf
    
    Parameters
    ----------
    pdf : str
        Path to a PDF file
    
    Returns
    -------
    list
        A list of dictionaries, each containing the extracted text and extraction time 
        for each page in the PDF
    """
    page_by_page = []
    doc = pymupdf.open(pdf)

    for page in doc:
        start_time = time.time()
        page_text = page.get_text()
        extraction_time = time.time() - start_time
        page_by_page.append({'text': page_text, 'extraction_time': extraction_time})

    return page_by_page


def pymupdf_extract_pdf_text(pdf):
    """
    Function to extract text from a PDF using pymupdf
    
    Parameters
    ----------
    pdf : str
        Path to a PDF file
    
    Returns
    -------
    list
        A list of dictionaries, each containing the extracted text and extraction time 
        for each page in the PDF
    """
    return extract_pdf_text(pdf, pymupdf_extract_page_text)

def pypdf_extract_page_text(pdf):
    """
    Function to extract text from a PDF using pypdf
    
    Parameters
    ----------
    pdf : str
        Path to a PDF file
    
    Returns
    -------
    list
        A list of dictionaries, each containing the extracted text and extraction time 
        for each page in the PDF
    """
    page_by_page = []
    doc = PdfReader(pdf)

    for page in doc.pages:
        start_time = time.time()
        try: 
            page_text = page.extract_text()
        except NameError as e: 
            page_text = "Page cannot be read"
        extraction_time = time.time() - start_time
        page_by_page.append({'text': page_text, 'extraction_time': extraction_time})

    return page_by_page

def pypdf_extract_pdf_text(pdf):
    """
    Function to extract text from a PDF using pypdf
    
    Parameters
    ----------
    pdf : str
        Path to a PDF file
    
    Returns
    -------
    list
        A list of dictionaries, each containing the extracted text and extraction time 
        for each page in the PDF
    """
    return extract_pdf_text(pdf, pypdf_extract_page_text)

def pdfminer_extract_page_text(pdf):
    """
    Function to extract text from a PDF using pdfminer
    
    Parameters
    ----------
    pdf : str
        Path to a PDF file
    
    Returns
    -------
    list
        A list of dictionaries, each containing the extracted text and extraction time 
        for each page in the PDF
    """
    page_by_page = []

    with open(file_path, 'rb') as f:
        parser = PDFParser(f)
        doc = PDFDocument(parser)
        parser.set_document(doc)
        pages = resolve1(doc.catalog['Pages'])
        pages_count = pages.get('Count', 0)

    for page in range(pages_count):
        start_time = time.time()
        try: 
            page_text = extract_text(file_path, page_numbers=[page])
        except NameError as e: 
            page_text = "Page cannot be read"
        extraction_time = time.time() - start_time
        page_by_page.append({'text': page_text, 'extraction_time': extraction_time})

    return page_by_page

def pdfminer_extract_pdf_text(pdf):
    """
    Helper function to extract text from a pdf using pdfminer.six page-wise.

    Args:
        file_path: Path of the pdf file

    Returns:
        dict: key: page number, value: page text
    """
    return extract_pdf_text(pdf, pypdf_extract_page_text)


def pymupdf_extract_text_final(pdf_bytes: bytes) -> str:
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
