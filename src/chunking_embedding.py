from src.chunking import *
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
import os
import warnings
from sentence_transformers import SentenceTransformer
import pymupdf
import tiktoken
import numpy as np



warnings.filterwarnings("ignore", message=".*resource_tracker.*")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")



def fixed_tokens(text, chunk_size=100, overlap=20):
    """
    Splits a text into chunks by slicing at a fixed interval.

    Args:
        text (str): The text to split.
        chunk_size (int, optional): The size of each chunk. Defaults to 100.
        overlap (int, optional): The amount of overlap between each chunk. Defaults to 20.

    Returns:
        list[str]: A list of chunks of text.
    """
    step = chunk_size - overlap
    chunks = []

    for i in range(0, len(text), step):
        chunk = text[i:i+chunk_size]   # slice by characters
        chunks.append(chunk)

        if i + chunk_size >= len(text):
            break

    return chunks

def recursive_char_split(text, chunk_size=750, overlap=70, 
                         separators=None, min_chunk_size=None):
    """
    Recursively splits text into chunks using progressively smaller separators.
    Ensures overlap between chunks and avoids cutting in the middle of sentences when possible.

    Args:
        text (str): The text to split.
        chunk_size (int, optional): Maximum chunk length. Defaults to 200.
        overlap (int, optional): Overlap size between chunks. Defaults to 20.
        separators (list, optional): List of separators (from largest to smallest). 
                                     Defaults to ["\n\n", ". ", " ", ""].
        min_chunk_size (int, optional): Minimum acceptable chunk size before merging. 
                                        Defaults to 0.2 * chunk_size.

    Returns:
        list[str]: List of text chunks.
    """
    if separators is None:
        separators = ["\n\n", ". ", " ", ""]

    if min_chunk_size is None:
        min_chunk_size = int(0.2 * chunk_size)

    chunks = [text]

    for sep in separators:
        new_chunks = []
        for c in chunks:
            if len(c) <= chunk_size:
                new_chunks.append(c)
                continue

            if sep:  # split by separator
                parts, cur = [], ""
                for p in c.split(sep):
                    piece = (p + sep).strip()
                    if not piece:
                        continue
                    if len(cur) + len(piece) <= chunk_size:
                        cur += piece + " "
                    else:
                        if cur:
                            new_chunks.append(cur.strip())
                        cur = piece
                if cur:
                    new_chunks.append(cur.strip())
            else:  # final fallback: split by characters
                step = chunk_size - overlap
                for i in range(0, len(c), step):
                    new_chunks.append(c[i:i+chunk_size].strip())

        chunks = new_chunks

    # --- Post-processing: merge small orphan chunks ---
    final_chunks = []
    for chunk in chunks:
        if final_chunks and len(chunk) < min_chunk_size:
            final_chunks[-1] += " " + chunk
        else:
            final_chunks.append(chunk)

    return final_chunks

def chunk_text(text, goal="exact_size", chunk_size=100, overlap=20):
    """
    Automatically selects which chunking function to use based on the goal.

    Parameters:
    - text (str): Text to chunk.
    - goal (str): "exact_size" for fixed token size,
                  "semantic" for preserving context,
                  "hybrid" for both.
    - chunk_size (int): Desired chunk size (characters).
    - overlap (int): Number of characters to overlap between chunks.
    
    Returns:
    - List of text chunks.
    """
    if goal == "exact_size":
        return fixed_tokens(text, chunk_size=chunk_size, overlap=overlap)
    
    elif goal == "semantic":
        return recursive_char_split(text, chunk_size=chunk_size, overlap=overlap)
    
    elif goal == "hybrid":
        # Step 1: semantic split
        semantic_chunks = recursive_char_split(text, chunk_size=chunk_size*2, overlap=overlap)
        # Step 2: enforce exact size on large chunks
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk) > chunk_size:
                final_chunks.extend(fixed_tokens(chunk, chunk_size=chunk_size, overlap=overlap))
            else:
                final_chunks.append(chunk)
        return final_chunks
    
    else:
        raise ValueError("Invalid goal. Choose 'exact_size', 'semantic', or 'hybrid'.")

def get_embeddings(chunks, batch_size=8):
    """
    Calculate the embeddings of a list of text chunks.

    Parameters:
    - chunks (list): List of text chunks
    - batch_size (int): Batch size for encoding. Defaults to 8.

    Returns:
    - embeddings (list): List of embeddings for the input text chunks
    """
    return model.encode(chunks, convert_to_tensor=True, show_progress_bar=True, batch_size=batch_size)

def cosine_sim(a, b):
    """
    Calculate the cosine similarity between two vectors.

    Parameters:
    a (list): First vector
    b (list): Second vector

    Returns:
    float: Cosine similarity between the two vectors
    """
    if len(a) != len(b):
        raise ValueError("Embedding lengths do not match")
    
    dot_value = 0
    magnitude_a = 0
    magnitude_b = 0

    for i, j in zip(a, b):
        dot_value += i * j
        magnitude_a += i**2
        magnitude_b += j**2
    
    return dot_value / (magnitude_a**0.5 * magnitude_b**0.5)
