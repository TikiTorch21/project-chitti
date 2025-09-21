import tiktoken
import numpy as np
from openai import OpenAI


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


def recursive_char_split(text, chunk_size=200, overlap=20):
    """
    Recursively splits a text into chunks by splitting at a given character (or list of characters), 
    and re-splitting the resulting chunks until they are all below a given size.

    Args:
        text (str): The text to split.
        chunk_size (int, optional): The maximum size of the chunks. Defaults to 200.
        overlap (int, optional): The amount of overlap between chunks. Defaults to 20.

    Returns:
        list: A list of the resulting chunks.
    """
    separators = ["\n\n", ". ", " ", ""]
    chunks = [text]

    for sep in separators:
        new_chunks = []
        for c in chunks:
            if len(c) <= chunk_size:
                new_chunks.append(c)
            elif sep:
                parts, cur = [], ""
                for p in c.split(sep):
                    piece = p + sep
                    if len(cur) + len(piece) <= chunk_size:
                        cur += piece
                    else:
                        if cur: new_chunks.append(cur)
                        cur = piece
                if cur: new_chunks.append(cur)
            else:  
                step = chunk_size - overlap
                for i in range(0, len(c), step):
                    new_chunks.append(c[i:i+chunk_size])
        chunks = new_chunks
    return chunks



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

