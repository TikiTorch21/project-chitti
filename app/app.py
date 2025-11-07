import sys
import os

project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_path)
from src.utils.pdf_utils import *
from src.chunking_embedding import *
from src.llm_utils import *
import pymupdf
from PIL import Image
from datetime import datetime

import streamlit as st

# ------------ Config ------------
MAX_FILES = 3
MAX_FILE_SIZE = 15 * 1024 * 1024  # 5 MB
PROMPT = """
You are a helpful teaching assistant. Your task is to answer the user's question given the context of the information. 
Always use the information given to you to answer the question, and do not make anything up. 

Question: {}

Context: {}
"""


# ------------ Helper ------------


@st.cache_data(show_spinner=False)
def render_page_image(
    pdf_bytes: bytes, page_number: int = 0, zoom: float = 2.0
) -> Image.Image:
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pix = doc.load_page(page_number).get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    mode = "RGBA" if pix.alpha else "RGB"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


# ------------ St UI ------------
st.set_page_config(page_title="Project Chitti", layout="centered")
st.title("Project Chitti")

with st.sidebar:
    # File uploader
    uploaded_files = st.file_uploader(
        f"Upload up to a {MAX_FILES} files. (< {MAX_FILE_SIZE // 1024 // 1024} MB each)",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        # Enforce file count limit
        if len(uploaded_files) > MAX_FILES:
            st.error(f"Please upload at max {MAX_FILES} files.")
        else:
            progress = st.progress(0)
            total = len(uploaded_files)
            for idx, uploaded_file in enumerate(uploaded_files, start=1):
                st.header(uploaded_file.name)

                # Check if it meets the allowed size threshold
                if uploaded_file.size > MAX_FILE_SIZE:
                    st.warning(
                        f"{uploaded_file.name} is > than {MAX_FILE_SIZE // 1024 // 1024}"
                    )
                else:
                    pdf_bytes = uploaded_file.read()

                    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
                    page_count = len(doc)
                    page_idx = st.slider(
                        "Choose page", min_value=1, max_value=page_count, value=1
                    )
                    try:
                        img = render_page_image(
                            pdf_bytes, page_number=page_idx - 1, zoom=1.0
                        )
                        st.image(
                            img, caption=f"Page {page_idx-1}", use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Could not render page image. Error: {e}")

                    # 2 - Extract & Display the PDF text
                    with st.expander("Show extracted text"):
                        text = clean_pdf_text(text=extract_text(pdf_bytes))
                        if text.strip():
                            st.text_area("Full text", text, height=600)
                        else:
                            st.write("_ NO TEXT FOUND _")

                    chunks = chunk_text(text=text)
                    pdf_embeddings = get_embeddings(chunks=chunks)
                    if "file_data" not in st.session_state:
                        st.session_state.file_data = {}

                    st.session_state.file_data[uploaded_file.name] = {
                        "chunks": chunks,
                        "embeddings": pdf_embeddings,
                        "text": text,
                    }
                    progress.progress(int(idx / total * 100))

                progress.empty()

# Chatbot portion


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Create main chat container
chat_container = st.container()

# Display all chat messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            st.caption(f"*{message['timestamp']}*")

# Chat input (always at bottom)
if prompt := st.chat_input("Ask Chitti"):
    if "file_data" not in st.session_state or not st.session_state.file_data:
        st.error("Please upload a PDF file first!")
        st.stop()

    # Embed prompt when it's entered
    prompt_embeddings = get_embeddings(chunks=prompt)

    # Get sim score of prompt embedding & pdf embeddings, then get top 3 chunks and their ids
    embedding_sim_score: dict = {
        c: cosine_sim(ce, prompt_embeddings) for c, ce in enumerate(pdf_embeddings)
    }
    relevant_chunk_ids = sorted(
        embedding_sim_score, key=embedding_sim_score.get, reverse=True
    )[:3]
    relevant_chunks = [chunks[id] for id in relevant_chunk_ids]

    for idx, chunk in enumerate(relevant_chunks):
        formatted_chunks_str = ""  # Reset for each new query
        formatted_chunks_str += f"CHUNK #{idx+1}: \n{chunk} \n\n"

    # Add user message to chat history
    timestamp = datetime.now().strftime("%H:%M:%S")
    user_message = {"role": "user", "content": prompt, "timestamp": timestamp}

    st.session_state.messages.append(user_message)

    # Display the new user message immediately
    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)
            st.caption(f"*{timestamp}*")

    temp_messages = st.session_state.messages.copy()

    temp_messages[-1] = {
        "role": "user",
        "content": PROMPT.format(prompt, formatted_chunks_str),
    }

    # Display formatted prompt..
    reply = llm_convo(messages=temp_messages)
    reply_text = f"{reply}"
    reply_timestamp = datetime.now().strftime("%H:%M:%S")

    assistant_message = {
        "role": "assistant",
        "content": reply_text,
        "timestamp": reply_timestamp,
    }
    st.session_state.messages.append(assistant_message)

    # Display assistant reply immediately
    with chat_container:
        with st.chat_message("assistant"):
            st.write(reply_text)
            st.caption(f"*{reply_timestamp}*")

    # Rerun to refresh the display
    st.rerun()


col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
