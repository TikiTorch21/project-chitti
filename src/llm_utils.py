import requests

def llm_convo(messages, model: str = "gemma3:1b"):
    """
    A continuous chat loop using Gemma via Ollama.
    Remembers conversation in this session only.
    Type 'exit' or 'quit' to stop.
    """

    # Send request with full message history
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    reply = response.json()["message"]["content"]

    return reply