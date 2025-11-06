import requests
from requests.exceptions import RequestException, Timeout

def llm_convo(messages, model: str = "gemma3:1b", timeout: int = 30):
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False
            },
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()["message"]["content"]
    except Timeout:
        raise Exception("Request timed out. Please check if Ollama is running.")
    except RequestException as e:
        raise Exception(f"Failed to connect to Ollama: {str(e)}")
    except KeyError:
        raise Exception("Unexpected response format from Ollama")