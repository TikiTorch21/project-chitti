import requests

def chat_with_gemma(model: str = "gemma3:1b"):
    """
    A continuous chat loop using Gemma via Ollama.
    Remembers conversation in this session only.
    Type 'exit' or 'quit' to stop.
    """
    messages = []

    print("💬 Chat started with Gemma (type 'exit' to quit):\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Ending chat. Goodbye!")
            break

        # Append user's message to history
        messages.append({"role": "user", "content": f"{user_input}. Answer in 2 lines please."})

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

        print(f"Gemma: {reply}\n")

        # Append assistant reply to history so it remembers context
        messages.append({"role": "assistant", "content": reply})
    
    return messages

chat_with_gemma()