from chunking import *
import torch
from transformers import AutoTokenizer, AutoModel

model_name = "nvidia/NV-Embed-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)


def get_embeddings(chunks, batch_size=8):
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        all_embeddings.append(embeddings.cpu())
    
    return torch.cat(all_embeddings, dim=0)  # Shape: (num_chunks, embedding_dim)