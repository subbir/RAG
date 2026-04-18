LLM_Model = {
    "model_name": "BAAI/bge-large-en-v1.5",
    "llm": "mistralai/Mistral-7B-Instruct-v0.2",
    #"llm": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B instead of 7B
    "max_new_tokens": 2048,
    "max_length": 256,
    "top_k": 3,
    "chunk_size": 300,
    "chunk_overlap": 50
}