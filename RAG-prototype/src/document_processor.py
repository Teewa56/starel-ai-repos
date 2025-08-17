import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

def load_documents(directory="data"):
    """
    Loads all text files from the specified directory and its subdirectories.
    Returns a list of dictionaries with 'text' and 'source'.
    """
    documents = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    documents.append({"text": text, "source": file_path})
    return documents

def chunk_documents(documents):
    """
    Splits documents into smaller chunks for better retrieval.
    """
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    chunked_docs = []
    max_length = 256  # Max tokens per chunk

    for doc in documents:
        tokens = tokenizer.tokenize(doc['text'])
        # A simple chunking method: split into chunks of max_length tokens
        for i in range(0, len(tokens), max_length):
            chunk_tokens = tokens[i:i+max_length]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunked_docs.append({
                "text": chunk_text,
                "source": doc['source']
            })
    return chunked_docs

def generate_embeddings(chunked_docs):
    """
    Generates embeddings for each document chunk.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [doc['text'] for doc in chunked_docs]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings, model