import faiss
import numpy as np
import torch

class Retriever:
    def __init__(self, embeddings, documents, embedding_model):
        self.documents = documents
        self.embedding_model = embedding_model
        
        # Convert embeddings to numpy array for FAISS
        embeddings_np = embeddings.cpu().numpy().astype('float32')
        dimension = embeddings_np.shape[1]
        
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)

    def retrieve(self, query, top_k=3):
        """
        Takes a query, generates its embedding, and searches the index for
        the top_k most similar document chunks.
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True).cpu().numpy().astype('float32').reshape(1, -1)
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the actual document chunks based on indices
        retrieved_chunks = [self.documents[i] for i in indices[0]]
        
        return retrieved_chunks