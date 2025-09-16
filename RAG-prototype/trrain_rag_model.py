import os
import pickle
import hashlib
from src.document_processor import load_documents, chunk_documents, generate_embeddings
from src.retriever import Retriever
from src.rag_system import RAGSystem

class RAGManager:
    def __init__(self, data_directory="data", cache_file="rag_cache.pkl"):
        self.data_directory = data_directory
        self.cache_file = cache_file
        self.rag_system = None
        
    def _get_data_hash(self, documents):
        """Generate hash of document content to detect changes"""
        content = ""
        for doc in documents:
            content += doc['text'] + doc['source']
        return hashlib.md5(content.encode()).hexdigest()
    
    def _save_rag_components(self, embeddings, chunked_docs, embedding_model, data_hash):
        """Save RAG components to cache file"""
        try:
            cache_data = {
                'embeddings': embeddings,
                'chunked_docs': chunked_docs,
                'embedding_model': embedding_model,
                'data_hash': data_hash,
                'cache_version': '1.0'
            }
            
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"RAG components saved to {self.cache_file}")
            
        except Exception as e:
            print(f"Error saving cache: {e}")
    
    def _load_rag_components(self):
        """Load RAG components from cache file"""
        if not os.path.exists(self.cache_file):
            return None
            
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache has required keys
            required_keys = ['embeddings', 'chunked_docs', 'embedding_model', 'data_hash']
            if not all(key in cache_data for key in required_keys):
                print("Cache file is corrupted or outdated")
                return None
                
            print("RAG components loaded from cache")
            return cache_data
            
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def _initialize_rag_system(self, force_rebuild=False):
        """Initialize or load RAG system with caching"""
        print("Initializing RAG System...")
        
        # 1. Load documents
        print("Loading documents...")
        documents = load_documents(directory=self.data_directory)
        
        if not documents:
            raise ValueError(f"No documents found in {self.data_directory}")
        
        data_hash = self._get_data_hash(documents)
        
        # 2. Try to load from cache - THIS IS WHERE CACHE IS USED!
        cached_data = None
        if not force_rebuild:
            print("🔍 Checking for cached RAG components...")
            cached_data = self._load_rag_components()  # CACHE LOADING HERE
            
        # 3. Check if cache is valid - CACHE VALIDATION
        use_cache = (
            cached_data is not None and
            cached_data.get('data_hash') == data_hash
        )
        
        if use_cache:
            print("✅ Using cached RAG components (CACHE HIT)...")
            # USING CACHED DATA INSTEAD OF REBUILDING
            embeddings = cached_data['embeddings']
            chunked_docs = cached_data['chunked_docs']
            embedding_model = cached_data['embedding_model']
        else:
            print("❌ Cache miss - Building RAG components from scratch...")
            
            # Process documents
            print("Chunking documents...")
            chunked_docs = chunk_documents(documents)
            
            # Generate embeddings (EXPENSIVE OPERATION - AVOIDED WITH CACHE)
            print("Generating embeddings...")
            embeddings, embedding_model = generate_embeddings(chunked_docs)
            
            # Save to cache - CACHE SAVING HERE
            print("💾 Saving to cache for future use...")
            self._save_rag_components(embeddings, chunked_docs, embedding_model, data_hash)
        
        # 4. Initialize retriever and RAG system
        print("Initializing retriever...")
        retriever = Retriever(embeddings, chunked_docs, embedding_model)
        
        print("Initializing RAG system...")
        self.rag_system = RAGSystem(retriever)
        
        print("RAG System initialized successfully!")
        return self.rag_system
    
    def get_rag_system(self, force_rebuild=False):
        """Get initialized RAG system"""
        if self.rag_system is None or force_rebuild:
            self._initialize_rag_system(force_rebuild)
        return self.rag_system
    
    def clear_cache(self):
        """Clear the cache file"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                print(f"Cache file {self.cache_file} removed")
            else:
                print("No cache file to remove")
        except Exception as e:
            print(f"Error removing cache: {e}")

# Global RAG manager instance
_rag_manager = None

def get_rag_manager():
    """Get global RAG manager instance"""
    global _rag_manager
    if _rag_manager is None:
        _rag_manager = RAGManager()
    return _rag_manager

def main(query, force_rebuild=False):
    """
    Main function to get response from RAG system
    
    Args:
        query (str): User query
        force_rebuild (bool): Force rebuild of RAG components
    
    Returns:
        str: Generated response
    """
    try:
        if not query or not isinstance(query, str):
            return "Error: Invalid query provided"
        
        # Get RAG manager and system
        rag_manager = get_rag_manager()
        rag_system = rag_manager.get_rag_system(force_rebuild=force_rebuild)
        
        # Generate response
        print(f"Processing query: {query[:50]}...")
        print("Thinking...")
        response = rag_system.generate_response(query)
        
        print(f"Response generated successfully")
        return response
        
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(error_msg)
        return error_msg

def rebuild_rag_system():
    """Force rebuild RAG system (useful for updates)"""
    try:
        rag_manager = get_rag_manager()
        rag_manager.clear_cache()
        rag_manager.get_rag_system(force_rebuild=True)
        print("RAG system rebuilt successfully")
    except Exception as e:
        print(f"Error rebuilding RAG system: {e}")

if __name__ == "__main__":
    # Test the system
    test_queries = [
        "When was FUTA created?",
        "What faculties are available in FUTA?",
        "Tell me about student life in FUTA"
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"{'='*50}")
        response = main(query)
        print(f"Response: {response}")