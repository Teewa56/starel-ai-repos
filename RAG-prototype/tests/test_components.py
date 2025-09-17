import traceback

def test_document_loading():
    """Test document loading"""
    print("\n🔍 Testing Document Loading...")
    try:
        from src.document_processor import load_documents
        documents = load_documents(directory="data")
        print(f"✅ Loaded {len(documents)} documents")
        for i, doc in enumerate(documents[:2]):  # Show first 2
            print(f"   Document {i+1}: {doc['source']} ({len(doc['text'])} characters)")
        return True, documents
    except Exception as e:
        print(f"❌ Document loading failed: {e}")
        traceback.print_exc()
        return False, None

def test_document_chunking(documents):
    """Test document chunking"""
    print("\n📄 Testing Document Chunking...")
    try:
        from src.document_processor import chunk_documents
        chunked_docs = chunk_documents(documents)
        print(f"✅ Created {len(chunked_docs)} chunks from {len(documents)} documents")
        print(f"   Sample chunk: {chunked_docs[0]['text'][:100]}...")
        return True, chunked_docs
    except Exception as e:
        print(f"❌ Document chunking failed: {e}")
        traceback.print_exc()
        return False, None

def test_embedding_generation(chunked_docs):
    """Test embedding generation"""
    print("\n🧠 Testing Embedding Generation...")
    try:
        from src.document_processor import generate_embeddings
        print("   Loading sentence transformer model...")
        embeddings, embedding_model = generate_embeddings(chunked_docs[:5])  # Test with first 5 chunks
        print(f"✅ Generated embeddings with shape: {embeddings.shape}")
        return True, embeddings, embedding_model
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        traceback.print_exc()
        return False, None, None

def test_retriever(embeddings, chunked_docs, embedding_model):
    """Test retriever functionality"""
    print("\n🔍 Testing Retriever...")
    try:
        from src.retriever import Retriever
        retriever = Retriever(embeddings, chunked_docs[:5], embedding_model)
        
        test_query = "What is FUTA?"
        results = retriever.retrieve(test_query, top_k=2)
        print(f"✅ Retrieved {len(results)} chunks for query: '{test_query}'")
        for i, result in enumerate(results):
            print(f"   Result {i+1}: {result['text'][:100]}...")
        return True, retriever
    except Exception as e:
        print(f"❌ Retriever failed: {e}")
        traceback.print_exc()
        return False, None

def test_web_scraper():
    """Test web scraping functionality"""
    print("\n🌐 Testing Web Scraper...")
    try:
        from src.web_scraper import FetchFromNet
        scraper = FetchFromNet()
        
        # Test DuckDuckGo search
        results = scraper.search_duckduckgo("FUTA university")
        print(f"✅ Web scraper returned {len(results)} results")
        if results:
            print(f"   Sample result: {results[0]}")
        return True
    except Exception as e:
        print(f"❌ Web scraper failed: {e}")
        traceback.print_exc()
        return False

def test_secure_input():
    """Test security checking"""
    print("\n🔒 Testing Security Input Checker...")
    try:
        from src.secure_input import SecurePrompt
        checker = SecurePrompt()
        
        # This might fail if API_KEY is not set - that's okay
        print("   Note: This test requires API_KEY environment variable")
        print("✅ Security module imported successfully")
        return True
    except Exception as e:
        print(f"❌ Security checker failed: {e}")
        print("   This is okay if API_KEY is not set")
        return False

def main():
    """Run all component tests"""
    print("🚀 Starting Component Tests for RAG System")
    print("=" * 50)
    
    # Test 1: Document Loading
    success, documents = test_document_loading()
    if not success:
        print("❌ Cannot proceed without documents")
        return
    
    # Test 2: Document Chunking  
    success, chunked_docs = test_document_chunking(documents)
    if not success:
        print("❌ Cannot proceed without chunked documents")
        return
    
    # Test 3: Embedding Generation
    success, embeddings, embedding_model = test_embedding_generation(chunked_docs)
    if not success:
        print("❌ Cannot proceed without embeddings")
        return
    
    # Test 4: Retriever
    success, retriever = test_retriever(embeddings, chunked_docs, embedding_model)
    if not success:
        print("❌ Retriever test failed")
    
    # Test 5: Web Scraper
    test_web_scraper()
    
    # Test 6: Security Checker
    test_secure_input()
    
    print("\n" + "=" * 50)
    print("🎉 Component testing completed!")
    print("   Next: Run full system tests")

if __name__ == "__main__":
    main()