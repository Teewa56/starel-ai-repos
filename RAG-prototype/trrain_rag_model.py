from src.document_processor import load_documents, chunk_documents, generate_embeddings
from src.retriever import Retriever
from src.rag_system import RAGSystem

def main(query):
    print("Initializing RAG System...")
    #make sure to save the model after training so as not to retrain every time, 
    # query the saved model if needed so you check for it first

    # 1. Load and process documents
    documents = load_documents(directory="data")
    chunked_docs = chunk_documents(documents)
    
    # 2. Generate embeddings for the chunks
    embeddings, embedding_model = generate_embeddings(chunked_docs)
    
    # 3. Initialize the retriever and RAG system
    retriever = Retriever(embeddings, chunked_docs, embedding_model)
    rag_system = RAGSystem(retriever)
    print("RAG System is ready. You can now ask questions.")
    print("Type 'exit' to quit.")
    
    while True:
        if query.lower() == 'exit':
            break
        print(".......Thinking!........")
        response = rag_system.generate_response(query)
        print(f"Response: {response}")
    return response

if __name__ == "__main__":
    main("When was FUTA created?")