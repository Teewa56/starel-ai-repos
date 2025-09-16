Retrieval-Augmented Generation (RAG) Prototype
----------------------------------------------
The RAG prototype demonstrates a powerful approach for providing factual and context-specific answers by using external knowledge sources. This is a crucial method for ensuring the AI assistant gives accurate information and avoids making up facts, a common issue with general-purpose AI.

How it Works
-------------------------------------------------
The process is like a student using a textbook to answer a question. When a student asks the AI about a specific topic, the system first 'looks up' the most relevant information from a small, curated set of documents. It then uses this retrieved information as a reference to formulate a precise and helpful response. The AI's knowledge is thus augmented by the retrieved documents.

Key Technologies
-------------------------------------------------
Models: We will use Hugging Face's Sentence-Transformers to convert text into numerical embeddings, which are then used for semantic search. For the generative part, we'll use a small, open-source large language model (LLM) such as google/gemma-2b-it.

Vector Store: Faiss-cpu will be used as an in-memory vector database. It's an efficient tool for finding the most relevant documents based on their embeddings.

Data: We'll use sample documents like snippets from a Nigerian curriculum, publicly available spiritual texts, and a generalized school handbook. These will be stored as simple text files in the rag_prototype/data/ directory.

Getting Started
-------------------------------------------------
To run the RAG prototype, navigate to its folder and install the dependencies.

Bash
-------------------------------------------------
cd rag_prototype
pip install -r requirements.txt
python main_rag.py

Also you need to get a token from higging face so to do that
1. Get a Hugging Face Token:

Go to your Hugging Face account settings at https://huggingface.co/settings/tokens.

Generate a new Access Token with at least a read role. Copy this token.

2. Log in from your terminal:

In your command prompt or terminal, run the following command:

Bash

huggingface-cli login

and do what you are asked to do
also go to https://huggingface.co/google/gemma-2b-it and authorize access

The terminal will prompt you to enter a question, and the system will provide an answer based on the documents it has access to.