<<<<<<< HEAD
# starel-ai-repos
This Repo will contain the prototype for starels AI contributions
=======
Starel AI Prototype
----------------------------
Overview
----------------------------
This repository contains the prototype for Starel's initial AI contributions. The project explores two distinct but complementary approaches to building powerful language model applications: Retrieval-Augmented Generation (RAG) and the fine-tuning of a free, open-source model.

The goal is to demonstrate the capabilities and trade-offs of each method, providing a foundation for future, more complex AI projects at Starel.

Project Structure & Focus Areas
-------------------------------
The prototype is divided into two main sections:

Retrieval-Augmented Generation (RAG):
-------------------------------------
Goal: To build a system that can answer questions based on a specific set of documents, going beyond the model's pre-trained knowledge.

Technology Stack: This prototype will leverage the Hugging Face ecosystem, which provides robust tools and pre-trained models for NLP tasks. We will use a pre-trained model for the generative part and a separate embedding model for the retrieval part.

Core Components:
------------------
Document Loading: Ingesting and processing a corpus of documents (e.g., PDFs, text files, markdown).

Chunking & Embedding: Splitting documents into smaller, manageable chunks and converting them into numerical representations (embeddings).

Vector Store: Storing the document embeddings in a vector database for efficient semantic search.

Retrieval: When a user poses a question, the system will retrieve the most relevant document chunks from the vector store.

Generation: The retrieved chunks will be passed to a large language model (LLM) as context, allowing it to generate an informed and accurate response.

Fine-tuning of a Free Model:
----------------------------
Goal: To adapt a general-purpose, pre-trained language model to a specific task or a particular style of output.

Technology Stack: We will use a "totally free" model, which in the context of open-source AI, typically refers to a model with a permissive license that allows for commercial use without cost. Note: The user's prompt mentioned "chat-gpt free model," but free-to-use versions of ChatGPT (from OpenAI) are not typically available for direct fine-tuning. This prototype will instead focus on fine-tuning a powerful, freely available open-source model from the Hugging Face Hub, as this is the standard practice for this type of task.

Core Components:
----------------------------
Model Selection: Identifying a suitable open-source model with a permissive license (e.g., a variant of Llama, Gemma, or Mistral).

Dataset Preparation: Creating a high-quality dataset of prompt-completion pairs to train the model on the desired task or style.

Training Script: Implementing a training loop using libraries like Hugging Face's transformers and peft (Parameter-Efficient Fine-Tuning) to efficiently update the model's weights.

Evaluation: Assessing the fine-tuned model's performance on a held-out test set to measure its success.

Getting Started
Prerequisites
Python 3.8+

pip package manager

Folder Structure
-------------------
starel-ai-repos/
├── rag-prototype/
│   ├── data/
│   │   ├── academic_docs/
│   │   │   └── nigeria_curriculum_excerpt.txt
│   │   ├── spiritual_texts/
│   │   │   └── sample_spiritual_guidance.txt
│   │   └── school_info/
│   │       └── sample_school_handbook.txt
│   ├── src/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Handles loading, chunking, embedding
│   │   ├── retriever.py           # Manages vector store and retrieval logic
│   │   └── rag_system.py          # Orchestrates retrieval and generation
│   ├── main_rag.py              # Entry point for RAG interaction
│   └── requirements.txt         # RAG-specific dependencies
│
├── finetuned-prototype/
│   ├── data/
│   │   └── nigeria_student_qa.jsonl  # Dataset for fine-tuning
│   ├── src/
│   │   ├── __init__.py
│   │   ├── dataset_preparer.py    # Prepares data for fine-tuning
│   │   ├── model_trainer.py       # Handles model loading and training loop
│   │   └── inference_handler.py   # For running inference on fine-tuned model
│   ├── train_finetune_model.py  # Script to initiate fine-tuning
│   ├── run_finetune_model.py    # Script to interact with fine-tuned model
│   └── requirements.txt         # Fine-tuning specific dependencies
| 
└── README.md

Installation
Clone the repository:

Bash
-------------------
git clone https://github.com/Teewa56/starel-ai-repos.git
cd starel-ai-prototype
Create and activate a virtual environment:

Bash
--------------------
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:

Bash
---------------------
pip install -r requirements.txt

Usage
---------------------
For RAG: Navigate to the rag-prototype/ directory. Follow the instructions in rag-prototype/README.md to run the document ingestion and query components.

For Fine-tuning: Navigate to the finetuned-prototype/ directory. Follow the instructions in finetuned-prototype/README.md to prepare your dataset and train the model.
>>>>>>> master
