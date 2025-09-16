Fine-tuning Prototype
----------------------------
The fine-tuning prototype shows how we can adapt a general-purpose AI model to a specific task or conversational style. While a RAG system provides knowledge, fine-tuning helps the AI to sound more helpful, empathetic, or knowledgeable about a particular subject based on how it's trained.

How it Works
--------------------------
Instead of training a model from scratch, we take a pre-trained, open-source model and update its weights using a small, custom dataset.  The goal is to teach the model to consistently generate responses that match the tone and format of our training data. We will use a technique called Parameter-Efficient Fine-Tuning (PEFT), which is designed to make this process much faster and less resource-intensive.

Data Set
--------------------------
The data set consist of roles , questions and responses that are common and specific for students
in the Federal university of technology, Akure, Nigeria

Key Technologies
--------------------------
Model: A small, efficient open-source LLM from Hugging Face, such as TinyLlama/TinyLlama-1.1B-Chat-v1.0 or google/gemma-2b-it.

Dataset: A small, custom dataset of question-and-answer pairs that are specific to the experience of Nigerian students. This dataset will be in a .jsonl format.

Frameworks: We will primarily use the Hugging Face transformers and peft libraries, which provide the tools needed to train the model efficiently.

Getting Started
--------------------------
First, you'll need to run the training script. This will create a fine-tuned version of the model.

Bash
--------------------------
cd finetune_prototype
pip install -r requirements.txt
python train_finetune_model.py

After training is complete, you can interact with the fine-tuned model by running the inference script.

Bash
--------------------------
python run_finetune_model.py