import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

class RAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        self.accelerator = Accelerator()
        
        # Load the LLM and tokenizer
        model_name = "google/gemma-2b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" # Automatically places the model on available devices (GPU/CPU)
        )
        
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)

    def generate_response(self, query):
        """
        Performs retrieval and then generates a response.
        """
        # Step 1: Retrieve relevant documents
        retrieved_chunks = self.retriever.retrieve(query)
        
        # Step 2: Create a prompt with the retrieved context
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        prompt = f"""
        Answer the following question based only on the provided context. If the answer cannot be found in the context, state "I'm sorry, I cannot find the answer to that in my knowledge base."

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        
        # Step 3: Generate the response from the LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Post-process the response to remove the prompt
        response_start_index = response.find("Answer:") + len("Answer:")
        final_response = response[response_start_index:].strip()
        
        return final_response