import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from .web_scraper import FetchFromNet
from .secure_input import SecurePrompt

class RAGSystem:
    def __init__(self, retriever):
        self.retriever = retriever
        self.accelerator = Accelerator()
        self.webscraper = FetchFromNet()
        self.checkPrompt = SecurePrompt()
        
        model_name = "google/gemma-2b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" 
        )
        
        self.model, self.tokenizer = self.accelerator.prepare(self.model, self.tokenizer)

    def generate_response(self, query):
        """
        Performs retrieval and then generates a response with web search augmentation.
        """
        # Step 1: Check if prompt is safe
        is_safe = self.checkPrompt.screen_prompt(query)
        if is_safe.lower().strip() != "yes":
            return "Sorry, I don't have the permission to process this request."
        
        # Step 2: Retrieve relevant documents from local knowledge base
        retrieved_chunks = self.retriever.retrieve(query)
        local_context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        # Step 3: Get additional information from web search
        web_summary = self.webscraper.get_search_summary(query)
        
        # Step 4: Create a comprehensive prompt
        prompt = f"""
            Answer the following question using both the provided context and additional web information.
            Prioritize the context information, but supplement with web information when relevant.
            If you cannot find a complete answer, state what you know and mention the limitations.

            Local Knowledge Base Context:
            {local_context}

            Additional Web Information:
            {web_summary}

            Question: {query}

            Answer:
            """
        
        # Step 5: Generate the response from the LLM
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Step 6: Post-process the response to remove the prompt
        response_start_index = response.find("Answer:") + len("Answer:")
        final_response = response[response_start_index:].strip()
        
        # Clean up the response
        if not final_response or len(final_response) < 10:
            final_response = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return final_response

    def generate_response_local_only(self, query):
        """
        Generate response using only local knowledge base (no web search)
        """
        # Check if prompt is safe
        is_safe = self.checkPrompt.screen_prompt(query)
        if is_safe.lower().strip() != "yes":
            return "Sorry, I don't have the permission to process this request."
        
        # Retrieve relevant documents
        retrieved_chunks = self.retriever.retrieve(query)
        context = "\n".join([chunk['text'] for chunk in retrieved_chunks])
        
        prompt = f"""
            Answer the following question based only on the provided context. 
            If the answer cannot be found in the context, state 
            "I'm sorry, I cannot find the answer to that in my knowledge base."

            Context:
            {context}

            Question: {query}

            Answer:
            """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_start_index = response.find("Answer:") + len("Answer:")
        final_response = response[response_start_index:].strip()
        
        return final_response if final_response else "I couldn't generate a proper response. Please try again."