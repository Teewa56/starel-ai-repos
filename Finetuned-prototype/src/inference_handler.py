import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class InferenceHandler:
    def __init__(self, base_model_id, finetuned_model_dir):
        self.base_model_id = base_model_id
        self.finetuned_model_dir = finetuned_model_dir
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """
        Loads the base model and merges it with the fine-tuned adapter weights.
        """
        print("Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        print("Loading fine-tuned adapter...")
        self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_dir)
        print("Fine-tuned model loaded successfully.")

    def generate_response(self, prompt):
        """
        Generates a response from the fine-tuned model.
        """
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
        return self.tokenizer.decode(output[0], skip_special_tokens=True)