import os
from src.inference_handler import InferenceHandler

def main():
    # Configuration
    base_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    finetuned_model_dir = "./finetuned_model"

    print("Initializing inference handler...")
    handler = InferenceHandler(base_model_id, finetuned_model_dir)
    
    print("\nModel is ready. Type 'exit' to quit.")
    while True:
        prompt = input("\nEnter Your Question: ")
        if prompt.lower() == "exit":
            break
        
        response = handler.generate_response(prompt)
        print(f"Model: {response}")

if __name__ == "__main__":
    main()