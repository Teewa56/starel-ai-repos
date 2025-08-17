import os
from src.dataset_preparer import load_and_prepare_dataset
from src.model_trainer import ModelTrainer

def main():
    # Configuration
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    dataset_path = "./data/nigeria_student_qa.json"
    output_dir = "./finetuned_model"

    print("Step 1: Loading and preparing dataset...")
    dataset = load_and_prepare_dataset(dataset_path)
    if dataset is None:
        return

    print("Step 2: Initializing and training model...")
    trainer = ModelTrainer(model_id, output_dir)
    trainer.train_model(dataset)

    print("\nTraining process finished.")

if __name__ == "__main__":
    main()