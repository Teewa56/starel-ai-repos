from datasets import load_dataset

def load_and_prepare_dataset(dataset_path):
    """
    Loads a JSONL dataset and returns it in a format suitable for fine-tuning.
    """
    try:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        print("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None