import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

class ModelTrainer:
    def __init__(self, model_id, output_dir):
        self.model_id = model_id
        self.output_dir = output_dir

    def train_model(self, dataset):
        """
        Sets up and runs the fine-tuning process on the provided dataset.
        """
        # Load Base Model and Tokenizer with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model = prepare_model_for_kbit_training(model)

        # Configure PEFT (LoRA)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=1,
            save_strategy="epoch",
            report_to="none"
        )

        # Initialize Trainer and Start Training
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            tokenizer=tokenizer,
            args=training_args,
            max_seq_length=512,
        )
        trainer.train()

        # Save the LoRA Adapter
        trainer.save_model(self.output_dir)
        print("\nFine-tuning complete! Model saved to:", self.output_dir)