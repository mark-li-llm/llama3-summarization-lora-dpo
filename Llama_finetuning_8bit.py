#!/usr/bin/env python3
"""
Distributed LoRA Fine-tuning with 8-bit Quantization
====================================================

High-performance distributed training implementation for large language models 
using LoRA adapters and 8-bit quantization. Optimized for multi-GPU environments
with data parallelism and memory efficiency.

Key Features:
    • Distributed training with Accelerate framework
    • 8-bit quantization for memory optimization
    • LoRA adapters for parameter-efficient fine-tuning
    • Early stopping and comprehensive logging
    • Production-ready checkpoint management

Usage:
    accelerate launch \
        --num_processes 2 \
        --mixed_precision fp16 \
        distributed_lora_trainer.py \
        --train_json ./data/train.json \
        --val_json ./data/val.json \
        --output_dir ./models/lora-finetuned

Dependencies:
    pip install transformers>=4.51.3 peft bitsandbytes accelerate datasets
"""

import argparse
import json
import os
import time
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


class LossLoggingCallback(TrainerCallback):
    """Custom callback for detailed loss logging during training."""
    
    def __init__(self, log_file: str = "training_loss.csv"):
        self.log_file = log_file
        # Initialize log file with headers
        with open(self.log_file, "w") as f:
            f.write("step,train_loss,eval_loss\n")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training loss."""
        if logs and "loss" in logs:
            with open(self.log_file, "a") as f:
                f.write(f"{state.global_step},{logs['loss']},\n")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log evaluation loss."""
        if metrics and "eval_loss" in metrics:
            with open(self.log_file, "a") as f:
                f.write(f"{state.global_step},,{metrics['eval_loss']}\n")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed LoRA Fine-tuning with 8-bit Quantization"
    )
    
    parser.add_argument(
        "--train_json",
        required=True,
        help="Path to training JSON file"
    )
    
    parser.add_argument(
        "--val_json",
        required=True,
        help="Path to validation JSON file"
    )
    
    parser.add_argument(
        "--model_id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier"
    )
    
    parser.add_argument(
        "--output_dir",
        default="./lora-finetuned-model",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--cache_dir",
        default="./cache",
        help="Directory for caching downloaded models"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Per-device training batch size"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    
    return parser.parse_args()


def load_dataset_from_json(json_path: str) -> Dataset:
    """Load dataset from JSON file and convert to HuggingFace Dataset format."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    # Convert to standard format
    processed_data = {
        "input": [],
        "label": [],
        "month": [],
        "folder_name": []
    }
    
    for item in raw_data:
        processed_data["input"].append(item.get("input", ""))
        processed_data["label"].append(item.get("label", ""))
        processed_data["month"].append(item.get("month", ""))
        processed_data["folder_name"].append(item.get("folder_name", ""))
    
    return Dataset.from_dict(processed_data)


def preprocess_data(examples: Dict, tokenizer, max_length: int = 1024) -> Dict:
    """Preprocess training data for instruction following format."""
    input_ids = []
    attention_masks = []
    labels = []
    
    for input_text, label_text in zip(examples["input"], examples["label"]):
        # Create instruction prompt
        prompt = (
            "Summarize the following text into a concise English abstract "
            "(under 300 words):\n\nText:\n" + input_text + "\n\nSummary:"
        )
        
        # Tokenize prompt and label separately
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)
        label_tokens = tokenizer(label_text, add_special_tokens=False)
        
        # Combine tokens
        combined_input_ids = prompt_tokens["input_ids"] + label_tokens["input_ids"]
        combined_attention_mask = prompt_tokens["attention_mask"] + label_tokens["attention_mask"]
        
        # Create labels (mask prompt tokens)
        combined_labels = [-100] * len(prompt_tokens["input_ids"]) + label_tokens["input_ids"]
        
        # Handle sequence length
        if len(combined_input_ids) > max_length:
            combined_input_ids = combined_input_ids[:max_length]
            combined_attention_mask = combined_attention_mask[:max_length]
            combined_labels = combined_labels[:max_length]
        else:
            # Pad sequences
            padding_length = max_length - len(combined_input_ids)
            combined_input_ids += [tokenizer.pad_token_id] * padding_length
            combined_attention_mask += [0] * padding_length
            combined_labels += [-100] * padding_length
        
        input_ids.append(combined_input_ids)
        attention_masks.append(combined_attention_mask)
        labels.append(combined_labels)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }


def setup_model_and_tokenizer(model_id: str, cache_dir: str):
    """Setup model with 8-bit quantization and LoRA configuration."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )
    
    # Setup device mapping for distributed training
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device_map = {"": local_rank}
    
    # Load base model with quantization
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(base_model, lora_config)
    
    return model, tokenizer


def create_training_arguments(output_dir: str, epochs: int, batch_size: int, learning_rate: float):
    """Create optimized training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        
        # Training parameters
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        
        # Optimization settings
        fp16=True,
        gradient_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        
        # Logging and evaluation
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=200,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Disable wandb/tensorboard for simplicity
        report_to="none",
    )


def main():
    """Main training pipeline."""
    print("🚀 Starting Distributed LoRA Fine-tuning")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup device for distributed training
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    print(f"📊 Model: {args.model_id}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🔄 Epochs: {args.epochs}")
    print(f"📱 Local Rank: {local_rank}")
    
    # Load datasets
    print("📚 Loading datasets...")
    train_dataset = load_dataset_from_json(args.train_json)
    val_dataset = load_dataset_from_json(args.val_json)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Setup model and tokenizer
    print("🔧 Setting up model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model_id, args.cache_dir)
    model.print_trainable_parameters()
    
    # Preprocess datasets
    print("🔄 Preprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_data(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_data(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Create training arguments
    training_args = create_training_arguments(
        args.output_dir, args.epochs, args.batch_size, args.learning_rate
    )
    
    # Initialize trainer
    print("🎯 Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            LossLoggingCallback()
        ],
    )
    
    # Start training
    print("🚀 Starting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.1f} seconds")
    
    # Save model
    print("💾 Saving model...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Cleanup distributed training
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    
    print("🎉 All done!")
    print(f"📁 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 