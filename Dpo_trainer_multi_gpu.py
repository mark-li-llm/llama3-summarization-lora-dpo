#!/usr/bin/env python3
"""
Multi-GPU DPO Fine-tuning for Large Language Models
===================================================

Optimized DPO training implementation with model parallelism for large language models.
Features balanced layer distribution across multiple GPUs and memory-efficient training.

Key Features:
    • Model parallelism with balanced device mapping
    • 8-bit optimizer for memory efficiency
    • Flash Attention integration
    • Reference-free DPO training

Example Usage:
    python multi_gpu_dpo_trainer.py \
        --train_json ./data/preferences.json \
        --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --output_dir ./models/dpo-trained \
        --epochs 5 --batch_size 2

Dependencies:
    pip install torch transformers trl datasets
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig


def parse_arguments():
    """Parse command line arguments for DPO training."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU DPO Fine-tuning for Large Language Models"
    )
    
    parser.add_argument(
        "--train_json", 
        required=True,
        help="Path to training JSON file with prompt/chosen/rejected fields"
    )
    
    parser.add_argument(
        "--model_id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier"
    )
    
    parser.add_argument(
        "--cache_dir",
        default="./cache",
        help="Directory for caching downloaded models"
    )
    
    parser.add_argument(
        "--output_dir", 
        default="./dpo-trained-model",
        help="Output directory for trained model"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3,
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
        default=5e-6,
        help="Learning rate for training"
    )
    
    parser.add_argument(
        "--beta", 
        type=float, 
        default=0.1,
        help="DPO regularization coefficient (higher = more conservative)"
    )
    
    return parser.parse_args()


def load_and_split_dataset(train_json_path):
    """Load dataset and split into train/eval sets."""
    dataset = load_dataset("json", data_files=train_json_path, split="train")
    dataset_split = dataset.train_test_split(test_size=0.05, seed=42)
    return dataset_split["train"], dataset_split["test"]


def setup_model_and_tokenizer(model_id, cache_dir):
    """Initialize model with multi-GPU configuration and tokenizer."""
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        cache_dir=cache_dir, 
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configure multi-GPU memory allocation
    device_count = torch.cuda.device_count()
    if device_count >= 2:
        max_memory = {i: "40GiB" for i in range(device_count)}
        device_map = "balanced"
        print(f"Multi-GPU setup: {device_count} GPUs with balanced distribution")
    else:
        max_memory = None
        device_map = "auto"
        print("Single-GPU setup")
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map=device_map,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    
    # Enable Flash Attention for performance
    torch.backends.cuda.enable_flash_sdp(True)
    
    return model, tokenizer


def create_training_config(args):
    """Create DPO training configuration."""
    return DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        
        # Logging and evaluation
        logging_steps=20,
        eval_steps=200,
        save_steps=500,
        
        # DPO specific parameters
        beta=args.beta,
        reference_free=True,  # Memory optimization
        
        # Optimization settings
        optim="adamw_bnb_8bit",  # 8-bit optimizer for memory efficiency
        gradient_checkpointing=True,
        fp16=True,
        
        # Monitoring and saving
        report_to=["tensorboard"],
        push_to_hub=False,
        save_total_limit=3,
    )


def main():
    """Main training pipeline."""
    print("🚀 Starting Multi-GPU DPO Training")
    print("=" * 50)
    
    # Parse arguments
    args = parse_arguments()
    
    print(f"📊 Target model: {args.model_id}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Training epochs: {args.epochs}")
    
    # Load and prepare dataset
    print("📚 Loading dataset...")
    train_dataset, eval_dataset = load_and_split_dataset(args.train_json)
    print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval samples")
    
    # Setup model and tokenizer
    print("🔧 Initializing model and tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(args.model_id, args.cache_dir)
    print(f"Model loaded: {model.num_parameters():,} parameters")
    
    # Create training configuration
    print("⚙️ Creating training configuration...")
    training_args = create_training_config(args)
    
    # Initialize DPO trainer
    print("🎯 Initializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Start training
    print("🚀 Starting training...")
    trainer.train()
    
    # Save trained model
    print("💾 Saving trained model...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print("✅ Training completed successfully!")
    print(f"📁 Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main() 