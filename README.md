# Llama3 Summarization with LoRA & DPO

A comprehensive fine-tuning pipeline for Llama-3 models focused on text summarization tasks. This repository provides multiple training approaches including LoRA (Low-Rank Adaptation), DPO (Direct Preference Optimization), and memory-efficient 8-bit quantization.

## ✨ Features

- **Multiple Training Methods**: LoRA fine-tuning, DPO training, and 8-bit quantization
- **Multi-GPU Support**: Distributed training with Accelerate framework and model parallelism
- **Memory Optimization**: 8-bit quantization and gradient checkpointing for large model training
- **Production-Ready**: Comprehensive logging, early stopping, and checkpoint management
- **Flexible Configuration**: Command-line arguments for easy experimentation

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers>=4.51.3 peft bitsandbytes accelerate datasets trl
```

### Prepare Your Data

Create training and validation JSON files with the following format:

```json
[
  {
    "input": "Your input text to be summarized...",
    "label": "Expected summary output...",
    "month": "optional_metadata",
    "folder_name": "optional_metadata"
  }
]
```

### Training Options

#### 1. LoRA Fine-tuning with Accelerate

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision fp16 \
  Llama_finetuning_accelerate_.py
```

#### 2. 8-bit Quantized Training

```bash
accelerate launch \
  --num_processes 2 \
  --mixed_precision fp16 \
  Llama_finetuning_8bit.py \
  --train_json ./data/train.json \
  --val_json ./data/val.json \
  --output_dir ./models/lora-finetuned \
  --epochs 10 \
  --batch_size 1 \
  --learning_rate 2e-5
```

#### 3. DPO Training (Direct Preference Optimization)

```bash
python Dpo_trainer_multi_gpu.py \
  --train_json ./data/preferences.json \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --output_dir ./models/dpo-trained \
  --epochs 5 \
  --batch_size 2 \
  --beta 0.1
```

## 📁 Repository Structure

```
llama3-summarization-lora-dpo/
├── Llama_finetuning_accelerate_.py   # Standard LoRA training with Accelerate
├── Llama_finetuning_8bit.py          # Memory-efficient 8-bit quantized training
├── Dpo_trainer_multi_gpu.py          # DPO training for preference optimization
└── README.md                         # This file
```

## 🔧 Configuration

### LoRA Configuration
- **Rank (r)**: 8
- **Alpha**: 32
- **Target Modules**: All attention and MLP projection layers
- **Dropout**: 0.05

### DPO Configuration
- **Beta**: 0.1 (regularization coefficient)
- **Reference-free**: Enabled for memory efficiency
- **Optimizer**: 8-bit AdamW

### Hardware Requirements

- **Minimum**: 1x GPU with 24GB VRAM (RTX 4090, A6000)
- **Recommended**: 2x GPUs with 40GB+ VRAM each (A100, H100)
- **Memory Optimization**: 8-bit quantization reduces memory usage by ~50%

## 📊 Training Features

### Advanced Optimizations
- **Flash Attention**: Enabled for improved training speed
- **Gradient Checkpointing**: Reduces memory usage during backpropagation
- **Mixed Precision (FP16)**: Faster training with lower memory footprint
- **Early Stopping**: Prevents overfitting with configurable patience

### Monitoring & Logging
- **Loss Tracking**: Detailed CSV logs for training and validation loss
- **TensorBoard**: Integration for DPO training visualization
- **Checkpoint Management**: Automatic saving with configurable intervals

## 🎯 Use Cases

1. **Document Summarization**: Academic papers, reports, articles
2. **Content Condensation**: Long-form content to brief summaries
3. **Research Applications**: Custom summarization models for specific domains
4. **Preference Learning**: Fine-tuning models based on human preferences

## 🔬 Technical Details

### Model Architecture
- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 8-bit weights and optimizers
- **Attention**: Flash Attention for memory efficiency

### Training Pipeline
1. **Data Preprocessing**: Instruction formatting with prompt templates
2. **Tokenization**: Automated padding and truncation
3. **Loss Masking**: Prompt tokens masked from loss calculation
4. **Distributed Training**: Multi-GPU support via Accelerate



