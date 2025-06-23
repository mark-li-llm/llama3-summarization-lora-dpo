#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accelerate-based LoRA Fine-tuning for Llama Models

Launch example:
accelerate launch \
  --num_processes 2 \
  --mixed_precision fp16 \
  accelerate_llama_finetuning.py
"""

import os, sys, json, time, subprocess
import torch
from datasets import Dataset
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ───────────────────────────────────────── system info
print("PYTHON:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print(subprocess.check_output("which python", shell=True).decode().strip())

# ───────────────────────────────────────── helpers
def load_finetune_dataset(json_file_path: str) -> Dataset:
    """Load custom JSON data into HuggingFace Dataset format."""
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cols = {k: [] for k in ["month", "folder_name", "input", "label"]}
    for item in raw:
        for k in cols:
            cols[k].append(item.get(k, ""))

    return Dataset.from_dict(cols)


def preprocess_function(examples, tokenizer, max_length: int = 1024):
    """
    Concatenate prompt + label; mask prompt region for loss calculation.
    """
    inputs, masks, labels = [], [], []

    for inp, lbl in zip(examples["input"], examples["label"]):
        prompt = (
            "Summarize the following text into a concise English abstract "
            "(under 300 words):\n\n"
            f"Text:\n{inp}\n\nSummary:"
        )
        prompt_enc = tokenizer(prompt, add_special_tokens=False)
        label_enc  = tokenizer(lbl,    add_special_tokens=False)

        input_ids  = prompt_enc["input_ids"] + label_enc["input_ids"]
        attn_mask  = prompt_enc["attention_mask"] + label_enc["attention_mask"]
        label_mask = [-100] * len(prompt_enc["input_ids"]) + label_enc["input_ids"]

        # pad / truncate
        if len(input_ids) > max_length:
            input_ids, attn_mask, label_mask = (
                input_ids[:max_length],
                attn_mask[:max_length],
                label_mask[:max_length],
            )
        else:
            pad = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad
            attn_mask += [0] * pad
            label_mask += [-100] * pad

        inputs.append(input_ids)
        masks.append(attn_mask)
        labels.append(label_mask)

    return {
        "input_ids": inputs,
        "attention_mask": masks,
        "labels": labels,
    }


class LossSaveCallback(TrainerCallback):
    def __init__(self, outfile: str = "loss_log.txt"):
        self.outfile = outfile
        with open(self.outfile, "w") as f:
            f.write("step,train_loss,eval_loss\n")

    def on_log(self, args, state, control, logs=None, **_):
        if logs and "loss" in logs:
            with open(self.outfile, "a") as f:
                f.write(f"{state.global_step},{logs['loss']},\n")

    def on_evaluate(self, args, state, control, metrics=None, **_):
        if metrics and "eval_loss" in metrics:
            with open(self.outfile, "a") as f:
                f.write(f"{state.global_step},,{metrics['eval_loss']}\n")


# ───────────────────────────────────────── main
def main():
    # paths & hyper-parameters
    model_id   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_cache = "./cache"
    output_dir  = "./finetuned_llama_model"
    train_json  = "./data/train.json"
    val_json    = "./data/val.json"

    epochs, bs, lr, max_len = 3, 1, 2e-5, 1024

    # ── datasets
    print("▶ Loading datasets …")
    train_ds = load_finetune_dataset(train_json)
    val_ds   = load_finetune_dataset(val_json)

    # ── tokenizer & base model
    print("▶ Loading tokenizer / base model …")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=model_cache)
    tok.pad_token = tok.pad_token or tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=model_cache,
        torch_dtype=torch.float16,   # fp16; device allocated by Accelerate
    )

    # ── LoRA
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "down_proj", "up_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    # ── tokenize
    print("▶ Tokenizing …")
    train_ds = train_ds.map(
        lambda x: preprocess_function(x, tok, max_len),
        batched=True, remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda x: preprocess_function(x, tok, max_len),
        batched=True, remove_columns=val_ds.column_names,
    )

    # ── training args (Transformers ≥ 4.50: evaluation_strategy → eval_strategy)
    hf_args = TrainingArguments(
        output_dir                = output_dir,
        overwrite_output_dir      = True,
        num_train_epochs          = epochs,
        per_device_train_batch_size = bs,
        learning_rate             = lr,
        fp16                      = True,
        ddp_find_unused_parameters= False,
        gradient_accumulation_steps = 1,
        save_steps                = 100,
        logging_strategy          = "steps",
        logging_steps             = 50,
        eval_strategy             = "steps",
        eval_steps                = 50,
        load_best_model_at_end    = True,
        metric_for_best_model     = "eval_loss",
        greater_is_better         = False,
        report_to                 = "none",
    )

    trainer = Trainer(
        model           = model,
        args            = hf_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tok,
        callbacks       = [
            EarlyStoppingCallback(early_stopping_patience=2),
            LossSaveCallback("loss_log.txt"),
        ],
    )

    # ── train
    print("▶ Starting training (multi-GPU via Accelerate) …")
    t0 = time.time()
    trainer.train()
    print(f"✓ Finished in {time.time() - t0:.1f}s")

    # ── save
    print("▶ Saving model …")
    trainer.model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    torch.distributed.destroy_process_group()
    print("✓ All done.")


if __name__ == "__main__":
    main() 