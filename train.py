"""
Training script for the Math Topic Classification project.

This script fine-tunes a pre-trained model from Hugging Face on the
math topic classification task using QLoRA (Quantized Low-Rank Adaptation).
"""

import os
import argparse
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import mlflow
from transformers import TrainingArguments, Trainer, BitsAndBytesConfig, set_seed
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.integrations import MLflowCallback
from datasets import Dataset
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Import from our modules
from utils.models import load_model_and_tokenizer
from utils.logging_utils import setup_logging, save_run_config

# Topics/Labels mapping
TOPICS = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Math Topic Classification Training with QLoRA")
    
    # Model configuration
    # to use Phi-3-mini, follow this link: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/commit/a53528a5e4554ddc4c53d4d34db9fbcfcc1f4aea
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                      help="Model name from Hugging Face")
    
    # Data configuration
    parser.add_argument("--train_path", type=str, default="data/train_split.csv", 
                      help="Path to training split data")
    parser.add_argument("--val_path", type=str, default="data/val_split.csv", 
                      help="Path to validation split data")
    parser.add_argument("--results_dir", type=str, default="results", 
                      help="Path to results directory")
    
    # Training configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, 
                      help="Number of updates steps to accumulate before backward pass")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                      help="Number of epochs to wait before early stopping")
    parser.add_argument("--lr_scheduler_type", type=str, default="linear", 
                      help="Learning rate scheduler type")
    parser.add_argument("--label_smoothing", type=float, default=0.2, help="Label smoothing")
    
    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=8, help="Rank of LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha parameter for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout for LoRA")
    
    # MLflow configuration
    parser.add_argument("--use_mlflow", action="store_true", help="Whether to use MLflow for tracking")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="./mlruns", 
                      help="MLflow tracking URI (local path or server URL)")
    parser.add_argument("--mlflow_experiment_name", type=str, default="math-topic-classification", 
                      help="MLflow experiment name")
    
    return parser.parse_args()

def preprocess_function(examples, tokenizer):
    """Tokenize the examples for training."""
    return tokenizer(
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=512, 
    )

def compute_metrics(eval_pred):
    """Compute metrics for the Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_micro": f1_score(labels, predictions, average='micro'),
        "f1_macro": f1_score(labels, predictions, average='macro'),
        "f1_weighted": f1_score(labels, predictions, average='weighted'),
        "precision_micro": precision_score(labels, predictions, average='micro'),
        "recall_micro": recall_score(labels, predictions, average='micro'),
    }

def main():
    # Parse arguments
    set_seed(42)
    args = parse_args()
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is not available. Training will be slow!")
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = args.model.split('/')[-1]
    results_dir = os.path.join(args.results_dir, f"{timestamp}_{model_name}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(results_dir)
    
    logger.info(f"Starting fine-tuning with model: {args.model}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    # Save the configuration
    config = vars(args)
    save_run_config(config, results_dir)

    if args.use_mlflow:
        logger.info("MLflow tracking is enabled, initializing...")
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.start_run(run_name=f"{timestamp}_{model_name}")
        
        # Log parameters
        mlflow.log_params({
            "model": args.model,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        })
        logger.info(f"MLflow initialized: {mlflow.get_tracking_uri()}")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Load data
        logger.info(f"Loading training data from: {args.train_path}")
        logger.info(f"Loading validation data from: {args.val_path}")
        train_df = pd.read_csv(args.train_path)
        val_df = pd.read_csv(args.val_path)
        
        # Convert to Hugging Face Dataset format
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Configure quantization 
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load tokenizer and model
        model, tokenizer = load_model_and_tokenizer(args.model, device=device, classifier=True, quantization_config=bnb_config)
        
        # Tokenize the datasets
        logger.info("Tokenizing datasets...")
        tokenized_train = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True
        )
        tokenized_val = val_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True
        )
        
        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        lora_module_names = set()
        for full_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                attr_name = full_name.split(".")[-1]
                lora_module_names.add(attr_name)
        
        print("lora_module_names: ", lora_module_names)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            #target_modules=['qkv_proj', 'o_proj'],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS"
        )
        
        # Get PEFT model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(results_dir, "checkpoints"),
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            weight_decay=args.weight_decay,
            logging_dir=os.path.join(results_dir, "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            save_total_limit=5,
            fp16=True,  
            report_to="tensorboard",
            metric_for_best_model="f1_micro",
            greater_is_better=True,
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            lr_scheduler_type=args.lr_scheduler_type
        )
        
        callbacks = [EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        if args.use_mlflow:
            callbacks.append(MLflowCallback())
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()

        # evaluate the model
        logger.info("Evaluating the model...")
        eval_results = trainer.evaluate()
        logger.info(f"Evaluation results: {eval_results}")
        
        if args.use_mlflow:
            # Log metrics to MLflow
            mlflow.log_metrics(eval_results)

        # save the model
        logger.info("Saving the model...")
        model.save_pretrained(os.path.join(results_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(results_dir, "final_model"))
        
        if args.use_mlflow:
            # Log model to MLflow
            mlflow.log_artifact(os.path.join(results_dir, "final_model"))

        # Record end time and log duration
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Training completed in {duration:.2f} seconds")
        
        if args.use_mlflow:
            mlflow.log_metric("training_duration_seconds", duration)
            mlflow.end_run()
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        if args.use_mlflow:
            mlflow.end_run(status="FAILED")
        raise
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 