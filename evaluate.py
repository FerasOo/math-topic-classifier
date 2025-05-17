"""
Script for evaluating a specific checkpoint on the validation set for the Math Topic Classification project.
"""

import os
import argparse
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from datasets import Dataset
from tqdm import tqdm
from peft import PeftModel, PeftConfig

# Import from our modules
from utils.logging_utils import setup_logging
from utils.evaluation_utils import (evaluate_predictions, 
                                save_evaluation_summary, 
                                plot_confusion_matrix)

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
    parser = argparse.ArgumentParser(description="Evaluate a specific checkpoint for Math Topic Classification")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory (e.g., results/20250501_220500_Qwen2.5-Math-7B/checkpoints/checkpoint-600)"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/val_split.csv",
        help="Path to the validation split CSV file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for inference"
    )
    return parser.parse_args()

def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the examples for evaluation."""
    return tokenizer(
        examples["question"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def generate_predictions(model, dataset, tokenizer, batch_size):
    """Generate predictions and probabilities for the validation set."""
    model.eval()
    all_predictions = []
    all_probabilities = []
    
    # Create DataLoader-like batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        inputs = preprocess_function(batch, tokenizer)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return all_predictions, all_probabilities

def create_eval_dir(checkpoint_path):
    """Create evaluation directory based on checkpoint path."""
    # Extract checkpoint number
    checkpoint_num = checkpoint_path.split('-')[-1]
    
    # Create eval directory path
    base_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    eval_dir = os.path.join(base_dir, f"eval-{checkpoint_num}")
    os.makedirs(eval_dir, exist_ok=True)
    
    return eval_dir

def main():
    # Parse arguments
    args = parse_args()
    
    # Create evaluation directory
    eval_dir = create_eval_dir(args.checkpoint_path)
    
    # Set up logging
    logger = setup_logging(eval_dir)
    logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
    logger.info(f"Results will be saved to: {eval_dir}")
    
    try:
        # Configure quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load PEFT config to get base model name
        peft_config = PeftConfig.from_pretrained(args.checkpoint_path)
        
        # Load base model with quantization
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=8,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': "[PAD]"})
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # Load PEFT model with adapters
        model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Load validation data
        logger.info(f"Loading validation data from: {args.val_path}")
        val_df = pd.read_csv(args.val_path)
        # from src.data_loader import load_data, split_train_val
        # train_df = load_data("data/train.csv")
        # train_df.rename(columns={"Question": "question"}, inplace=True)
        # train_df['id'] = list(range(len(train_df)))
        # _, val_df = split_train_val(train_df, val_size=0.1)
        
        # Convert to Hugging Face Dataset
        val_dataset = Dataset.from_pandas(val_df)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions, probabilities = generate_predictions(model, val_dataset, tokenizer, args.batch_size)
        
        # Create detailed predictions DataFrame
        results_df = pd.DataFrame({
            'id': val_df['id'],
            'question': val_df['question'],
            'true_label': val_df['label'],
            'predicted_label': predictions
        })
        
        # Add probability columns for each class
        for i in range(len(TOPICS)):
            results_df[f'prob_class_{i}'] = [prob[i] for prob in probabilities]
        
        # Save detailed predictions
        results_path = os.path.join(eval_dir, 'validation_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Detailed results saved to: {results_path}")
        
        # Evaluate predictions
        metrics = evaluate_predictions(
            results_df['true_label'].values,
            results_df['predicted_label'].values,
            TOPICS
        )
        
        # Save evaluation metrics
        summary_path, json_path = save_evaluation_summary(
            metrics,
            eval_dir,
            peft_config.base_model_name_or_path
        )
        logger.info(f"Evaluation metrics saved to: {summary_path}")
        
        # Plot confusion matrix
        class_names = [TOPICS[i] for i in range(len(TOPICS))]
        plot_confusion_matrix(
            np.array(metrics['confusion_matrix']),
            class_names,
            save_path=os.path.join(eval_dir, 'confusion_matrix.png')
        )
        
        # Print key metrics
        logger.info(f"\nValidation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"F1-micro: {metrics['f1_micro']:.4f}")
        logger.info(f"F1-macro: {metrics['f1_macro']:.4f}")
        
        # Print prediction distribution
        logger.info(f"\nPrediction distribution:")
        pred_counts = pd.Series(predictions).value_counts().sort_index()
        for topic_id, count in pred_counts.items():
            logger.info(f"Topic {topic_id} ({TOPICS[topic_id]}): {count} predictions")
            
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main() 