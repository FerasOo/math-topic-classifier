"""
Script for generating predictions using a trained model for the Math Topic Classification project.
"""

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
    parser = argparse.ArgumentParser(description="Generate predictions for Math Topic Classification")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test.csv",
        help="Path to the test data CSV file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.csv",
        help="Path to save predictions"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for inference"
    )
    return parser.parse_args()


def preprocess_function(examples, tokenizer, max_length=512):
    """Tokenize the examples for prediction."""
    return tokenizer(
        examples["Question"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )

def generate_predictions(model, dataset, tokenizer, batch_size):
    """Generate predictions for the test set."""
    model.eval()
    all_predictions = []
    
    # Create DataLoader-like batches
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i + batch_size]
        inputs = preprocess_function(batch, tokenizer)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
    
    return all_predictions

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging("prediction_logs")
    logger.info(f"Loading model from: {args.model_path}")
    
    try:
        # Configure quantization (same as training)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # First load the PEFT config to get the base model name
        peft_config = PeftConfig.from_pretrained(args.model_path)
        
        # Load the base model with quantization
        base_model = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=8,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': "[PAD]"})
        base_model.config.pad_token_id = tokenizer.pad_token_id

        # Load the PEFT model with adapters
        model = PeftModel.from_pretrained(base_model, args.model_path)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Load test data
        logger.info(f"Loading test data from: {args.test_path}")
        test_df = pd.read_csv(args.test_path)
        
        # Convert to Hugging Face Dataset
        test_dataset = Dataset.from_pandas(test_df)
        
        # Generate predictions
        logger.info("Generating predictions...")
        predictions = generate_predictions(model, test_dataset, tokenizer, args.batch_size)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'id': test_df['id'],
            'label': predictions
        })
        
        # Save predictions
        logger.info(f"Saving predictions to: {args.output_path}")
        submission_df.to_csv(args.output_path, index=False)
        
        
        logger.info(f"Prediction distribution:")
        value_counts = submission_df['label'].value_counts().sort_index()
        for topic_id, count in value_counts.items():
            logger.info(f"Topic {topic_id} ({TOPICS[topic_id]}): {count} predictions")
            
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise
    
    logger.info("Predictions completed successfully!")

if __name__ == "__main__":
    main() 

# zip -r ./results.zip ./results/
# unzip results.zip -d ./results/