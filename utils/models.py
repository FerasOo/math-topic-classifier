"""
Model loading and inference utilities for the Math Topic Classification project.
"""

import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModelForSequenceClassification
from transformers import BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_name, device="cuda", classifier=True, quantization_config=None):
    """
    Load a model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): Name of the model to load
        device (str): Device to use
        classifier (bool): Whether to load a classifier model or a causal language model
        quantization_config (BitsAndBytesConfig): Quantization configuration
        
    Returns:
        tuple: (model, tokenizer)
    """
    
    logger.info(f"Loading model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        # Add default PAD token if it doesn't exist
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = "[PAD]"
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        if not classifier:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device if device != "cpu" else None,
                quantization_config=quantization_config,
                torch_dtype="auto",
                trust_remote_code=True,
            )
        else:
            # For classifier models
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=8,  # 8 math topics
                quantization_config=quantization_config,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            ).to(device)
        
        model.config.pad_token_id = tokenizer.pad_token_id

        logger.info(f"Model loaded successfully: {model_name}")
        return model, tokenizer
    
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        raise

def predict_with_classifier(model, tokenizer, questions, device="cuda"):
    """
    Make predictions using a classifier model.
    
    Args:
        model: Hugging Face classifier model
        tokenizer: Hugging Face tokenizer
        questions (list): List of questions
        device (str): Device to use
        
    Returns:
        list: Predicted labels (0-7)
    """
    model.to(device)
    model.eval()
    
    predictions = []
    
    for question in questions:
        inputs = tokenizer(
            question, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
        
        predictions.append(predicted_label)
    
    return predictions 