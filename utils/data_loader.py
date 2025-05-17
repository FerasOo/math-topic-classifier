"""
Data loading and preprocessing utilities for the Math Topic Classification project.
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import random

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_path, sort_by_length=True, num_samples=None, random_seed=42):
    """
    Load data from CSV file.
    
    Args:
        data_path (str): Path to the CSV file
        sort_by_length (bool): Whether to sort the data by question length
        num_samples (int, optional): Number of samples to load. If None, load all.
        random_seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Loaded data
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    
    # If num_samples is specified, sample a subset
    if num_samples is not None and num_samples < len(df):
        set_seed(random_seed)
        df = df.sample(num_samples, random_state=random_seed)
    
    if sort_by_length:
        df = df.sort_values(by='Question', key=lambda x: x.str.len())
    
    return df

def split_train_val(df, val_size=0.1, random_seed=42):
    """
    Split training data into train and validation sets.
    
    Args:
        df (pd.DataFrame): Training data
        val_size (float): Validation set size (proportion)
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df)
    """
    if 'label' not in df.columns:
        raise ValueError("DataFrame must contain 'label' column for stratification")
    
    train_df, val_df = train_test_split(
        df, 
        test_size=val_size, 
        random_state=random_seed,
        stratify=df['label']
    )
    
    return train_df, val_df
