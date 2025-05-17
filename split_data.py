"""
Script to split train.csv into train and validation sets.

This script takes a training CSV file and splits it into train and validation sets,
with options for random state and stratification on labels.
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description="Split training data into train and validation sets")
    
    parser.add_argument("--train_path", type=str, default="data/train.csv",
                      help="Path to training data CSV file")
    parser.add_argument("--random_state", type=int, default=46,
                      help="Random state for reproducibility")
    parser.add_argument("--stratify", type=bool, default=True,
                      help="Whether to stratify the split based on labels")
    parser.add_argument("--val_size", type=float, default=0.7,
                      help="Size of validation split (between 0 and 1)")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load the training data
    print(f"Loading training data from: {args.train_path}")
    df = pd.read_csv(args.train_path)
    
    # Add id column if it doesn't exist
    if 'id' not in df.columns:
        df['id'] = range(len(df))
    
    # rename Question to question
    df = df.rename(columns={'Question': 'question'})
    
    # reorder columns
    df = df[['id', 'question', 'label']]
    
    # Ensure required columns exist
    required_columns = ['question', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Split the data
    train_df, val_df = train_test_split(
        df,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=df['label'] if args.stratify else None
    )
    
    # Sort both splits by question length
    #train_df['length'] = train_df['question'].str.len()
    val_df['length'] = val_df['question'].str.len()
    
    #train_df = train_df.sort_values('length').drop('length', axis=1)
    val_df = val_df.sort_values('length').drop('length', axis=1)
    
    # Save the splits
    train_output = os.path.join(os.path.dirname(args.train_path), 'train_split.csv')
    val_output = os.path.join(os.path.dirname(args.train_path), 'val_split.csv')
    
    print(f"\nSplit summary (samples):")
    print(f"Original: {len(df):,} | Train: {len(train_df):,} | Val: {len(val_df):,}")
    
    if args.stratify:
        # Get distributions
        orig_dist = df['label'].value_counts(normalize=True)
        train_dist = train_df['label'].value_counts(normalize=True)
        val_dist = val_df['label'].value_counts(normalize=True)
        
        print("\nLabel distribution (%):")
        print("Label | Original | Train | Val")
        print("-" * 30)
        for label in sorted(orig_dist.index):
            print(f"{label:5d} | {orig_dist[label]:7.1%} | {train_dist[label]:5.1%} | {val_dist[label]:5.1%}")
    
    # Print length statistics
    print("\nQuestion length statistics:")
    print("Split | Min | Mean | Max")
    print("-" * 30)
    print(f"Train | {len(train_df['question'].iloc[0]):3d} | {train_df['question'].str.len().mean():.0f} | {train_df['question'].str.len().max():3d}")
    print(f"Val   | {len(val_df['question'].iloc[0]):3d} | {val_df['question'].str.len().mean():.0f} | {val_df['question'].str.len().max():3d}")
    
    # Save files
    train_df.to_csv(train_output, index=False)
    val_df.to_csv(val_output, index=False)
    print(f"\nFiles saved as: {os.path.basename(train_output)} and {os.path.basename(val_output)}")

if __name__ == "__main__":
    main() 