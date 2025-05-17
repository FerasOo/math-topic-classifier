"""
Script to embed math questions using stella_en_1.5B_v5 model via SentenceTransformer.
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import DATA_CONFIG, EMBEDDINGS_CONFIG

def embed_and_save():
    """Embed questions from both train and test sets and save the embeddings."""
    # Create embeddings directory
    os.makedirs(EMBEDDINGS_CONFIG['embeddings_dir'], exist_ok=True)
    
    # Load model
    print(f"Loading model {EMBEDDINGS_CONFIG['model_name']}...")
    model = SentenceTransformer(EMBEDDINGS_CONFIG['model_name'], trust_remote_code=True).cuda()
    
    # Process train and test sets
    for split in ['train', 'test']:
        print(f"\nProcessing {split} set...")
        
        # Load data
        data_path = DATA_CONFIG['train_path'] if split == 'train' else DATA_CONFIG['test_path']
        df = pd.read_csv(data_path)
        questions = df['Question'].tolist()
        
        # Encode questions (queries need prompt, documents don't)
        embeddings = model.encode(
            questions,
            batch_size=EMBEDDINGS_CONFIG['batch_size'],
            show_progress_bar=True,
            prompt_name=EMBEDDINGS_CONFIG['prompt_name']  # Only needed for queries/questions
        )
        
        # Save embeddings
        output_path = os.path.join(EMBEDDINGS_CONFIG['embeddings_dir'], f"{split}_embeddings.npy")
        np.save(output_path, embeddings)
        print(f"Saved {split} embeddings with shape {embeddings.shape} to {output_path}")

if __name__ == "__main__":
    embed_and_save() 