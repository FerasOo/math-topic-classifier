import os
import argparse
import pandas as pd
import random
import re
import torch
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import Dataset


# Set random seed for reproducibility
# random.seed(42)
# np.random.seed(42)
# torch.manual_seed(42)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(42)


# Topic mapping for reference
TOPIC_NAMES = {
    0: "Algebra",
    1: "Geometry and Trigonometry",
    2: "Calculus and Analysis",
    3: "Probability and Statistics",
    4: "Number Theory",
    5: "Combinatorics and Discrete Math",
    6: "Linear Algebra",
    7: "Abstract Algebra and Topology"
}

# Default augmentation fractions per topic (label: (number_frac, translation_frac))
DEFAULT_AUGMENTATION_FRACTIONS = {
    0: (0.1, 0.0),  # Algebra
    1: (0.1, 0.1),  # Geometry and Trigonometry
    2: (0.5, 0.5),  # Calculus and Analysis
    3: (1.0, 1.0),  # Probability and Statistics
    4: (0.2, 0.3),  # Number Theory
    5: (0.1, 0.1),  # Combinatorics and Discrete Math
    6: (3.0, 3.0),  # Linear Algebra
    7: (3.0, 3.0)   # Abstract Algebra and Topology
}

DEFAULT_AUGMENTATION_FRACTIONS = {
    0: (0.2, 0.0),  # Algebra
    1: (0.1, 0),  # Geometry and Trigonometry
    2: (0.2, 0),  # Calculus and Analysis
    3: (3.0, 0),  # Probability and Statistics
    4: (0.3, 0),  # Number Theory
    5: (0.3, 0),  # Combinatorics and Discrete Math
    6: (3.0, 0),  # Linear Algebra
    7: (3.0, 0)   # Abstract Algebra and Topology
}


def parse_args():
    parser = argparse.ArgumentParser(description="Data augmentation for math questions")
    parser.add_argument("--input_file", type=str, default="data/train_split.csv",
                        help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, default="data/augmented_train.csv",
                        help="Path to save the augmented CSV file")
    parser.add_argument("--translation_model", type=str, default="Helsinki-NLP/opus-mt-en-de",
                        help="Model for translating to another language")
    parser.add_argument("--back_translation_model", type=str, default="Helsinki-NLP/opus-mt-de-en",
                        help="Model for translating back to English")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for model inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run models on (cuda or cpu)")
    return parser.parse_args()


def load_translation_models(translation_model, back_translation_model, device):
    """Load models for back-translation."""
    print(f"Loading translation models: {translation_model} and {back_translation_model}")
    # Forward translation model (e.g., en -> de)
    f_tok = AutoTokenizer.from_pretrained(translation_model)
    f_mod = AutoModelForSeq2SeqLM.from_pretrained(translation_model).to(device)
    # Backward translation model (e.g., de -> en)
    b_tok = AutoTokenizer.from_pretrained(back_translation_model)
    b_mod = AutoModelForSeq2SeqLM.from_pretrained(back_translation_model).to(device)
    return (f_mod, f_tok), (b_mod, b_tok)


def back_translate(questions, forward_pair, backward_pair, device, batch_size=8):
    """Translate questions to another language and back to English using direct generate calls."""
    print("Performing back-translation with truncation...")
    forward_model, forward_tokenizer = forward_pair
    backward_model, backward_tokenizer = backward_pair
    translated_questions = []
    valid_indices = []  # Track which questions were successfully translated

    # Determine safe truncation length (e.g., 90% of model max)
    max_input_len = int(0.9 * forward_tokenizer.model_max_length)
    max_output_len = forward_tokenizer.model_max_length

    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i:i+batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(questions))))
        
        # Tokenize and truncate inputs
        inputs = forward_tokenizer(batch,
                                   return_tensors='pt',
                                   padding=True,
                                   truncation=True,
                                   max_length=max_input_len)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate forward translations
        with torch.no_grad():
            f_outputs = forward_model.generate(**inputs,
                                               temperature=0.7,
                                               top_p=0.9,
                                               top_k=50,
                                               do_sample=True,
                                               max_length=max_output_len)
        intermediate_texts = forward_tokenizer.batch_decode(f_outputs,
                                                             skip_special_tokens=True)

        # Backward tokenization/truncation
        b_inputs = backward_tokenizer(intermediate_texts,
                                      return_tensors='pt',
                                      padding=True,
                                      truncation=True,
                                      max_length=max_input_len)
        b_inputs = {k: v.to(device) for k, v in b_inputs.items()}
        
        # Generate back translations
        with torch.no_grad():
            b_outputs = backward_model.generate(**b_inputs,
                                                temperature=0.7,
                                                top_p=0.9,
                                                do_sample=True,
                                                top_k=50,
                                                max_length=max_output_len)
        back_texts = backward_tokenizer.batch_decode(b_outputs,
                                                     skip_special_tokens=True)

        # Filter valid translations and keep track of their indices
        for idx, text in zip(batch_indices, back_texts):
            if len(set(text)) >= 5:  # Only keep questions with at least 5 distinct characters
                translated_questions.append(text)
                valid_indices.append(idx)

    return translated_questions, valid_indices


def change_numbers_in_questions(questions):
    """Replace numbers in math questions with different values."""
    print("Changing numbers in questions...")
    modified_questions = []
    valid_indices = []  # Track which questions were successfully modified
    
    for idx, question in enumerate(tqdm(questions)):
        numbers = re.findall(r'(?<![a-zA-Z])-?\d+(?:\.\d+)?', question)

        # skip questions with less than 3 numbers
        if len(numbers) < 3:
            continue
        modified = question
        for number in numbers:
            if re.search(r'^\d+\.|^\(\d+\)', number):
                continue
            try:
                num_val = float(number)
                if abs(num_val) < 10:
                    new_val = random.choice([2,3,4,5,7,11,13,17]) if num_val != 0 else random.choice([1,2,3])
                elif abs(num_val) < 100:
                    new_val = int(num_val * random.choice([2,3,5,10]))
                else:
                    new_val = int(num_val * random.uniform(0.5,2))
                if num_val < 0:
                    new_val = -abs(new_val)
                if number.isdigit() or (number.startswith('-') and number[1:].isdigit()):
                    new_val = int(new_val)
                modified = re.sub(r'(?<![a-zA-Z])' + re.escape(number) + r'(?!\d)', str(new_val), modified, 1)
            except ValueError:
                continue
        modified_questions.append(modified)
        valid_indices.append(idx)
    return modified_questions, valid_indices


def print_distribution_stats(df, stage=""):
    dist = df['label'].value_counts(normalize=True).sort_index()
    counts = df['label'].value_counts().sort_index()
    
    if stage:
        print(f"\n{stage} Distribution:")
    print("Label | Topic                           | Count | Percentage")
    print("-" * 60)
    for label in sorted(dist.index):
        topic_name = TOPIC_NAMES[label]
        print(f"{label:5d} | {topic_name:30s} | {counts[label]:5d} | {dist[label]:9.1%}")

def main():
    args = parse_args()
    print(f"Loading data from {args.input_file}")
    df = pd.read_csv(args.input_file)
    orig_len = len(df)
    print(f"Original dataset size: {orig_len} questions")
    
    # Print original distribution
    print_distribution_stats(df, "Original")
    
    # Initialize augmented data list with original data
    augmented = [df]
    augmented_counts = {label: {'num': 0, 'trans': 0} for label in TOPIC_NAMES.keys()}
    
    # Load translation models if needed
    translation_needed = any(frac[1] > 0 for frac in DEFAULT_AUGMENTATION_FRACTIONS.values())
    if translation_needed:
        forward_pair, backward_pair = load_translation_models(
            args.translation_model,
            args.back_translation_model,
            args.device
        )
    
    # Process each topic separately
    for label, (num_frac, trans_frac) in DEFAULT_AUGMENTATION_FRACTIONS.items():
        topic_data = df[df['label'] == label]
        
        # Number substitution augmentation
        if num_frac > 0:
            num_count = int(len(topic_data) * num_frac)
            if num_count > 0:
                # Use replacement if fraction > 1
                replace = num_frac > 1
                samples = topic_data.sample(n=num_count, replace=replace)
                modified, valid_indices = change_numbers_in_questions(samples['question'].tolist())
                
                # Create new dataframe only for valid augmented questions
                if modified:  # Check if we have any valid modifications
                    ndf = samples.iloc[valid_indices].copy()
                    ndf['question'] = modified
                    ndf['id'] = ndf['id'].astype(str) + f'_number'
                    augmented.append(ndf)
                    augmented_counts[label]['num'] = len(ndf)
        
        # Back-translation augmentation
        if trans_frac > 0:
            trans_count = int(len(topic_data) * trans_frac)
            if trans_count > 0:
                # Use replacement if fraction > 1
                replace = trans_frac > 1
                samples = topic_data.sample(n=trans_count, replace=replace)
                translated, valid_indices = back_translate(
                    samples['question'].tolist(),
                    forward_pair,
                    backward_pair,
                    args.device,
                    args.batch_size
                )
                
                # Create new dataframe only for valid translations
                if translated:  # Check if we have any valid translations
                    tdf = samples.iloc[valid_indices].copy()
                    tdf['question'] = translated
                    tdf['id'] = tdf['id'].astype(str) + f'_translate'
                    augmented.append(tdf)
                    augmented_counts[label]['trans'] = len(tdf)
    
    # Print augmentation statistics
    print("\nAugmentation Statistics:")
    print("Label | Topic                           | Number Aug | Translation Aug")
    print("-" * 75)
    for label in sorted(TOPIC_NAMES.keys()):
        topic_name = TOPIC_NAMES[label]
        num_count = augmented_counts[label]['num']
        trans_count = augmented_counts[label]['trans']
        print(f"{label:5d} | {topic_name:30s} | {num_count:10d} | {trans_count:15d}")
    
    # Combine all augmented data
    combined = pd.concat(augmented, ignore_index=True)
    
    # Print final distribution before sorting
    print_distribution_stats(combined, "Final")
    
    # Sort by question length
    #combined['length'] = combined['question'].str.len()
    #combined = combined.sort_values('length').drop('length', axis=1)
    
    # Save augmented dataset
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    combined.to_csv(args.output_file, index=False)
    print(f"\nSaved augmented dataset to {args.output_file}")
    print(f"Total questions: {len(combined):,} (Original: {orig_len:,}, Added: {len(combined) - orig_len:,})")
    print(f"Question length range: {len(combined['question'].iloc[0])} to {len(combined['question'].iloc[-1])} characters")

if __name__ == "__main__":
    main()

# python train.py --use_dagshub