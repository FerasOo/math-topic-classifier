"""
Evaluation utilities for the Math Topic Classification project.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_predictions(y_true, y_pred, label_names=None):
    """
    Evaluate predictions and return metrics.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        label_names (dict, optional): Mapping from label indices to label names
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Get classification report
    if label_names:
        target_names = [label_names[i] for i in range(len(label_names))]
        report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    else:
        report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Return all metrics
    return {
        'accuracy': accuracy,
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }

def save_evaluation_summary(metrics, save_dir, model_name):
    """
    Save evaluation metrics to a text file and JSON file.
    
    Args:
        metrics (dict): Evaluation metrics
        save_dir (str): Directory to save the summary
        model_name (str): Name of the model
    """
    # Extract metrics
    accuracy = metrics['accuracy']
    f1_micro = metrics['f1_micro']
    f1_macro = metrics['f1_macro']
    f1_weighted = metrics['f1_weighted']
    report = metrics['classification_report']
    
    # Create summary text
    summary = f"Model: {model_name}\n"
    summary += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    summary += f"Accuracy: {accuracy:.4f}\n"
    summary += f"F1-micro (competition metric): {f1_micro:.4f}\n"
    summary += f"F1-macro: {f1_macro:.4f}\n"
    summary += f"F1-weighted: {f1_weighted:.4f}\n\n"
    summary += "Classification Report:\n"
    
    # Format classification report for text
    for cls in report:
        if cls not in ['accuracy', 'macro avg', 'weighted avg']:
            cls_metrics = report[cls]
            summary += f"  Class {cls}: Precision: {cls_metrics['precision']:.4f}, "
            summary += f"Recall: {cls_metrics['recall']:.4f}, F1: {cls_metrics['f1-score']:.4f}, "
            summary += f"Support: {cls_metrics['support']}\n"
    
    summary += f"\n  Macro avg: Precision: {report['macro avg']['precision']:.4f}, "
    summary += f"Recall: {report['macro avg']['recall']:.4f}, F1: {report['macro avg']['f1-score']:.4f}\n"
    summary += f"  Weighted avg: Precision: {report['weighted avg']['precision']:.4f}, "
    summary += f"Recall: {report['weighted avg']['recall']:.4f}, F1: {report['weighted avg']['f1-score']:.4f}\n"
    
    # Save summary to text file
    txt_path = os.path.join(save_dir, 'evaluation_summary.txt')
    with open(txt_path, 'w') as f:
        f.write(summary)
    
    # Save metrics to JSON file
    json_path = os.path.join(save_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return txt_path, json_path

def plot_confusion_matrix(cm, classes, save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        cm (ndarray): Confusion matrix
        classes (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 