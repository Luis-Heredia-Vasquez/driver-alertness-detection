#!/usr/bin/env python
"""
Evaluation script for driver alertness detection model.

Loads a trained checkpoint, evaluates on test set, computes metrics,
and generates plots (confusion matrix, ROC curve).

Usage:
    python scripts/evaluate.py --checkpoint outputs/models/best_model.pt
    python scripts/evaluate.py --checkpoint outputs/models/best_model.pt --output-dir outputs/plots/
"""
import sys
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_model(checkpoint_path, num_classes=2, device='cpu'):
    """Load model from checkpoint."""
    model = SimpleCNN(num_classes=num_classes)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Try to load as state_dict or full checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    return model


def create_test_dataset(num_samples=500):
    """Create synthetic test dataset."""
    X_test = torch.randn(num_samples, 3, 64, 64)
    y_test = torch.randint(0, 2, (num_samples,))
    
    test_dataset = TensorDataset(X_test, y_test)
    logger.info(f"Created test dataset with {num_samples} samples")
    return test_dataset


def evaluate_model(model, dataloader, device, num_classes=2):
    """
    Evaluate model and collect predictions and labels.
    
    Returns:
        predictions: np.array of predicted labels
        probabilities: np.array of prediction probabilities
        labels: np.array of true labels
    """
    model.eval()
    predictions = []
    probabilities_list = []
    labels = []
    
    with torch.no_grad():
        for images, batch_labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())
            probabilities_list.extend(probs.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    predictions = np.array(predictions)
    probabilities = np.array(probabilities_list)
    labels = np.array(labels)
    
    return predictions, probabilities, labels


def compute_metrics(y_true, y_pred, y_proba):
    """Compute classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    logger.info(f"Metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, "
                f"Recall={recall:.4f}, F1={f1:.4f}")
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, output_dir, class_names=None):
    """Plot and save confusion matrix."""
    if class_names is None:
        class_names = ['Alert', 'Drowsy']
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    
    # Set ticks and labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix')
    
    output_path = Path(output_dir) / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {output_path}")
    plt.close()


def plot_roc_curve(y_true, y_proba, output_dir, class_names=None):
    """Plot and save ROC curve."""
    if class_names is None:
        class_names = ['Alert', 'Drowsy']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get probabilities for positive class (Drowsy = 1)
    if y_proba.shape[1] > 1:
        y_score = y_proba[:, 1]
    else:
        y_score = y_proba.flatten()
    
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {class_names[1]} Detection')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    output_path = Path(output_dir) / 'roc_curve.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved ROC curve to {output_path}")
    plt.close()


def plot_metrics_summary(metrics, output_dir):
    """Plot metrics summary bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    ax.set_ylim([0, 1.0])
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics')
    ax.grid(axis='y', alpha=0.3)
    
    output_path = Path(output_dir) / 'metrics_summary.png'
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    logger.info(f"Saved metrics summary to {output_path}")
    plt.close()


@click.command()
@click.option('--checkpoint', required=True, help='Path to model checkpoint')
@click.option('--config', default='configs/default.yaml', help='Config YAML path')
@click.option('--output-dir', default='outputs/plots/', help='Output directory for plots')
@click.option('--device', default='auto', help='Device: auto, cpu, or cuda')
@click.option('--num-samples', default=500, help='Number of test samples')
def main(checkpoint, config, output_dir, device, num_samples):
    """Evaluate trained model on test set."""
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cfg = load_config(config)
    num_classes = cfg['default']['model']['num_classes']
    
    # Select device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint, num_classes=num_classes, device=device)
    
    # Create test dataset
    test_dataset = create_test_dataset(num_samples=num_samples)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Evaluate
    logger.info("Running evaluation...")
    predictions, probabilities, labels = evaluate_model(model, test_loader, device, num_classes)
    
    # Compute metrics
    metrics = compute_metrics(labels, predictions, probabilities)
    
    # Generate classification report
    class_names = ['Alert', 'Drowsy']
    report = classification_report(labels, predictions, target_names=class_names, digits=4)
    logger.info(f"\nClassification Report:\n{report}")
    
    # Generate plots
    logger.info("Generating plots...")
    plot_confusion_matrix(labels, predictions, output_dir, class_names)
    plot_roc_curve(labels, probabilities, output_dir, class_names)
    plot_metrics_summary(metrics, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Checkpoint: {checkpoint}")
    print(f"Test samples: {num_samples}")
    print(f"Device: {device}")
    print(f"\nMetrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"\nPlots saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()

