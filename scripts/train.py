#!/usr/bin/env python
"""
Training script with Click CLI for the driver alertness detection model.

Loads configuration, prepares dataset, trains SimpleCNN model, and saves
checkpoints to outputs/models/ directory.

Usage:
    python scripts/train.py --config configs/default.yaml --data-dir data/ --output-dir outputs/
    python scripts/train.py --resume outputs/models/checkpoint_epoch_10.pt
"""
import os
import sys
from pathlib import Path
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN
from src.utils.config import load_config
from src.utils.logger import get_logger


logger = get_logger(__name__)


class TrainingContext:
    """Context manager for training state."""
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.start_epoch = 0
        self.best_loss = float('inf')
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def build_model(self):
        """Instantiate model."""
        cfg = self.config['default']
        self.model = SimpleCNN(num_classes=cfg['model']['num_classes'])
        self.model = self.model.to(self.device)
        logger.info(f"Model created: {cfg['model']['name']}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")
    
    def build_optimizer(self):
        """Instantiate optimizer."""
        cfg = self.config['default']['training']
        opt_name = cfg.get('optimizer', 'adam').lower()
        
        if opt_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=cfg['learning_rate'],
                weight_decay=cfg.get('weight_decay', 0.0)
            )
        elif opt_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=cfg['learning_rate'],
                momentum=cfg.get('momentum', 0.9),
                weight_decay=cfg.get('weight_decay', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
        
        logger.info(f"Optimizer: {opt_name} (lr={cfg['learning_rate']})")
    
    def build_scheduler(self):
        """Instantiate learning rate scheduler."""
        cfg = self.config['default']['training']
        scheduler_name = cfg.get('scheduler', 'cosine').lower()
        
        if scheduler_name == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg['epochs'],
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.get('scheduler_step', 10),
                gamma=0.1
            )
        elif scheduler_name == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            logger.warning(f"Unknown scheduler: {scheduler_name}, skipping")
            self.scheduler = None
    
    def build_criterion(self):
        """Instantiate loss function."""
        cfg = self.config['default']['training']
        criterion_name = cfg.get('criterion', 'cross_entropy').lower()
        
        if criterion_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")
        
        logger.info(f"Loss function: {criterion_name}")


def load_or_create_dataset(data_dir, input_size=(64, 64), num_samples=1000):
    """
    Load or create synthetic dataset for testing.
    
    In production, this should load real data from data_dir.
    """
    logger.info(f"Loading dataset from {data_dir}...")
    
    # Create synthetic dataset for testing
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.15)
    
    X_train = torch.randn(num_train, 3, input_size[0], input_size[1])
    y_train = torch.randint(0, 2, (num_train,))
    
    X_val = torch.randn(num_val, 3, input_size[0], input_size[1])
    y_val = torch.randint(0, 2, (num_val,))
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    logger.info(f"Dataset created: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    return train_dataset, val_dataset


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx+1}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f} Acc: {100*correct/total:.2f}%"
            )
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    avg_acc = 100 * correct / total
    return avg_loss, avg_acc


def save_checkpoint(model, optimizer, epoch, loss, output_dir, is_best=False):
    """Save model checkpoint."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    best_path = output_dir / "best_model.pt"
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat()
    }, checkpoint_path)
    
    if is_best:
        torch.save(model.state_dict(), best_path)
        logger.info(f"Saved best model to {best_path}")
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer):
    """Resume training from checkpoint."""
    if not Path(checkpoint_path).exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return 0
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    logger.info(f"Resumed from checkpoint (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")
    return start_epoch


@click.command()
@click.option('--config', default='configs/default.yaml', help='Config YAML path')
@click.option('--data-dir', default='data/', help='Data directory')
@click.option('--output-dir', default='outputs/', help='Output directory for models')
@click.option('--resume', default=None, help='Resume from checkpoint')
@click.option('--device', default='auto', help='Device: auto, cpu, or cuda')
def main(config, data_dir, output_dir, resume, device):
    """Train SimpleCNN for driver alertness detection."""
    
    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    
    # Load configuration
    cfg = load_config(config)
    logger.info(f"Loaded config from {config}")
    
    # Select device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create training context
    ctx = TrainingContext(cfg, device=device)
    ctx.build_model()
    ctx.build_optimizer()
    ctx.build_criterion()
    ctx.build_scheduler()
    
    # Load or create dataset
    train_dataset, val_dataset = load_or_create_dataset(
        data_dir,
        input_size=tuple(cfg['default']['model']['input_size'][1:]),
        num_samples=1000
    )
    
    batch_size = cfg['default']['training']['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Resume from checkpoint if provided
    if resume:
        ctx.start_epoch = load_checkpoint(resume, ctx.model, ctx.optimizer)
    
    # Training loop
    epochs = cfg['default']['training']['epochs']
    models_dir = Path(output_dir) / 'models'
    
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(ctx.start_epoch, epochs):
        # Train
        train_loss, train_acc = train_epoch(
            ctx.model, train_loader, ctx.optimizer,
            ctx.criterion, device, epoch + 1, epochs
        )
        ctx.history['train_loss'].append(train_loss)
        ctx.history['train_acc'].append(train_acc)
        
        # Validate
        val_loss, val_acc = evaluate(ctx.model, val_loader, ctx.criterion, device)
        ctx.history['val_loss'].append(val_loss)
        ctx.history['val_acc'].append(val_acc)
        
        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%"
        )
        
        # Save checkpoint
        if (epoch + 1) % cfg['default']['logging']['save_interval'] == 0:
            is_best = val_loss < ctx.best_loss
            if is_best:
                ctx.best_loss = val_loss
            save_checkpoint(ctx.model, ctx.optimizer, epoch, val_loss, models_dir, is_best=is_best)
        
        # Learning rate scheduler
        if ctx.scheduler:
            ctx.scheduler.step()
    
    # Final save
    final_path = save_checkpoint(ctx.model, ctx.optimizer, epochs - 1, ctx.history['val_loss'][-1], models_dir)
    
    logger.info(f"Training completed. Best model saved.")
    logger.info(f"Models saved to: {models_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Config: {config}")
    print(f"Epochs: {epochs}")
    print(f"Final Train Loss: {ctx.history['train_loss'][-1]:.4f}")
    print(f"Final Train Acc: {ctx.history['train_acc'][-1]:.2f}%")
    print(f"Final Val Loss: {ctx.history['val_loss'][-1]:.4f}")
    print(f"Final Val Acc: {ctx.history['val_acc'][-1]:.2f}%")
    print(f"Best Val Loss: {ctx.best_loss:.4f}")
    print(f"Output: {final_path}")
    print("="*60)


if __name__ == '__main__':
    main()

