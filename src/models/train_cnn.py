import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast  # For Mixed Precision Training
import timm
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Add parent directory to path to access utils and src
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from YOUR actual locations
from utils.config import config  # config is in utils folder
from src.data_loader import get_dataloaders  # data_loader is in src folder

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
import warnings
import gc  # For garbage collection
warnings.filterwarnings('ignore')

class EfficientNetTrainer:
    def __init__(self, num_classes=None):
        self.device = config.DEVICE
        self.num_classes = num_classes
        self.model = None
        self.optimizer = None
        self.scheduler_cosine = None
        self.scheduler_plateau = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.class_names = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []
        
        # Mixed Precision Training Setup
        self.use_amp = config.USE_MIXED_PRECISION
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"✓ Mixed Precision Training: Enabled")
            print(f"  Using batch sizes: CNN={config.BATCH_SIZES['cnn']}")
        else:
            self.scaler = None
            print(f"⚠ Mixed Precision Training: Disabled")
            print(f"  Using reduced batch sizes: CNN={config.BATCH_SIZES['cnn']}")
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Clear GPU cache before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    def setup_model(self):
        """Initialize EfficientNet-B3 with pretrained weights"""
        print("\n" + "="*60)
        print("INITIALIZING EFFICIENTNET-B3")
        print("="*60)
        
        # Clear cache before model creation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create model
        self.model = timm.create_model(
            'efficientnet_b3', 
            pretrained=True,
            num_classes=self.num_classes,
            drop_rate=0.3,           # Dropout for regularization
            drop_path_rate=0.2       # Stochastic depth
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model loaded: efficientnet_b3")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Device: {self.device}")
        
        if torch.cuda.is_available():
            # Print GPU memory after model loading
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU Memory after model load: {memory_allocated:.2f}/{memory_reserved:.2f} GB")
            print(f"✓ Available GPU Memory: {(config.GPU_MEMORY - memory_allocated):.2f} GB")
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LR,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Setup schedulers
        self.scheduler_cosine = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.NUM_EPOCHS,
            eta_min=1e-6
        )
        
        self.scheduler_plateau = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=5, 
            factor=0.5,
            min_lr=1e-6,
            verbose=True
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"✓ Optimizer: AdamW (lr={config.LR})")
        print(f"✓ Loss: CrossEntropyLoss with label smoothing=0.1")
        print(f"✓ Schedulers: CosineAnnealing + ReduceLROnPlateau")
        
    def get_dataloaders(self):
        """Get data loaders for CNN"""
        print("\n" + "="*60)
        print("LOADING DATA")
        print("="*60)
        
        self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dataloaders("cnn")
        
        if self.train_loader is None:
            raise Exception("Failed to load data. Please run preprocessing pipeline first.")
        
        # Update num_classes if needed
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
            # Update config
            config.update_num_classes(self.num_classes)
        
        print(f"✓ Number of classes: {self.num_classes}")
        print(f"✓ Classes: {', '.join(self.class_names[:5])}..." if len(self.class_names) > 5 else f"✓ Classes: {', '.join(self.class_names)}")
        print(f"✓ Training samples: {len(self.train_loader.dataset):,}")
        print(f"✓ Validation samples: {len(self.val_loader.dataset):,}")
        print(f"✓ Test samples: {len(self.test_loader.dataset):,}")
        print(f"✓ Batch size: {config.BATCH_SIZES['cnn']}")
        print(f"✓ Training batches: {len(self.train_loader)}")
        print(f"✓ Image size: {config.IMAGE_SIZES['cnn']}x{config.IMAGE_SIZES['cnn']}")
        
    def train_epoch(self, epoch):
        """Train for one epoch with Mixed Precision"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # MIXED PRECISION TRAINING
            if self.use_amp:
                # Forward pass with autocast (FP16)
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard FP32 training
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            avg_loss = running_loss / (batch_idx + 1)
            
            if batch_idx % 5 == 0:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{current_acc:.2f}%',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                        'GPU': f'{memory_used:.2f}GB'
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'Acc': f'{current_acc:.2f}%',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
            
            # Periodic cache clearing to prevent fragmentation
            if torch.cuda.is_available() and (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / batch_count
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_count = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]  ")
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                # Use autocast for validation too
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                current_acc = 100. * correct / total
                avg_loss = running_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })
        
        epoch_loss = running_loss / batch_count
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, resume_from=None):
        """Main training loop"""
        print("\n" + "="*60)
        print("EFFICIENTNET-B3 TRAINING PIPELINE")
        print("="*60)
        
        # Print system info
        print("\n📊 System Information:")
        print(f"   Device: {config.DEVICE}")
        if torch.cuda.is_available():
            print(f"   GPU: {config.GPU_NAME}")
            print(f"   GPU Memory: {config.GPU_MEMORY:.1f} GB")
            print(f"   Mixed Precision: {'Enabled (FP16)' if self.use_amp else 'Disabled (FP32)'}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}")
        
        # Load data
        self.get_dataloaders()
        
        # Setup model
        self.setup_model()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from and os.path.exists(resume_from):
            print(f"\n📂 Resuming from checkpoint: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.use_amp and 'scaler_state_dict' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            self.train_losses = checkpoint.get('train_losses', [])
            self.val_losses = checkpoint.get('val_losses', [])
            self.train_accs = checkpoint.get('train_accs', [])
            self.val_accs = checkpoint.get('val_accs', [])
            self.best_val_acc = checkpoint.get('best_val_acc', 0)
            self.best_epoch = checkpoint.get('best_epoch', 0)
            print(f"✓ Resumed from epoch {start_epoch}")
        
        patience_counter = 0
        
        print(f"\n🚀 Starting training for {config.NUM_EPOCHS - start_epoch} epochs")
        print("="*60)
        
        try:
            for epoch in range(start_epoch, config.NUM_EPOCHS):
                epoch_start_time = datetime.now()
                
                # Train
                train_loss, train_acc = self.train_epoch(epoch)
                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                
                # Validate
                val_loss, val_acc = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # Store learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
                
                # Update schedulers
                self.scheduler_cosine.step()
                self.scheduler_plateau.step(val_loss)
                
                # Calculate epoch time
                epoch_time = (datetime.now() - epoch_start_time).total_seconds()
                
                # Print epoch summary
                print(f"\n{'='*60}")
                print(f"📊 Epoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
                print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
                print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
                print(f"   Learning Rate: {current_lr:.2e}")
                print(f"   Epoch Time: {epoch_time:.1f}s")
                
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3
                    print(f"   GPU Memory: {memory_used:.2f}/{memory_reserved:.2f} GB")
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    patience_counter = 0
                    
                    best_model_path = config.SAVED_MODELS_DIR / "efficientnet_b3_best.pth"
                    self.save_checkpoint(str(best_model_path), epoch, is_best=True)
                    print(f"   ✅ New best model! Val Acc improved to {val_acc:.2f}%")
                else:
                    patience_counter += 1
                    print(f"   ⚠️  No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                
                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = config.SAVED_MODELS_DIR / f"efficientnet_b3_epoch_{epoch+1}.pth"
                    self.save_checkpoint(str(checkpoint_path), epoch)
                
                # Early stopping
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    break
                
                # Clear cache at end of epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print("="*60)
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            interrupt_path = config.SAVED_MODELS_DIR / f"efficientnet_b3_interrupted_epoch_{epoch+1}.pth"
            self.save_checkpoint(str(interrupt_path), epoch)
            print(f"💾 Checkpoint saved to: {interrupt_path}")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ GPU Out of Memory Error!")
                print(f"Current batch size: {config.BATCH_SIZES['cnn']}")
                print("Suggestions:")
                print("  1. Reduce batch size in config.py")
                print("  2. Enable mixed precision training (already enabled)")
                print("  3. Reduce image size")
                print("  4. Use gradient accumulation")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        print(f"🎯 Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"📊 Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"⏱️  Total Epochs Trained: {len(self.train_losses)}")
        
        # Save final model
        final_model_path = config.SAVED_MODELS_DIR / "efficientnet_b3_final.pth"
        self.save_checkpoint(str(final_model_path), epoch)
    
    def evaluate(self, checkpoint_path=None):
        """Comprehensive evaluation on test set"""
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)
        
        # Load best model if checkpoint provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded successfully")
        
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        test_loss = 0.0
        correct = 0
        total = 0
        
        print("\nRunning inference on test set...")
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc="Testing"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        test_acc = 100. * correct / total
        test_loss = test_loss / len(self.test_loader)
        
        # Calculate metrics
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        print(f"\n📊 Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   F1 Score (Weighted): {f1:.4f}")
        
        # Classification report
        print("\n📈 Per-Class Performance:")
        report = classification_report(all_targets, all_preds, 
                                      target_names=self.class_names, 
                                      output_dict=True)
        
        # Print formatted report
        print(f"\n{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<12}")
        print("-" * 78)
        for class_name in self.class_names:
            stats = report[class_name]
            print(f"{class_name:<30} {stats['precision']:<12.4f} {stats['recall']:<12.4f} "
                  f"{stats['f1-score']:<12.4f} {int(stats['support']):<12}")
        
        # Save results
        results = {
            'model': 'EfficientNet-B3',
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'f1_score_weighted': f1,
            'best_val_accuracy': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses),
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = config.LOG_DIR / f"efficientnet_results_{config.RUN_ID}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
        
        # Plot visualizations
        self.plot_confusion_matrix(all_targets, all_preds)
        self.plot_training_history()
        
        return test_acc, test_loss, f1
    
    def plot_confusion_matrix(self, targets, preds):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - EfficientNet-B3', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = config.LOG_DIR / f'confusion_matrix_efficientnet_{config.RUN_ID}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {cm_path}")
        plt.close()
    
    def plot_training_history(self):
        """Plot and save training history"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss plot
        axes[0].plot(epochs, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.val_losses, 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(epochs, self.train_accs, 'b-', label='Train Acc', linewidth=2)
        axes[1].plot(epochs, self.val_accs, 'r-', label='Val Acc', linewidth=2)
        axes[1].axhline(y=self.best_val_acc, color='g', linestyle='--', 
                       label=f'Best: {self.best_val_acc:.2f}%', alpha=0.7)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning rate plot
        if self.learning_rates:
            axes[2].plot(epochs, self.learning_rates, 'g-', linewidth=2)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_yscale('log')
        
        plt.suptitle('EfficientNet-B3 Training History', fontsize=16, y=1.05)
        plt.tight_layout()
        
        history_path = config.LOG_DIR / f'training_history_efficientnet_{config.RUN_ID}.png'
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history saved to: {history_path}")
        plt.close()
    
    def save_checkpoint(self, filepath, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_cosine_state_dict': self.scheduler_cosine.state_dict(),
            'scheduler_plateau_state_dict': self.scheduler_plateau.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'learning_rates': self.learning_rates,
            'class_names': self.class_names,
            'num_classes': self.num_classes,
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': {
                'batch_size': config.BATCH_SIZES['cnn'],
                'image_size': config.IMAGE_SIZES['cnn'],
                'learning_rate': config.LR,
                'weight_decay': config.WEIGHT_DECAY,
                'mixed_precision': self.use_amp
            }
        }
        
        # Save scaler state if using mixed precision
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        if is_best:
            print(f"   💾 Best model saved: {filepath}")
        else:
            print(f"   💾 Checkpoint saved: {filepath}")

def main():
    """Main function to run training"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train EfficientNet-B3 on Plant Disease Dataset')
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume training')
    parser.add_argument('--evaluate-only', action='store_true', 
                       help='Only run evaluation on test set')
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Checkpoint path for evaluation')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = EfficientNetTrainer()
    
    if args.evaluate_only:
        # Only evaluate
        if not args.checkpoint:
            args.checkpoint = str(config.SAVED_MODELS_DIR / "efficientnet_b3_best.pth")
        
        if not os.path.exists(args.checkpoint):
            print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
            print("Please train the model first or provide a valid checkpoint path.")
            return
        
        trainer.get_dataloaders()
        trainer.setup_model()
        trainer.evaluate(checkpoint_path=args.checkpoint)
    else:
        # Train and evaluate
        trainer.train(resume_from=args.resume)
        
        # Evaluate best model
        best_model_path = config.SAVED_MODELS_DIR / "efficientnet_b3_best.pth"
        if best_model_path.exists():
            trainer.evaluate(checkpoint_path=str(best_model_path))

if __name__ == "__main__":
    main()