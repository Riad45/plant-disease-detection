"""
Vision Transformer (ViT-Base) Training for Plant Disease Detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import timm
from tqdm import tqdm
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.config import config
from src.data_loader import get_dataloaders
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import pandas as pd
from datetime import datetime
import json
import warnings
import gc
warnings.filterwarnings('ignore')

class ViTTrainer:
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
        
        # Mixed Precision Training
        self.use_amp = config.USE_MIXED_PRECISION
        if self.use_amp:
            self.scaler = GradScaler()
            print(f"✓ Mixed Precision Training: Enabled")
        else:
            self.scaler = None
            print(f"⚠ Mixed Precision Training: Disabled")
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
    def setup_model(self):
        """Initialize Vision Transformer Base"""
        print("\n" + "="*60)
        print("INITIALIZING VISION TRANSFORMER BASE (ViT-B/16)")
        print("="*60)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create Vision Transformer Base model
        self.model = timm.create_model(
            'vit_base_patch16_224',  # ViT-Base with 16x16 patches
            pretrained=True,
            num_classes=self.num_classes,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.1
        )
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✓ Model: ViT-Base (Patch 16, 224x224)")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Device: {self.device}")
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"✓ GPU Memory after model load: {memory_allocated:.2f}/{memory_reserved:.2f} GB")
        
        # Setup optimizer with ViT-appropriate learning rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LR,  # Use same LR as EfficientNet for fair comparison
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        print(f"✓ Optimizer: AdamW (lr={config.LR})")
        print(f"✓ Loss: CrossEntropyLoss with label smoothing=0.1")
        print(f"✓ Batch size: {config.BATCH_SIZES['vit']}")
        print(f"✓ Image size: {config.IMAGE_SIZES['vit']}x{config.IMAGE_SIZES['vit']}")
        
    def get_dataloaders(self):
        """Get data loaders for ViT"""
        print("\n" + "="*60)
        print("LOADING DATA FOR ViT")
        print("="*60)
        
        self.train_loader, self.val_loader, self.test_loader, self.class_names = get_dataloaders("vit")
        
        if self.train_loader is None:
            raise Exception("Failed to load data. Please run preprocessing pipeline first.")
        
        # Update num_classes if needed
        if self.num_classes is None:
            self.num_classes = len(self.class_names)
            config.update_num_classes(self.num_classes)
        
        print(f"✓ Number of classes: {self.num_classes}")
        print(f"✓ Training samples: {len(self.train_loader.dataset):,}")
        print(f"✓ Validation samples: {len(self.val_loader.dataset):,}")
        print(f"✓ Test samples: {len(self.test_loader.dataset):,}")
        print(f"✓ Batch size: {config.BATCH_SIZES['vit']}")
        print(f"✓ Training batches: {len(self.train_loader)}")
        print(f"✓ Classes: {self.class_names}")
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision training
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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
            
            # Update progress bar (with accuracy like your CNN)
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
            
            # Periodic cache clearing
            if torch.cuda.is_available() and (batch_idx + 1) % 100 == 0:
                torch.cuda.empty_cache()
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [Val]  ")
            
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
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
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("VISION TRANSFORMER BASE TRAINING PIPELINE")
        print("="*60)
        
        # Print system info
        print(f"📊 System Information:")
        print(f"   Device: {config.DEVICE}")
        if torch.cuda.is_available():
            print(f"   GPU: {config.GPU_NAME}")
            print(f"   GPU Memory: {config.GPU_MEMORY:.1f} GB")
            print(f"   Mixed Precision: {'Enabled (FP16)' if self.use_amp else 'Disabled (FP32)'}")
            print(f"   Batch Size: {config.BATCH_SIZES['vit']}")
            print(f"   Image Size: {config.IMAGE_SIZES['vit']}x{config.IMAGE_SIZES['vit']}")
        
        # Load data
        self.get_dataloaders()
        
        # Setup model
        self.setup_model()
        
        patience_counter = 0
        
        print(f"\n🚀 Starting training for {config.NUM_EPOCHS} epochs")
        print("="*60)
        
        try:
            for epoch in range(config.NUM_EPOCHS):
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
                    print(f"   GPU Memory Used: {memory_used:.2f} GB")
                
                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    patience_counter = 0
                    
                    best_model_path = config.SAVED_MODELS_DIR / "vit_base_best.pth"
                    self.save_model(str(best_model_path))
                    print(f"   ✅ New best model saved! Val Acc: {val_acc:.2f}%")
                else:
                    patience_counter += 1
                    print(f"   ⚠️ No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                
                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = config.SAVED_MODELS_DIR / f"vit_base_epoch_{epoch+1}.pth"
                    self.save_checkpoint(str(checkpoint_path), epoch)
                
                # Early stopping
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                    break
                
                # Clear cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    gc.collect()
                
                print("="*60)
            
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
            interrupt_path = config.SAVED_MODELS_DIR / f"vit_base_interrupted_epoch_{epoch+1}.pth"
            self.save_checkpoint(str(interrupt_path), epoch)
            print(f"💾 Checkpoint saved to: {interrupt_path}")
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n❌ GPU Out of Memory Error!")
                print("Try reducing batch size for ViT in config.py")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e
        
        print("\n" + "="*60)
        print("ViT-BASE TRAINING COMPLETED")
        print("="*60)
        print(f"🎯 Best Validation Accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"📊 Best Validation Loss: {self.best_val_loss:.4f}")
        
        # Save final model
        final_model_path = config.SAVED_MODELS_DIR / "vit_base_final.pth"
        self.save_checkpoint(str(final_model_path), epoch)
    
    def evaluate(self):
        """Comprehensive evaluation on test set"""
        print("\n" + "="*60)
        print("ViT-BASE EVALUATION ON TEST SET")
        print("="*60)
        
        # Load best model
        best_model_path = config.SAVED_MODELS_DIR / "vit_base_best.pth"
        if os.path.exists(best_model_path):
            print(f"Loading best model from: {best_model_path}")
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded successfully")
        
        self.model.eval()
        all_preds = []
        all_targets = []
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
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_acc = 100. * correct / total
        test_loss = test_loss / len(self.test_loader)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        print(f"\n📊 ViT-Base Test Results:")
        print(f"   Test Loss: {test_loss:.4f}")
        print(f"   Test Accuracy: {test_acc:.2f}%")
        print(f"   F1 Score (Weighted): {f1:.4f}")
        
        # Compare with EfficientNet
        print(f"\n🔍 Comparison with EfficientNet-B3:")
        print(f"   EfficientNet-B3: 99.76% accuracy")
        print(f"   ViT-Base: {test_acc:.2f}% accuracy")
        print(f"   Difference: {test_acc - 99.76:+.2f}%")
        
        # Classification report
        print(f"\n📈 Classification Report:")
        print(classification_report(all_targets, all_preds, target_names=self.class_names))
        
        # Confusion matrix
        self.plot_confusion_matrix(all_targets, all_preds)
        
        # Training history
        self.plot_training_history()
        
        # Save results
        results = {
            'model': 'Vision-Transformer-Base',
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'f1_score_weighted': f1,
            'best_val_accuracy': self.best_val_acc,
            'best_epoch': self.best_epoch,
            'total_epochs': len(self.train_losses),
            'comparison': {
                'efficientnet_accuracy': 99.76,
                'difference': test_acc - 99.76
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = config.LOG_DIR / f"vit_base_results_{config.RUN_ID}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {results_path}")
        
        return test_acc, test_loss, f1
    
    def plot_confusion_matrix(self, targets, preds):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(targets, preds)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix - Vision Transformer Base', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = config.LOG_DIR / f'confusion_matrix_vit_base_{config.RUN_ID}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {cm_path}")
        plt.show()
    
    def plot_training_history(self):
        """Plot and save training history"""
        if len(self.train_losses) == 0:
            print("⚠️  No training history to plot")
            return
            
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('ViT-Base: Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('ViT-Base: Training and Validation Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(config.LOG_DIR / f'training_history_vit_base_{config.RUN_ID}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filename):
        """Save model weights only"""
        torch.save(self.model.state_dict(), filename)
    
    def save_checkpoint(self, filename, epoch):
        """Save full checkpoint"""
        checkpoint = {
            'epoch': epoch + 1,
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
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch
        }
        
        if self.use_amp and self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)

def main():
    """Main function to run ViT training"""
    # Initialize trainer
    trainer = ViTTrainer(num_classes=15)  # Will be updated from data
    
    # Train and evaluate
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()