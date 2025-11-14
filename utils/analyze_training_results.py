"""
Analyze the training results to understand why accuracy is so high
"""
import json
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_results():
    print("📊 ANALYZING TRAINING RESULTS")
    print("=" * 60)
    
    # Find the latest results file
    logs_dir = Path("logs")
    result_files = list(logs_dir.glob("efficientnet_results_*.json"))
    
    if not result_files:
        print("❌ No results files found in logs/ directory")
        return
    
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"📁 Analyzing: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print(f"\n🎯 Model: {results['model']}")
    print(f"📅 Timestamp: {results['timestamp']}")
    print(f"✅ Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"📉 Test Loss: {results['test_loss']:.4f}")
    print(f"🎯 Best Val Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"🔢 Best Epoch: {results['best_epoch']}")
    print(f"🔄 Total Epochs: {results['total_epochs']}")
    
    # Analyze per-class performance
    report = results['classification_report']
    
    print(f"\n📈 PER-CLASS PERFORMANCE SUMMARY:")
    print("-" * 50)
    
    perfect_classes = []
    good_classes = []
    poor_classes = []
    
    for class_name, metrics in report.items():
        if class_name in ['accuracy', 'macro avg', 'weighted avg']:
            continue
            
        f1 = metrics['f1-score']
        precision = metrics['precision']
        recall = metrics['recall']
        
        if f1 > 0.99:
            perfect_classes.append(class_name)
        elif f1 > 0.95:
            good_classes.append(class_name)
        else:
            poor_classes.append(class_name)
    
    print(f"🎯 Perfect classes (F1 > 0.99): {len(perfect_classes)}")
    print(f"✅ Good classes (F1 > 0.95): {len(good_classes)}") 
    print(f"⚠️  Poor classes (F1 <= 0.95): {len(poor_classes)}")
    
    if poor_classes:
        print(f"\nClasses needing improvement:")
        for cls in poor_classes:
            f1 = report[cls]['f1-score']
            print(f"  - {cls}: F1 = {f1:.3f}")
    
    # Check if this is expected for PlantVillage
    print(f"\n💡 INTERPRETATION:")
    if results['test_accuracy'] > 99:
        print("• 99%+ accuracy is unusually high for real-world datasets")
        print("• PlantVillage is known to be very 'clean' with clear class distinctions")
        print("• This might indicate the dataset is easier than real field conditions")
        print("• Consider testing on more challenging datasets for real-world applicability")
    else:
        print("• Results are in expected range for this problem")

def check_training_plots():
    """Check if training plots show normal learning"""
    print(f"\n📈 CHECKING TRAINING PLOTS")
    print("=" * 60)
    
    logs_dir = Path("logs")
    plot_files = list(logs_dir.glob("training_history_efficientnet_*.png"))
    
    if plot_files:
        latest_plot = max(plot_files, key=lambda x: x.stat().st_mtime)
        print(f"📊 Training plot: {latest_plot.name}")
        print("• Open this file to check if learning curves look normal")
        print("• Look for: Smooth decrease in loss, gradual increase in accuracy")
        print("• Watch for: Sudden jumps (possible issues) or flat lines (not learning)")
    else:
        print("❌ No training plot files found")

if __name__ == "__main__":
    analyze_results()
    check_training_plots()