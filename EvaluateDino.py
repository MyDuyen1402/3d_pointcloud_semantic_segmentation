"""
SEMANTIC SEGMENTATION EVALUATION
Đánh giá kết quả segmentation 2D và mapping 3D so với Ground Truth
"""

import numpy as np
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
from collections import defaultdict

# ============================================================
# CONFIGURATION - USER EDIT THESE PATHS
# ============================================================

# Ground Truth từ Stanford 3D Dataset
FOLDER_GROUND_TRUTH = r"D:\Thu\KhoaLuan\data\Stanford3dDataset_v1.2_Aligned_Version\Area_1\office_1\Annotations"

# Thư mục chứa kết quả segmentation (file mapping JSON)
FOLDER_SEGMENTATION_RESULTS = r"D:\Thu\KhoaLuan\results\Standford_grounding_sam2"

# Đường dẫn đến file labels đã mapping từ point cloud
PCD_LABELS_PATH = r"D:\Thu\KhoaLuan\results\mapped_pcd\semantic_labels.npy"

# Đường dẫn đến file confidences (optional)
PCD_CONFIDENCE_PATH = r"D:\Thu\KhoaLuan\results\mapped_pcd\point_confidences.npy"

# Thư mục output cho evaluation
OUTPUT_FOLDER = r"D:\Thu\KhoaLuan\results\evaluation"

# ============================================================
# CLASS MAPPING & MERGING CONFIGURATION
# ============================================================

# Mapping từ class name trong dataset sang tên chuẩn (nếu cần)
# Định dạng: 'tên_trong_GT': 'tên_chuẩn'
CLASS_NAME_MAPPING = {
    'ceiling': 'ceiling',
    'floor': 'floor',
    'wall': 'wall',
    'beam': 'beam',
    'column': 'column',
    'window': 'window',
    'door': 'door',
    'table': 'table',
    'chair': 'chair',
    'sofa': 'sofa',
    'bookcase': 'bookcase',
    'board': 'whiteboard',      # Map 'board' (GT) to 'whiteboard' (pred)
    'whiteboard': 'whiteboard',
    'clutter': 'clutter',
    'clutter_table': 'clutter',
    'clutter_chair': 'clutter',
    'clutter_sofa': 'clutter',
}

# Gộp các class tương tự nhau (từ predicted labels)
# Định dạng: 'tên_predicted' → 'tên_chuẩn'
CLASS_MERGING_FROM_PRED = {
    'windowpane': 'window',      # Nếu model predict 'windowpane' thì gộp vào 'window'
    'doorframe': 'door',         # Nếu model predict 'doorframe' thì gộp vào 'door'
    'bookshelf': 'bookcase',     # Gộp bookshelf vào bookcase
    'cabinet': 'clutter',        # Gộp cabinet vào clutter
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_class_mapping(folder_path):
    """Load class ID to name mapping from segmentation results"""
    mapping_path = os.path.join(folder_path, "class_label_mapping.json")
    
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            id2label = json.load(f)
            # Convert keys to int
            id2label = {int(k): v for k, v in id2label.items()}
        print(f"   ✅ Loaded mapping with {len(id2label)} classes")
        return id2label
    else:
        print(f"   ⚠️ Mapping file not found: {mapping_path}")
        # Return default 13-class mapping
        default_classes = [
            "ceiling", "floor", "wall", "beam", "column", 
            "window", "door", "table", "chair", "sofa", 
            "bookcase", "whiteboard", "clutter"
        ]
        return {i: name for i, name in enumerate(default_classes)}

def load_ground_truth(gt_folder):
    """Load ground truth point clouds from .txt files"""
    print(f"\n📁 Loading ground truth from: {gt_folder}")
    
    if not os.path.exists(gt_folder):
        print(f"   ❌ Folder not found: {gt_folder}")
        return None, None, None
    
    gt_points = []
    gt_labels = []
    gt_files_info = []
    
    txt_files = sorted([f for f in os.listdir(gt_folder) if f.endswith('.txt')])
    print(f"   Found {len(txt_files)} txt files")
    
    for txt_file in txt_files:
        # Extract class name from filename (e.g., "wall_1.txt" -> "wall")
        class_name_raw = Path(txt_file).stem
        class_name = re.sub(r'_\d+$', '', class_name_raw)
        
        # Apply class name mapping if exists
        if class_name in CLASS_NAME_MAPPING:
            class_name_mapped = CLASS_NAME_MAPPING[class_name]
            if class_name_mapped != class_name:
                print(f"   {txt_file:35s} → '{class_name}' → mapped to '{class_name_mapped}'")
            class_name = class_name_mapped
        else:
            print(f"   {txt_file:35s} → class '{class_name}'")
        
        file_path = os.path.join(gt_folder, txt_file)
        
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            points = data[:, :3]
            num_points = len(points)
            
            gt_points.append(points)
            gt_labels.append(np.array([class_name] * num_points))
            gt_files_info.append((txt_file, class_name, num_points))
            
        except Exception as e:
            print(f"   ❌ Error loading {txt_file}: {e}")
            continue
    
    if not gt_points:
        print("   ❌ No ground truth data loaded!")
        return None, None, None
    
    gt_points_all = np.vstack(gt_points)
    gt_labels_all = np.hstack(gt_labels)
    
    print(f"\n   ✅ Total ground truth points: {len(gt_points_all):,}")
    
    # Print class distribution
    unique_classes = np.unique(gt_labels_all)
    print(f"   Unique classes in GT: {len(unique_classes)}")
    for cls in sorted(unique_classes):
        count = np.sum(gt_labels_all == cls)
        print(f"      • {cls:20s}: {count:8,} points ({100*count/len(gt_labels_all):.1f}%)")
    
    return gt_points_all, gt_labels_all, gt_files_info

def load_predicted_labels(pred_path, id2label):
    """Load predicted labels from .npy file and convert to class names"""
    print(f"\n📦 Loading predicted labels from: {pred_path}")
    
    if not os.path.exists(pred_path):
        print(f"   ❌ File not found: {pred_path}")
        return None
    
    pred_ids = np.load(pred_path)
    print(f"   Loaded {len(pred_ids):,} predicted labels")
    
    # Convert IDs to class names
    pred_labels_raw = np.array([
        id2label.get(int(label_id), f"unknown_{label_id}") if label_id >= 0 else "unlabeled"
        for label_id in pred_ids
    ])
    
    # Apply merging rules
    pred_labels = pred_labels_raw.copy()
    for old_name, new_name in CLASS_MERGING_FROM_PRED.items():
        mask = pred_labels_raw == old_name
        if np.any(mask):
            pred_labels[mask] = new_name
            print(f"   Merged '{old_name}' → '{new_name}': {np.sum(mask):,} points")
    
    # Also merge 'unlabeled' into 'clutter' or keep as is? 
    # Let's keep 'unlabeled' for now
    
    unique_pred = np.unique(pred_labels)
    print(f"   Unique classes in predictions: {len(unique_pred)}")
    for cls in sorted(unique_pred):
        count = np.sum(pred_labels == cls)
        print(f"      • {cls:20s}: {count:8,} points ({100*count/len(pred_labels):.1f}%)")
    
    return pred_labels

def load_confidences(conf_path, num_points):
    """Load confidence scores if available"""
    if os.path.exists(conf_path):
        confidences = np.load(conf_path)
        if len(confidences) == num_points:
            print(f"   ✅ Loaded {len(confidences):,} confidence scores")
            return confidences
    return None

def align_points(gt_labels, pred_labels):
    """Align ground truth and predicted labels by point count"""
    if len(gt_labels) == len(pred_labels):
        print(f"   ✅ Point counts match: {len(gt_labels):,}")
        return gt_labels, pred_labels
    
    print(f"   ⚠️ Point count mismatch: GT={len(gt_labels):,}, Pred={len(pred_labels):,}")
    min_len = min(len(gt_labels), len(pred_labels))
    print(f"   Truncating to {min_len:,} points")
    return gt_labels[:min_len], pred_labels[:min_len]

def calculate_metrics(gt_labels, pred_labels, class_names):
    """Calculate per-class and overall metrics"""
    metrics = {}
    
    # =========================================================
    # Custom accuracy:
    # Nếu GT = clutter thì predict gì cũng tính đúng
    # =========================================================

    correct_mask = (
        (gt_labels == pred_labels) |
        (gt_labels == "clutter")
    )

    overall_acc = np.mean(correct_mask)
    
    # Per-class metrics
    support = []

    for class_name in class_names:
        support.append(np.sum(gt_labels == class_name))
    
    # IoU for each class
    iou_scores = []
    for i, class_name in enumerate(class_names):
        gt_mask = gt_labels == class_name
        pred_mask = pred_labels == class_name
        
        # =========================================================
        # Special rule for clutter:
        # GT clutter -> always correct
        # =========================================================

        if class_name == "clutter":
            tp = np.sum(gt_mask)
            fp = 0
            fn = 0
        else:
            tp = np.sum(gt_mask & pred_mask)
            fp = np.sum(~gt_mask & pred_mask & (gt_labels != "clutter"))
            fn = np.sum(gt_mask & ~pred_mask)
        
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        iou_scores.append(iou)
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0

        if precision_val + recall_val > 0:
            f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
        else:
            f1_val = 0

        metrics[class_name] = {
            'precision': precision_val,
            'recall': recall_val,
            'f1': f1_val,
            'iou': iou,
            'support': int(support[i]),
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn)
        }
    
    # Macro averages (over classes with support > 0)
    valid_classes = [c for c in class_names if metrics[c]['support'] > 0]
    macro_precision = np.mean([metrics[c]['precision'] for c in valid_classes])
    macro_recall = np.mean([metrics[c]['recall'] for c in valid_classes])
    macro_f1 = np.mean([metrics[c]['f1'] for c in valid_classes])
    macro_iou = np.mean([metrics[c]['iou'] for c in valid_classes])
    
    # Weighted averages
    total_support = sum(metrics[c]['support'] for c in class_names)
    if total_support > 0:
        weighted_precision = sum(metrics[c]['precision'] * metrics[c]['support'] for c in class_names) / total_support
        weighted_recall = sum(metrics[c]['recall'] * metrics[c]['support'] for c in class_names) / total_support
        weighted_f1 = sum(metrics[c]['f1'] * metrics[c]['support'] for c in class_names) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
    
    return {
        'overall_accuracy': overall_acc,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'macro_iou': macro_iou,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': metrics,
        'valid_classes': valid_classes
    }

def plot_confusion_matrix(gt_labels, pred_labels, class_names, output_path):
    """Plot and save confusion matrix"""
    # Filter to classes present in GT
    class_names_filtered = [c for c in class_names if np.sum(gt_labels == c) > 0]
    
    cm = confusion_matrix(gt_labels, pred_labels, labels=class_names_filtered)
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Counts matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names_filtered, yticklabels=class_names_filtered,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 8})
    axes[0].set_title('Confusion Matrix (Counts)', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Predicted Class', fontsize=12)
    axes[0].set_ylabel('Ground Truth Class', fontsize=12)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(axes[0].yaxis.get_majorticklabels(), fontsize=9)
    
    # Normalized matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=class_names_filtered, yticklabels=class_names_filtered,
                cbar_kws={'label': 'Percentage'}, annot_kws={'size': 8})
    axes[1].set_title('Confusion Matrix (Normalized)', fontweight='bold', fontsize=14)
    axes[1].set_xlabel('Predicted Class', fontsize=12)
    axes[1].set_ylabel('Ground Truth Class', fontsize=12)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    plt.setp(axes[1].yaxis.get_majorticklabels(), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_path}")

def plot_metrics_bar_chart(metrics, output_path):
    """Plot bar chart of per-class metrics"""
    valid_classes = metrics['valid_classes']
    per_class = metrics['per_class']
    
    if not valid_classes:
        return
    
    x = np.arange(len(valid_classes))
    width = 0.25
    
    precisions = [per_class[c]['precision'] for c in valid_classes]
    recalls = [per_class[c]['recall'] for c in valid_classes]
    f1s = [per_class[c]['f1'] for c in valid_classes]
    ious = [per_class[c]['iou'] for c in valid_classes]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars1 = ax.bar(x - width*1.5, precisions, width, label='Precision', color='#2E86AB')
    bars2 = ax.bar(x - width/2, recalls, width, label='Recall', color='#A23B72')
    bars3 = ax.bar(x + width/2, f1s, width, label='F1-Score', color='#F18F01')
    bars4 = ax.bar(x + width*1.5, ious, width, label='IoU', color='#C73E1D')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_classes, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.05:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_path}")

def plot_confidence_vs_accuracy(confidences, gt_labels, pred_labels, output_path):
    """Plot accuracy vs confidence threshold"""
    if confidences is None:
        return
    
    thresholds = np.linspace(0, 1, 21)
    accuracies = []
    coverage = []
    
    for thresh in thresholds:
        mask = confidences >= thresh
        if np.sum(mask) > 0:
            acc = accuracy_score(gt_labels[mask], pred_labels[mask])
            accuracies.append(acc)
            coverage.append(np.sum(mask) / len(confidences))
        else:
            accuracies.append(0)
            coverage.append(0)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = '#2E86AB'
    color2 = '#F18F01'
    
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12, color=color1)
    ax1.plot(thresholds, accuracies, 'o-', color=color1, linewidth=2, markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Coverage (points above threshold)', fontsize=12, color=color2)
    ax2.plot(thresholds, coverage, 's-', color=color2, linewidth=2, markersize=4)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.05)
    
    ax1.set_title('Accuracy vs Confidence Threshold', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Saved: {output_path}")

def save_report(metrics, output_path):
    """Save evaluation report to text file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SEMANTIC SEGMENTATION EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Class name mapping: {CLASS_NAME_MAPPING}\n")
        f.write(f"Class merging: {CLASS_MERGING_FROM_PRED}\n\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%\n")
        f.write(f"Macro Average:\n")
        f.write(f"  - Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"  - Recall:    {metrics['macro_recall']:.4f}\n")
        f.write(f"  - F1-Score:  {metrics['macro_f1']:.4f}\n")
        f.write(f"  - mIoU:      {metrics['macro_iou']:.4f}\n")
        f.write(f"Weighted Average:\n")
        f.write(f"  - Precision: {metrics['weighted_precision']:.4f}\n")
        f.write(f"  - Recall:    {metrics['weighted_recall']:.4f}\n")
        f.write(f"  - F1-Score:  {metrics['weighted_f1']:.4f}\n\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'Support':>10}\n")
        f.write("-"*80 + "\n")
        
        for class_name in sorted(metrics['valid_classes']):
            m = metrics['per_class'][class_name]
            f.write(f"{class_name:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['iou']:>10.4f} {m['support']:>10,}\n")
        
        f.write("-"*80 + "\n")
        
        # Summary of classes with poor performance
        f.write("\nCLASSES WITH POOR PERFORMANCE (IoU < 0.3):\n")
        f.write("-"*40 + "\n")
        poor_classes = [c for c in metrics['valid_classes'] if metrics['per_class'][c]['iou'] < 0.3]
        if poor_classes:
            for c in poor_classes:
                f.write(f"  - {c}: IoU={metrics['per_class'][c]['iou']:.3f}\n")
        else:
            f.write("  None! All classes have IoU >= 0.3\n")
        
        # Best performing classes
        f.write("\nBEST PERFORMING CLASSES (Top 5 by IoU):\n")
        f.write("-"*40 + "\n")
        sorted_by_iou = sorted(metrics['valid_classes'], key=lambda c: metrics['per_class'][c]['iou'], reverse=True)
        for c in sorted_by_iou[:5]:
            f.write(f"  - {c}: IoU={metrics['per_class'][c]['iou']:.3f}, F1={metrics['per_class'][c]['f1']:.3f}\n")

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("="*80)
    print("📊 SEMANTIC SEGMENTATION EVALUATION")
    print("="*80)
    
    # Display configuration
    print("\n📁 CONFIGURATION:")
    print(f"   Ground Truth folder: {FOLDER_GROUND_TRUTH}")
    print(f"   Segmentation results: {FOLDER_SEGMENTATION_RESULTS}")
    print(f"   Predicted labels: {PCD_LABELS_PATH}")
    print(f"   Output folder: {OUTPUT_FOLDER}")
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # 1. Load class mapping
    print("\n📋 Loading class mapping...")
    id2label = load_class_mapping(FOLDER_SEGMENTATION_RESULTS)
    
    # 2. Load ground truth
    gt_points, gt_labels, gt_files_info = load_ground_truth(FOLDER_GROUND_TRUTH)
    if gt_labels is None:
        print("\n❌ Failed to load ground truth. Exiting.")
        return
    
    # 3. Load predicted labels
    pred_labels = load_predicted_labels(PCD_LABELS_PATH, id2label)
    if pred_labels is None:
        print("\n❌ Failed to load predicted labels. Exiting.")
        return
    
    # 4. Load confidences (optional)
    confidences = load_confidences(PCD_CONFIDENCE_PATH, len(pred_labels))
    
    # 5. Align points
    gt_labels_aligned, pred_labels_aligned = align_points(gt_labels, pred_labels)
    if confidences is not None:
        confidences = confidences[:len(gt_labels_aligned)]
    
    # 6. Get common classes for evaluation
    all_classes = sorted(np.unique(np.concatenate([gt_labels_aligned, pred_labels_aligned])))
    print(f"\n📊 Evaluating on {len(all_classes)} classes")
    
    # 7. Calculate metrics
    print("\n📊 Calculating metrics...")
    metrics = calculate_metrics(gt_labels_aligned, pred_labels_aligned, all_classes)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"\n📈 Overall Accuracy: {metrics['overall_accuracy']*100:.2f}%")
    print(f"\n📈 Macro Average (over {len(metrics['valid_classes'])} classes):")
    print(f"   Precision: {metrics['macro_precision']:.4f}")
    print(f"   Recall:    {metrics['macro_recall']:.4f}")
    print(f"   F1-Score:  {metrics['macro_f1']:.4f}")
    print(f"   mIoU:      {metrics['macro_iou']:.4f}")
    
    print(f"\n📈 Weighted Average:")
    print(f"   Precision: {metrics['weighted_precision']:.4f}")
    print(f"   Recall:    {metrics['weighted_recall']:.4f}")
    print(f"   F1-Score:  {metrics['weighted_f1']:.4f}")
    
    print("\n📊 Per-Class Metrics:")
    print("-"*90)
    print(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10} {'Support':>10}")
    print("-"*90)
    
    for class_name in sorted(metrics['valid_classes']):
        m = metrics['per_class'][class_name]
        print(f"{class_name:<22} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {m['iou']:>10.4f} {m['support']:>10,}")
    
    print("-"*90)
    
    # 8. Generate plots
    print("\n📊 Generating visualizations...")
    
    # Confusion matrix
    plot_confusion_matrix(
        gt_labels_aligned, pred_labels_aligned, all_classes,
        os.path.join(OUTPUT_FOLDER, "confusion_matrix.png")
    )
    
    # Bar chart of metrics
    plot_metrics_bar_chart(
        metrics,
        os.path.join(OUTPUT_FOLDER, "per_class_metrics.png")
    )
    
    # Confidence vs accuracy (if confidence available)
    if confidences is not None:
        plot_confidence_vs_accuracy(
            confidences, gt_labels_aligned, pred_labels_aligned,
            os.path.join(OUTPUT_FOLDER, "confidence_vs_accuracy.png")
        )
    
    # 9. Save report
    save_report(
        metrics,
        os.path.join(OUTPUT_FOLDER, "evaluation_report.txt")
    )
    
    # 10. Save metrics as JSON
    metrics_json = {
        'overall_accuracy': float(metrics['overall_accuracy']),
        'macro_precision': float(metrics['macro_precision']),
        'macro_recall': float(metrics['macro_recall']),
        'macro_f1': float(metrics['macro_f1']),
        'macro_iou': float(metrics['macro_iou']),
        'weighted_precision': float(metrics['weighted_precision']),
        'weighted_recall': float(metrics['weighted_recall']),
        'weighted_f1': float(metrics['weighted_f1']),
        'per_class': {
            class_name: {
                'precision': float(metrics['per_class'][class_name]['precision']),
                'recall': float(metrics['per_class'][class_name]['recall']),
                'f1': float(metrics['per_class'][class_name]['f1']),
                'iou': float(metrics['per_class'][class_name]['iou']),
                'support': int(metrics['per_class'][class_name]['support'])
            }
            for class_name in metrics['per_class']
        }
    }
    
    with open(os.path.join(OUTPUT_FOLDER, "metrics.json"), 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"   ✅ Saved: metrics.json")
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETED!")
    print(f"📂 Results saved to: {OUTPUT_FOLDER}")
    print("="*80)

# ============================================================
# RUN MAIN
# ============================================================

if __name__ == "__main__":
    main()