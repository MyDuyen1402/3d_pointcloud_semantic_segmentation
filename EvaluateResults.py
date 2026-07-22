import numpy as np
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re

# ===== CẤU HÌNH =====
folder_ground_truth = r"D:\Thu\KhoaLuan\data\Stanford3dDataset_v1.2_Aligned_Version\Area_1\office_1\Annotations"
folder_segmented = r"D:\Thu\KhoaLuan\results\Standford"
pcd_labels_path = r"D:\Thu\KhoaLuan\results\mapped_pcd\semantic_labels.npy"
output_folder = r"D:\Thu\KhoaLuan\results\evaluation_by_classname"

os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("📊 SEMANTIC SEGMENTATION EVALUATION (With Class Merging)")
print("="*70)

# ===== ĐỊNH NGHĨA CLASS MAPPING ĐỂ GỘP =====
# Ví dụ: gộp window và windowpane, các class tương tự
class_merging = {
    #'window': 'window',           # GT 'window' -> 'window'
    'window': 'windowpane',       # GT 'windowpane' -> 'window'
    # 'board': 'wood',              # Nếu muốn gộp
    # 'beam': 'ceiling',            # Nếu muốn gộp beam vào ceiling
    # Thêm các mapping khác nếu cần
}

# ===== 1. LOAD CLASS MAPPING =====
print("\n📋 Loading class ID to name mapping...")

mapping_path = os.path.join(folder_segmented, "class_label_mapping.json")

try:
    with open(mapping_path, 'r') as f:
        id2label = json.load(f)
        id2label = {int(k): v for k, v in id2label.items()}
    print(f"✅ Loaded class mapping with {len(id2label)} classes")
except Exception as e:
    print(f"❌ Could not load mapping: {e}")
    exit()

# ===== 2. LOAD GROUND TRUTH =====
print(f"\n📁 Loading ground truth from: {folder_ground_truth}")

if not os.path.exists(folder_ground_truth):
    print(f"❌ Folder not found: {folder_ground_truth}")
    exit()

gt_all_points = []
gt_all_labels = []
gt_file_info = []

txt_files = sorted([f for f in os.listdir(folder_ground_truth) if f.endswith('.txt')])
print(f"✅ Found {len(txt_files)} txt files\n")

for txt_file in txt_files:
    class_name = Path(txt_file).stem
    class_name_clean = re.sub(r'_\d+$', '', class_name)
    
    # ===== ÁP DỤNG MERGING CHO GT =====
    if class_name_clean in class_merging:
        class_name_original = class_name_clean
        class_name_clean = class_merging[class_name_clean]
        print(f"   {txt_file:35s} → '{class_name_original:15s}' → MERGED to '{class_name_clean}'")
    else:
        print(f"   {txt_file:35s} → Class '{class_name_clean:20s}'")
    
    file_path = os.path.join(folder_ground_truth, txt_file)
    try:
        data = np.loadtxt(file_path)
        points = data[:, :3] if data.ndim > 1 else data.reshape(1, -1)
        num_points = len(points)
        
        gt_all_points.append(points)
        gt_all_labels.append(np.array([class_name_clean] * num_points))
        gt_file_info.append((txt_file, class_name_clean, num_points))
    
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:40]}")
        continue

gt_points_all = np.vstack(gt_all_points)
gt_labels_all = np.hstack(gt_all_labels)

print(f"\n✅ Total ground truth points: {len(gt_points_all)}")

unique_gt_classnames = np.unique(gt_labels_all)
print(f"✅ Unique classes in GT (after merging): {len(unique_gt_classnames)}")
for class_name in unique_gt_classnames:
    count = np.sum(gt_labels_all == class_name)
    print(f"      • {class_name:20s} - {count:8d} points ({100*count/len(gt_labels_all):.1f}%)")

# ===== 3. LOAD PREDICTED LABELS =====
print("\n📦 Loading predicted labels...")

if os.path.exists(pcd_labels_path):
    predicted_labels_id = np.load(pcd_labels_path)
    print(f"✅ Loaded {len(predicted_labels_id)} predicted labels")
else:
    print(f"❌ File not found: {pcd_labels_path}")
    exit()

# ===== 4. CONVERT PREDICTED IDs TO CLASS NAMES (WITH MERGING) =====
print("\n🔄 Converting and merging predicted labels...")

predicted_labels_raw = np.array([
    id2label.get(int(label_id), f"Unknown_{label_id}") if label_id >= 0 else "Unlabeled"
    for label_id in predicted_labels_id
])

# Áp dụng merging cho predicted labels
predicted_labels_name = predicted_labels_raw.copy()
for old_name, new_name in class_merging.items():
    predicted_labels_name[predicted_labels_raw == old_name] = new_name

print(f"✅ Converted and merged")
print(f"   Unique classes in predictions: {len(np.unique(predicted_labels_name))}")

# ===== 5. CHECK POINT COUNT =====
print("\n🔍 Checking point counts...")
print(f"   Ground truth:  {len(gt_labels_all)}")
print(f"   Predictions:   {len(predicted_labels_name)}")

if len(gt_labels_all) != len(predicted_labels_name):
    min_len = min(len(gt_labels_all), len(predicted_labels_name))
    gt_labels_all = gt_labels_all[:min_len]
    predicted_labels_name = predicted_labels_name[:min_len]
    print(f"   Trimmed to {min_len} points")
else:
    print(f"✅ Point counts match!")

# ===== 6. EVALUATION =====
print("\n" + "="*70)
print("📊 EVALUATION RESULTS (After Class Merging)")
print("="*70)

all_class_names = sorted(np.unique(np.concatenate([gt_labels_all, predicted_labels_name])))

correct_predictions = np.sum(gt_labels_all == predicted_labels_name)
total_predictions = len(gt_labels_all)
overall_accuracy = correct_predictions / total_predictions

print(f"\n📈 Overall Accuracy: {overall_accuracy*100:.2f}%")
print(f"   ({correct_predictions}/{total_predictions} correct predictions)")

# Per-class metrics
class_metrics = {}

print("\n📊 Per-Class Metrics:")
print("   " + "-"*100)
print(f"   {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>8} {'IoU':>8} {'TP':>7} {'FP':>7} {'FN':>7} {'Support':>8}")
print("   " + "-"*100)

for class_name in all_class_names:
    gt_mask = gt_labels_all == class_name
    pred_mask = predicted_labels_name == class_name
    
    tp = np.sum(gt_mask & pred_mask)
    fp = np.sum(~gt_mask & pred_mask)
    fn = np.sum(gt_mask & ~pred_mask)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    support = np.sum(gt_mask)
    
    class_metrics[class_name] = {
        'precision': precision, 'recall': recall, 'f1': f1, 'iou': iou,
        'support': support, 'tp': tp, 'fp': fp, 'fn': fn
    }
    
    if support > 0:
        print(f"   {class_name:<25} {precision:>10.4f} {recall:>10.4f} {f1:>8.4f} {iou:>8.4f} {tp:>7d} {fp:>7d} {fn:>7d} {support:>8d}")

print("   " + "-"*100)

# Macro average (chỉ trên class có trong GT)
gt_classes = [c for c in all_class_names if class_metrics[c]['support'] > 0]
macro_precision = np.mean([class_metrics[c]['precision'] for c in gt_classes])
macro_recall = np.mean([class_metrics[c]['recall'] for c in gt_classes])
macro_f1 = np.mean([class_metrics[c]['f1'] for c in gt_classes])
macro_iou = np.mean([class_metrics[c]['iou'] for c in gt_classes])

print(f"\n📊 Macro Average:")
print(f"   Precision: {macro_precision:.4f} | Recall: {macro_recall:.4f} | F1: {macro_f1:.4f} | mIoU: {macro_iou:.4f}")

# Weighted average
weights = np.array([class_metrics[c]['support'] for c in gt_classes])
if np.sum(weights) > 0:
    weights = weights / np.sum(weights)
    weighted_precision = np.sum([class_metrics[c]['precision'] * w for c, w in zip(gt_classes, weights)])
    weighted_recall = np.sum([class_metrics[c]['recall'] * w for c, w in zip(gt_classes, weights)])
    weighted_f1 = np.sum([class_metrics[c]['f1'] * w for c, w in zip(gt_classes, weights)])
    
    print(f"\n📊 Weighted Average:")
    print(f"   Precision: {weighted_precision:.4f} | Recall: {weighted_recall:.4f} | F1: {weighted_f1:.4f}")

# ===== 7. CONFUSION MATRIX =====
print("\n📊 Creating confusion matrix...")

cm_classes = gt_classes
cm = confusion_matrix(gt_labels_all, predicted_labels_name, labels=cm_classes)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Counts
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=cm_classes, yticklabels=cm_classes,
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Counts) - After Merging', fontweight='bold')
axes[0].set_xlabel('Predicted Class')
axes[0].set_ylabel('Ground Truth Class')
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

# Normalized
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.nan_to_num(cm_norm)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=cm_classes, yticklabels=cm_classes,
            cbar_kws={'label': 'Percentage'})
axes[1].set_title('Confusion Matrix (Normalized) - After Merging', fontweight='bold')
axes[1].set_xlabel('Predicted Class')
axes[1].set_ylabel('Ground Truth Class')
plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
cm_path = os.path.join(output_folder, "confusion_matrix_merged.png")
plt.savefig(cm_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {cm_path}")
plt.close()

# ===== 8. SAVE REPORT =====
print("\n📝 Saving report...")

report_path = os.path.join(output_folder, "evaluation_report_merged.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("SEMANTIC SEGMENTATION EVALUATION REPORT (With Class Merging)\n")
    f.write("="*70 + "\n\n")
    
    f.write("CLASS MERGING RULES:\n")
    for old, new in class_merging.items():
        f.write(f"  • '{old}' → '{new}'\n")
    f.write("\n")
    
    f.write(f"Overall Accuracy: {overall_accuracy*100:.2f}%\n")
    f.write(f"Mean IoU (mIoU): {macro_iou:.4f}\n")
    f.write(f"Macro F1-Score: {macro_f1:.4f}\n\n")
    
    f.write("Per-Class Metrics:\n")
    f.write("-"*70 + "\n")
    for class_name in gt_classes:
        m = class_metrics[class_name]
        f.write(f"  {class_name:20s} - P:{m['precision']:.3f} R:{m['recall']:.3f} F1:{m['f1']:.3f} IoU:{m['iou']:.3f}\n")

print(f"✅ Saved: {report_path}")

print("\n" + "="*70)
print("✅ EVALUATION COMPLETED!")
print(f"📂 Results: {output_folder}")
print("="*70)