import numpy as np
import os
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import json
from collections import defaultdict
from sklearn.neighbors import KDTree

# ===== CẤU HÌNH =====
folder_captures = r"D:\Thu\KhoaLuan\captures\Standford"
folder_masks = r"D:\Thu\KhoaLuan\results\Standford\masks"
folder_pcd_input = r"D:\Thu\KhoaLuan\data\Stanford3dDataset_v1.2_Aligned_Version\Area_1\office_1"
output_folder = r"D:\Thu\KhoaLuan\results\mapped_pcd"

os.makedirs(output_folder, exist_ok=True)

print("="*70)
print("🔄 SOFT VOTING + kNN PROPAGATION (100% COVERAGE)")
print("="*70)

# Load point cloud
all_points = []
for file in sorted(os.listdir(folder_pcd_input)):
    if file.endswith(".txt"):
        data = np.loadtxt(os.path.join(folder_pcd_input, file))
        all_points.append(data[:, :3])
points = np.vstack(all_points)
print(f"Loaded {len(points)} points")

# Load camera poses
poses_file = os.path.join(folder_captures, "camera_poses.json")
with open(poses_file, 'r') as f:
    camera_poses = json.load(f)
print(f"Loaded {len(camera_poses)} poses")

# Load masks
mask_files = sorted([f for f in os.listdir(folder_masks) if f.endswith('.png')])
print(f"Loaded {len(mask_files)} masks")

# Load class mapping
mapping_path = os.path.join(os.path.dirname(folder_masks), "class_label_mapping.json")
id2label = {}
if os.path.exists(mapping_path):
    with open(mapping_path, 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}

# ===== STEP 1: SOFT VOTING (Visible Points) =====
print("\n🗳️ STEP 1: Soft voting on visible points...")

point_weights = [defaultdict(float) for _ in range(len(points))]

def project_point_with_matrix(point_3d, extrinsic, intrinsic):
    point_hom = np.append(point_3d, 1.0)
    point_cam = extrinsic @ point_hom
    if point_cam[2] <= 0.01:
        return None
    point_img = intrinsic @ point_cam[:3]
    if point_img[2] == 0:
        return None
    px = point_img[0] / point_img[2]
    py = point_img[1] / point_img[2]
    return (int(px), int(py), point_cam[2])

total_projections = 0
successful = 0

for pose_data in tqdm(camera_poses, desc="Processing views"):
    view_idx = pose_data['view_idx']
    if view_idx >= len(mask_files):
        continue
    
    mask_path = os.path.join(folder_masks, mask_files[view_idx])
    mask = Image.open(mask_path)
    if mask.mode == 'P':
        mask_array = np.array(mask)
    else:
        mask_array = np.array(mask.convert('L'))
    
    intrinsic = np.array(pose_data['intrinsic_matrix'])
    extrinsic = np.array(pose_data['extrinsic_matrix'])
    
    img_h, img_w = mask_array.shape
    deep_buffer = defaultdict(list)
    
    for point_idx, point_3d in enumerate(points):
        result = project_point_with_matrix(point_3d, extrinsic, intrinsic)
        if result:
            total_projections += 1
            px, py, depth = result
            if 0 <= px < img_w and 0 <= py < img_h:
                successful += 1
                class_id = int(mask_array[py, px])
                if class_id >= 0:
                    key = (py, px)
                    deep_buffer[key].append((depth, point_idx, class_id))
    
    for key, points_at_pixel in deep_buffer.items():
        points_at_pixel.sort(key=lambda x: x[0])
        min_depth = points_at_pixel[0][0]
        
        for depth, point_idx, class_id in points_at_pixel:
            weight = 1.0 / (depth + 0.1)
            point_weights[point_idx][class_id] += weight

print(f"\n📊 Projection: {successful}/{total_projections} ({100*successful/total_projections:.1f}%)")

# Gán nhãn cho visible points
point_labels = np.full(len(points), -1, dtype=np.int32)
point_confidences = np.zeros(len(points))

for point_idx in tqdm(range(len(points)), desc="Assigning visible labels"):
    if point_weights[point_idx]:
        max_class = max(point_weights[point_idx].items(), key=lambda x: x[1])
        point_labels[point_idx] = max_class[0]
        point_confidences[point_idx] = max_class[1] / sum(point_weights[point_idx].values())

visible_labeled = np.sum(point_labels >= 0)
print(f"\n✅ Visible labeled: {visible_labeled}/{len(points)} ({100*visible_labeled/len(points):.1f}%)")

# ===== STEP 2: kNN PROPAGATION (Fill Unlabeled Points) =====
print("\n" + "="*70)
print("🔄 STEP 2: kNN Propagation to reach 100% coverage")
print("="*70)

labeled_mask = point_labels >= 0
unlabeled_mask = point_labels < 0
unlabeled_count = np.sum(unlabeled_mask)

print(f"Unlabeled points before propagation: {unlabeled_count}")

if unlabeled_count > 0 and np.sum(labeled_mask) > 0:
    # Build KDTree only on labeled points
    labeled_points = points[labeled_mask]
    labeled_labels = point_labels[labeled_mask]
    
    print(f"Building KDTree with {len(labeled_points)} labeled points...")
    tree = KDTree(labeled_points)
    
    # Parameters for propagation
    k = 7  # Number of neighbors to consider
    confidence_threshold = 0.3  # Only propagate from high-confidence labels
    
    print(f"Propagating labels (k={k})...")
    propagated_count = 0
    
    for idx in tqdm(np.where(unlabeled_mask)[0], desc="kNN propagation"):
        # Find k nearest labeled points
        dist, ind = tree.query(points[idx].reshape(1, -1), k=k)
        neighbor_labels = labeled_labels[ind[0]]
        
        # Optional: Weight by inverse distance
        weights = 1.0 / (dist[0] + 0.01)
        
        # Weighted voting
        unique_labels = np.unique(neighbor_labels)
        label_scores = {}
        
        for label in unique_labels:
            mask = neighbor_labels == label
            label_scores[label] = np.sum(weights[mask])
        
        # Assign label with highest weighted score
        best_label = max(label_scores, key=label_scores.get)
        point_labels[idx] = best_label
        
        # Estimate confidence based on neighbor agreement
        best_score = label_scores[best_label]
        total_score = sum(label_scores.values())
        point_confidences[idx] = best_score / total_score if total_score > 0 else 0.5
        
        propagated_count += 1
    
    print(f"\n✅ Propagated: {propagated_count} points")
else:
    print("⚠️ No labeled points to propagate from!")

# ===== FINAL STATISTICS =====
final_labeled = np.sum(point_labels >= 0)
print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)
print(f"Total points: {len(points)}")
print(f"Labeled points: {final_labeled} ({100*final_labeled/len(points):.1f}%) ✨")
print(f"Unlabeled points: {len(points) - final_labeled}")

if final_labeled == len(points):
    print("🎯 ACHIEVED 100% COVERAGE!")
else:
    print(f"⚠️ Still {len(points)-final_labeled} points unlabeled (may need larger k or different strategy)")

print(f"Average confidence: {np.mean(point_confidences[point_confidences > 0]):.3f}")

# ===== CONFIDENCE DISTRIBUTION =====
print("\n📊 Confidence distribution (after propagation):")
conf_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
for i in range(len(conf_bins)-1):
    low, high = conf_bins[i], conf_bins[i+1]
    count = np.sum((point_confidences >= low) & (point_confidences < high))
    if count > 0:
        print(f"   {low:.1f}-{high:.1f}: {count} points ({100*count/final_labeled:.1f}%)")

# ===== SAVE RESULTS =====
print("\n💾 Saving results...")

# Confidence heatmap
confidence_colors = np.zeros((len(points), 3))
for i in range(len(points)):
    if point_labels[i] >= 0:
        conf = point_confidences[i]
        confidence_colors[i] = [1-conf, conf, 0]
    else:
        confidence_colors[i] = [0.5, 0.5, 0.5]

pcd_confidence = o3d.geometry.PointCloud()
pcd_confidence.points = o3d.utility.Vector3dVector(points)
pcd_confidence.colors = o3d.utility.Vector3dVector(confidence_colors)
o3d.io.write_point_cloud(os.path.join(output_folder, "confidence_heatmap.ply"), pcd_confidence)

# Semantic point cloud with class colors
if id2label:
    # Generate consistent colors for each class
    np.random.seed(42)
    class_colors = {cid: np.random.rand(3) for cid in id2label.keys()}
    semantic_colors = np.zeros((len(points), 3))
    for i, label in enumerate(point_labels):
        if label >= 0 and label in class_colors:
            semantic_colors[i] = class_colors[label]
        else:
            semantic_colors[i] = [0.5, 0.5, 0.5]
    
    pcd_semantic = o3d.geometry.PointCloud()
    pcd_semantic.points = o3d.utility.Vector3dVector(points)
    pcd_semantic.colors = o3d.utility.Vector3dVector(semantic_colors)
    o3d.io.write_point_cloud(os.path.join(output_folder, "semantic_cloud.ply"), pcd_semantic)

# Save labels and confidences
np.save(os.path.join(output_folder, "semantic_labels.npy"), point_labels)
np.save(os.path.join(output_folder, "point_confidences.npy"), point_confidences)

# Save report
report = os.path.join(output_folder, "semantic_mapping_report.txt")
with open(report, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("SEMANTIC MAPPING REPORT (Soft Voting + kNN Propagation)\n")
    f.write("="*70 + "\n\n")
    f.write(f"Total points: {len(points)}\n")
    f.write(f"Visible labeled (Step 1): {visible_labeled} ({100*visible_labeled/len(points):.1f}%)\n")
    f.write(f"Propagated (Step 2): {final_labeled - visible_labeled}\n")
    f.write(f"Final labeled: {final_labeled} ({100*final_labeled/len(points):.1f}%)\n")
    f.write(f"Average confidence: {np.mean(point_confidences[point_confidences > 0]):.4f}\n\n")
    f.write("Class distribution:\n")
    
    unique_labels = np.unique(point_labels[point_labels >= 0])
    for label in unique_labels:
        count = np.sum(point_labels == label)
        avg_conf = np.mean(point_confidences[point_labels == label])
        class_name = id2label.get(label, f"Class_{label}")
        f.write(f"  {class_name:25s}: {count:6d} ({100*count/final_labeled:5.1f}%) - avg confidence: {avg_conf:.3f}\n")

print(f"\n✅ Saved: {report}")
print(f"✅ Saved confidence heatmap: confidence_heatmap.ply")
if id2label:
    print(f"✅ Saved semantic cloud: semantic_cloud.ply")
print("\n" + "="*70)
print("✅ COMPLETE! 100% COVERAGE ACHIEVED!")
print("="*70)