"""
SOFT VOTING + Z-BUFFER + kNN PROPAGATION WITH CONFIDENCE
Class-balanced weights dựa trên pixel counts (không phải image counts)
Support: Multiple rooms, segments, checkpoint/resume
FIXED: Save points.npy for evaluation
"""

import numpy as np
import os
from PIL import Image
import open3d as o3d
from tqdm import tqdm
import json
from collections import defaultdict
from sklearn.neighbors import KDTree

# ============================================================
# CONFIGURATION
# ============================================================

FOLDER_CAPTURES = r"D:\Thu\KhoaLuan\captures\Standford\Area_6"
FOLDER_MASKS_ROOT = r"D:\Thu\KhoaLuan\results\Standford_grounding_sam2\Area_6"
FOLDER_PCD_INPUT = r"D:\Thu\KhoaLuan\data\Stanford3dDataset_v1.2_Aligned_Version\Area_6"
MAPPING_PATH = r"D:\Thu\KhoaLuan\results\Standford_grounding_sam2\class_label_mapping.json"
FULL_SUMMARY_PATH = r"D:\Thu\KhoaLuan\results\Standford_grounding_sam2\full_detection_summary.json"
OUTPUT_ROOT = r"D:\Thu\KhoaLuan\results\mapped_pcd\Area_6"

# Checkpoint file
CHECKPOINT_FILE = os.path.join(OUTPUT_ROOT, "mapping_checkpoint.json")

# Hyperparameters
K_NEIGHBORS = 7
CONFIDENCE_THRESHOLD = 0.3
PROPAGATION_CONFIDENCE_THRESHOLD = 0.6
DEPTH_WEIGHT_FACTOR = 0.1
DEPTH_TOLERANCE = 0.05
MIN_TOTAL_WEIGHT = 0.5
MIN_VISIBLE_VIEWS = 2

# Class-Balanced params
BETA = 0.999
BOOST_FACTOR = {'beam': 2.0, 'whiteboard': 1.0}

# ============================================================
# CLASS COLORS
# ============================================================

CLASS_COLORS_CONSISTENT = {
    'ceiling': [0.78, 0.78, 0.78],
    'floor': [0.55, 0.27, 0.07],
    'wall': [0.78, 0.71, 0.59],
    'beam': [1.0, 0.0, 0.0],
    'column': [0.59, 0.59, 0.59],
    'window': [0.53, 0.81, 0.92],
    'door': [0.80, 0.52, 0.25],
    'table': [1.00, 0.65, 0.00],
    'chair': [0.00, 0.78, 0.00],
    'sofa': [1.00, 0.41, 0.71],
    'bookcase': [0.63, 0.39, 0.24],
    'whiteboard': [0.39, 0.39, 0.78],
}

# ============================================================
# CLASS-BALANCED WEIGHT COMPUTATION
# ============================================================

def compute_class_balanced_weights_from_pixels(full_summary_path, beta=0.999, boost_factor=None):
    if not os.path.exists(full_summary_path):
        print(f"   ⚠️ File not found: {full_summary_path}")
        return None, None, 0
    
    with open(full_summary_path, 'r') as f:
        results = json.load(f)
    
    class_pixel_counts = defaultdict(int)
    total_pixels = 0
    
    for r in results:
        pixel_counts = r.get('class_pixel_counts', {})
        for class_name, count in pixel_counts.items():
            class_pixel_counts[class_name] += count
            total_pixels += count
    
    default_classes = [
        "whiteboard", "beam", "column", "bookcase", "door", "window",
        "table", "chair", "sofa", "wall", "floor", "ceiling"
    ]
    
    print(f"\n   📊 Computing Class-Balanced Weights from PIXEL COUNTS (β={beta})")
    print("="*80)
    print(f"   {'Class':<15} {'Pixels':<15} {'Ratio':<10} {'1-β^n':<15} {'Base Weight':<12} {'Final':<10}")
    print("-"*80)
    
    weights = {}
    for class_name in default_classes:
        pixel_count = class_pixel_counts.get(class_name, 0)
        beta_pow_n = beta ** pixel_count
        if 1 - beta_pow_n < 1e-6:
            effective_num = 1e-6
        else:
            effective_num = 1 - beta_pow_n
        base_weight = (1 - beta) / effective_num
        boost = boost_factor.get(class_name, 1.0) if boost_factor else 1.0
        final_weight = base_weight * boost
        weights[class_name] = final_weight
        
        ratio = pixel_count / total_pixels if total_pixels > 0 else 0
        print(f"   {class_name:<15} {pixel_count:<15,} {ratio:<10.4f} {(1-beta_pow_n):<15.6f} {base_weight:<12.4f} {final_weight:<10.4f}")
    
    print("-"*80)
    
    max_weight = max(weights.values())
    target_max = 10.0
    for class_name in weights:
        weights[class_name] = weights[class_name] / max_weight * target_max
    
    print(f"\n   📊 Normalized weights (max={target_max}):")
    for class_name, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        pixel_count = class_pixel_counts.get(class_name, 0)
        print(f"      {class_name:<15} pixels={pixel_count:>12,} → weight={weight:.4f}")
    
    return weights, class_pixel_counts, total_pixels

def load_class_mapping_with_balanced_weights(mapping_path, full_summary_path, beta=0.999, boost_factor=None):
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r') as f:
            raw_mapping = json.load(f)
            id2label = {int(k): v for k, v in raw_mapping.items()}
            label2id = {v: int(k) for k, v in raw_mapping.items()}
        print(f"   ✅ Loaded mapping with {len(id2label)} classes")
    else:
        print(f"   ⚠️ Mapping file not found, using default 12 classes")
        default_classes = [
            "whiteboard", "beam", "column", "bookcase", "door", "window",
            "table", "chair", "sofa", "wall", "floor", "ceiling"
        ]
        id2label = {i: name for i, name in enumerate(default_classes)}
        label2id = {name: i for i, name in enumerate(default_classes)}
    
    class_weights, class_pixel_counts, total_pixels = compute_class_balanced_weights_from_pixels(
        full_summary_path, beta, boost_factor
    )
    
    if class_weights:
        id_weights = {label2id[name]: weight for name, weight in class_weights.items() if name in label2id}
    else:
        id_weights = {cid: 1.0 for cid in id2label.keys()}
        class_weights = {name: 1.0 for name in id2label.values()}
    
    return id2label, label2id, id_weights, class_weights

# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint.get('processed_items', [])
    return []

def save_checkpoint(processed_items):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'processed_items': processed_items}, f, indent=2)

# ============================================================
# SCANNING FUNCTIONS
# ============================================================

def find_all_items(pcd_root, capture_root, masks_root):
    items = []
    
    for root, dirs, files in os.walk(capture_root):
        png_files = [f for f in files if f.endswith('.png') and 'view_' in f]
        if not png_files:
            continue
        
        rel_path = os.path.relpath(root, capture_root)
        is_segment = 'segment_' in rel_path or 'seg_' in rel_path
        
        if is_segment:
            parent_room = rel_path.split('\\')[0] if '\\' in rel_path else rel_path.replace('_seg', '')
            pcd_path = os.path.join(pcd_root, parent_room)
        else:
            pcd_path = os.path.join(pcd_root, rel_path)
        
        if not os.path.exists(pcd_path):
            print(f"   ⚠️ PCD not found: {pcd_path}")
            continue
        
        masks_path = os.path.join(masks_root, rel_path, "masks")
        if not os.path.exists(masks_path):
            alt_masks = os.path.join(masks_root, rel_path.replace('\\', '_'), "masks")
            if os.path.exists(alt_masks):
                masks_path = alt_masks
            else:
                print(f"   ⚠️ Masks not found: {rel_path}")
                continue
        
        conf_path = os.path.join(masks_root, rel_path, "confidences")
        has_conf = os.path.exists(conf_path)
        
        items.append({
            'name': rel_path.replace('\\', '_').replace('/', '_'),
            'display_name': rel_path,
            'type': 'segment' if is_segment else 'room',
            'pcd_path': pcd_path,
            'capture_path': root,
            'masks_path': masks_path,
            'conf_path': conf_path if has_conf else None,
            'num_images': len(png_files)
        })
    
    return sorted(items, key=lambda x: x['name'])

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_point_cloud(folder_path):
    all_points = []

    txt_files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith('.txt')
    ])

    if not txt_files:
        return None

    for file in tqdm(txt_files, desc="   Loading files", leave=False):
        file_path = os.path.join(folder_path, file)
        data = np.loadtxt(file_path)

        # XYZRGB
        if data.shape[1] >= 6:
            all_points.append(data[:, :6])
        else:
            print(f"⚠️ {file} thiếu RGB")
            all_points.append(data[:, :3])

    return np.vstack(all_points)

def load_camera_poses(captures_folder):
    poses_file = os.path.join(captures_folder, "camera_poses.json")
    if not os.path.exists(poses_file):
        return None
    
    with open(poses_file, 'r') as f:
        camera_poses = json.load(f)
    
    if 'camera_poses' in camera_poses:
        camera_poses = camera_poses['camera_poses']
    
    return camera_poses

def load_masks(masks_folder):
    mask_files = sorted([f for f in os.listdir(masks_folder) if f.endswith('.png') and 'mask_' in f])
    if not mask_files:
        return None
    return mask_files

def load_confidence_map(conf_folder, mask_file):
    if conf_folder is None:
        return None
    
    conf_file = mask_file.replace('mask_', 'conf_').replace('.png', '.npy')
    conf_path = os.path.join(conf_folder, conf_file)
    
    if os.path.exists(conf_path):
        return np.load(conf_path)
    return None

def project_point_with_matrix(point_3d, extrinsic, intrinsic):
    point_hom = np.append(point_3d, 1.0)
    point_cam = extrinsic @ point_hom
    
    if point_cam[2] <= 0.01:
        return None
    
    point_img = intrinsic @ point_cam[:3]
    
    if point_img[2] == 0:
        return None
    
    px = int(point_img[0] / point_img[2])
    py = int(point_img[1] / point_img[2])
    depth = point_cam[2]
    
    return (px, py, depth)

# ============================================================
# SOFT VOTING VỚI Z-BUFFER VÀ CONFIDENCE
# ============================================================

def soft_voting_with_zbuffer_and_confidence(
    points, camera_poses, mask_files, masks_folder, conf_folder,
    id_weights, depth_weight_factor=0.1, depth_tolerance=0.05,
    min_visible_views=2
):
    point_weights = [defaultdict(float) for _ in range(len(points))]
    point_view_counts = np.zeros(len(points))
    
    total_projections = 0
    successful_projections = 0
    filtered_by_occlusion = 0
    
    for pose_data in tqdm(camera_poses, desc="   Processing views", leave=False):
        view_idx = pose_data.get('view_idx', pose_data.get('index', 0))
        if view_idx >= len(mask_files):
            continue
        
        mask_path = os.path.join(masks_folder, mask_files[view_idx])
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        conf_map = load_confidence_map(conf_folder, mask_files[view_idx])
        
        intrinsic = np.array(pose_data['intrinsic_matrix'])
        extrinsic = np.array(pose_data['extrinsic_matrix'])
        
        img_h, img_w = mask_array.shape
        
        depth_buffer = np.full((img_h, img_w), np.inf)
        projections = []
        
        for point_idx, point_3d in enumerate(points):
            result = project_point_with_matrix(point_3d, extrinsic, intrinsic)
            if result is None:
                continue
            
            px, py, depth = result
            total_projections += 1
            
            if 0 <= px < img_w and 0 <= py < img_h:
                projections.append((point_idx, px, py, depth))
                if depth < depth_buffer[py, px]:
                    depth_buffer[py, px] = depth
        
        for point_idx, px, py, depth in projections:
            if abs(depth - depth_buffer[py, px]) > depth_tolerance:
                filtered_by_occlusion += 1
                continue
            
            successful_projections += 1
            
            class_id = int(mask_array[py, px])
            if class_id == 255 or class_id not in id_weights:
                continue
            
            base_weight = 1.0 / (depth + depth_weight_factor)
            class_weight = id_weights.get(class_id, 1.0)
            
            mask_conf = 1.0
            if conf_map is not None and 0 <= py < conf_map.shape[0] and 0 <= px < conf_map.shape[1]:
                mask_conf = conf_map[py, px]
            
            final_weight = base_weight * class_weight * mask_conf
            
            point_weights[point_idx][class_id] += final_weight
            point_view_counts[point_idx] += 1
    
    print(f"\n   📊 Projection stats:")
    print(f"      Total projections: {total_projections:,}")
    print(f"      Successful (in image): {successful_projections:,} ({100*successful_projections/total_projections:.1f}%)")
    print(f"      Filtered by occlusion: {filtered_by_occlusion:,}")
    
    return point_weights, point_view_counts

# ============================================================
# ASSIGN LABELS
# ============================================================

def assign_labels_from_weights(point_weights, point_view_counts, min_total_weight=0.5, min_visible_views=2):
    point_labels = np.full(len(point_weights), -1, dtype=np.int32)
    point_confidences = np.zeros(len(point_weights))
    
    for point_idx in tqdm(range(len(point_weights)), desc="   Assigning labels", leave=False):
        if point_view_counts[point_idx] < min_visible_views:
            continue
        
        if not point_weights[point_idx]:
            continue
        
        total_weight = sum(point_weights[point_idx].values())
        
        if total_weight < min_total_weight:
            continue
        
        max_class = max(point_weights[point_idx].items(), key=lambda x: x[1])
        
        point_labels[point_idx] = max_class[0]
        point_confidences[point_idx] = max_class[1] / total_weight
    
    return point_labels, point_confidences

# ============================================================
# kNN PROPAGATION
# ============================================================

def knn_propagation_with_threshold(
    points, point_labels, point_confidences, id_weights,
    k=7, confidence_threshold=0.3, propagation_threshold=0.6
):
    labeled_mask = point_labels >= 0
    unlabeled_mask = point_labels < 0
    unlabeled_count = np.sum(unlabeled_mask)
    
    if unlabeled_count == 0 or np.sum(labeled_mask) == 0:
        return point_labels, point_confidences
    
    high_conf_mask = (point_labels >= 0) & (point_confidences >= confidence_threshold)
    high_conf_count = np.sum(high_conf_mask)
    
    if high_conf_count < 5:
        high_conf_mask = labeled_mask
        high_conf_count = np.sum(high_conf_mask)
    
    labeled_points = points[high_conf_mask]
    labeled_labels = point_labels[high_conf_mask]
    labeled_confidences = point_confidences[high_conf_mask]
    
    tree = KDTree(labeled_points)
    unlabeled_indices = np.where(unlabeled_mask)[0]
    
    propagated_count = 0
    
    for idx in tqdm(unlabeled_indices, desc="   Propagating", leave=False):
        dist, ind = tree.query(points[idx].reshape(1, -1), k=min(k, len(labeled_points)))
        
        neighbor_labels = labeled_labels[ind[0]]
        neighbor_confidences = labeled_confidences[ind[0]]
        distance_weights = 1.0 / (dist[0] + 0.01)
        
        label_scores = {}
        for label, dist_w, conf in zip(neighbor_labels, distance_weights, neighbor_confidences):
            if label < 0:
                continue
            class_weight = id_weights.get(label, 1.0)
            label_scores[label] = label_scores.get(label, 0) + dist_w * conf * class_weight
        
        if not label_scores:
            continue
        
        best_label = max(label_scores, key=label_scores.get)
        best_score = label_scores[best_label]
        total_score = sum(label_scores.values())
        
        propagation_conf = best_score / total_score if total_score > 0 else 0
        
        if propagation_conf >= propagation_threshold:
            point_labels[idx] = best_label
            point_confidences[idx] = propagation_conf
            propagated_count += 1
    
    print(f"\n   📊 Propagated: {propagated_count}/{unlabeled_count} points ({100*propagated_count/unlabeled_count:.1f}%)")
    
    return point_labels, point_confidences

# ============================================================
# ✅ SAVE RESULTS (FIXED: LƯU ĐẦY ĐỦ FILE)
# ============================================================

def save_results(item_name, points, point_labels, point_confidences, id2label, output_root):
    output_folder = os.path.join(output_root, item_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # ✅ 1. Save labels (đã có)
    np.save(os.path.join(output_folder, f"{item_name}_labels.npy"), point_labels)
    
    # ✅ 2. Save confidences (đã có)
    np.save(os.path.join(output_folder, f"{item_name}_confidences.npy"), point_confidences)
    
    # ✅ 3. THÊM: Save points coordinates (QUAN TRỌNG CHO EVALUATION)
    np.save(os.path.join(output_folder, f"{item_name}_points.npy"), points)
    
    # ✅ 4. THÊM: Save class mapping riêng cho item này (để evaluation dễ dàng)
    class_mapping = {int(k): v for k, v in id2label.items()}
    with open(os.path.join(output_folder, f"{item_name}_class_mapping.json"), 'w') as f:
        json.dump(class_mapping, f, indent=2)
    
    # ✅ 5. Save metadata
    metadata = {
        'name': item_name,
        'total_points': len(points),
        'labeled_points': int(np.sum(point_labels >= 0)),
        'coverage': float(100 * np.sum(point_labels >= 0) / len(points)),
        'avg_confidence': float(np.mean(point_confidences[point_labels >= 0])) if np.sum(point_labels >= 0) > 0 else 0,
        'classes_present': [id2label.get(int(l), 'unknown') for l in np.unique(point_labels) if l >= 0]
    }
    with open(os.path.join(output_folder, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # ✅ 6. Save confidence heatmap PLY
    conf_colors = np.zeros((len(points), 3))
    for i in range(len(points)):
        if point_labels[i] >= 0:
            conf = point_confidences[i]
            conf_colors[i] = [1-conf, conf, 0]
        else:
            conf_colors[i] = [0.5, 0.5, 0.5]
    
    pcd_conf = o3d.geometry.PointCloud()
    pcd_conf.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_conf.colors = o3d.utility.Vector3dVector(conf_colors)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{item_name}_confidence.ply"), pcd_conf)
    
    # ✅ 7. Save semantic PLY
    sem_colors = np.zeros((len(points), 3))
    for i, label in enumerate(point_labels):
        if label >= 0 and label in id2label:
            class_name = id2label[label]
            sem_colors[i] = CLASS_COLORS_CONSISTENT.get(class_name, [0.5, 0.5, 0.5])
        else:
            sem_colors[i] = [0.3, 0.3, 0.3]
    
    pcd_sem = o3d.geometry.PointCloud()
    pcd_sem.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd_sem.colors = o3d.utility.Vector3dVector(sem_colors)
    o3d.io.write_point_cloud(os.path.join(output_folder, f"{item_name}_semantic.ply"), pcd_sem)
    
    return metadata

# ============================================================
# MAIN FUNCTION
# ============================================================

def main():
    print("="*70)
    print("🔍 3D SEMANTIC MAPPING - Z-BUFFER + CONFIDENCE + PIXEL WEIGHTS")
    print("="*70)
    
    print(f"\n📊 Config:")
    print(f"   Beta (class-balanced): {BETA}")
    print(f"   kNN neighbors: {K_NEIGHBORS}")
    print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"   Propagation threshold: {PROPAGATION_CONFIDENCE_THRESHOLD}")
    print(f"   Min visible views: {MIN_VISIBLE_VIEWS}")
    print(f"   Min total weight: {MIN_TOTAL_WEIGHT}")
    print(f"   Depth tolerance: {DEPTH_TOLERANCE}")
    
    # Load class mapping with weights từ pixel counts
    print("\n📋 Loading class mapping...")
    id2label, label2id, id_weights, class_weights = load_class_mapping_with_balanced_weights(
        MAPPING_PATH, FULL_SUMMARY_PATH, beta=BETA, boost_factor=BOOST_FACTOR
    )
    
    # ✅ Save global class mapping
    global_mapping_path = os.path.join(OUTPUT_ROOT, "pcd_class_mapping.json")
    with open(global_mapping_path, 'w') as f:
        json.dump(id2label, f, indent=2)
    print(f"   💾 Saved global mapping to {global_mapping_path}")
    
    # Find all items
    print("\n📁 Scanning for rooms and segments...")
    items = find_all_items(FOLDER_PCD_INPUT, FOLDER_CAPTURES, FOLDER_MASKS_ROOT)
    
    rooms = [i for i in items if i['type'] == 'room']
    segments = [i for i in items if i['type'] == 'segment']
    print(f"   Found {len(rooms)} rooms, {len(segments)} segments")
    
    # Load checkpoint
    processed = load_checkpoint()
    print(f"   Already processed: {len(processed)} items")
    
    all_results = []
    
    for idx, item in enumerate(items):
        item_name = item['name']  # ✅ Thêm dòng này
        item_type = item['type']
        if item['name'] in processed:
            print(f"\n⏭️  Skip {item['display_name']} [{item['type']}] (already done)")
            meta_path = os.path.join(OUTPUT_ROOT, item['name'], "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    all_results.append(json.load(f))
            continue
        
        print("\n" + "="*70)
        print(f"[{idx+1}/{len(items)}] Processing: {item['display_name']} [{item['type']}]")
        print(f"   Images: {item['num_images']}")
        print("="*70)
        
        try:
            # Load point cloud
            points = load_point_cloud(item['pcd_path'])
            if points is None:
                print(f"   ❌ No PCD files found")
                continue

            xyz = points[:, :3]
            rgb = points[:, 3:6] if points.shape[1] >= 6 else None
            
            # Load camera poses
            camera_poses = load_camera_poses(item['capture_path'])
            if camera_poses is None:
                print(f"   ❌ No camera poses found")
                continue
            print(f"   Views: {len(camera_poses)}")
            
            # Load masks
            mask_files = load_masks(item['masks_path'])
            if mask_files is None:
                print(f"   ❌ No masks found")
                continue
            
            # Soft voting với z-buffer
            point_weights, point_view_counts = soft_voting_with_zbuffer_and_confidence(
                xyz, camera_poses, mask_files, item['masks_path'], item['conf_path'],
                id_weights, DEPTH_WEIGHT_FACTOR, DEPTH_TOLERANCE, MIN_VISIBLE_VIEWS
            )
            
            # Assign labels
            point_labels, point_confidences = assign_labels_from_weights(
                point_weights, point_view_counts, MIN_TOTAL_WEIGHT, MIN_VISIBLE_VIEWS
            )
            
            initial_labeled = np.sum(point_labels >= 0)
            print(f"\n   📊 After voting: {initial_labeled:,}/{len(points):,} labeled ({100*initial_labeled/len(points):.1f}%)")
            
            # kNN propagation
            point_labels, point_confidences = knn_propagation_with_threshold(
                xyz, point_labels, point_confidences, id_weights,
                K_NEIGHBORS, CONFIDENCE_THRESHOLD, PROPAGATION_CONFIDENCE_THRESHOLD
            )
            
            # ✅ Save results (đã được sửa để lưu đầy đủ)
            metadata = save_results(item['name'], points, point_labels, point_confidences, id2label, OUTPUT_ROOT)
            all_results.append(metadata)
            
            # Update checkpoint
            processed.append(item['name'])
            save_checkpoint(processed)
            
            final_labeled = np.sum(point_labels >= 0)
            print(f"\n   ✅ Done: {final_labeled:,}/{len(points):,} ({100*final_labeled/len(points):.1f}%)")
            print(f"   Avg confidence: {metadata['avg_confidence']:.3f}")
            print(f"   💾 Saved: {item_name}_labels.npy, {item_name}_points.npy, {item_name}_confidence.ply, {item_name}_semantic.ply")
            
        except Exception as e:
            print(f"\n   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*70)
    print("📊 FINAL SUMMARY")
    print("="*70)
    
    if all_results:
        # ✅ Sửa cách phân loại rooms vs segments
        rooms_done = [r for r in all_results if 'segment' not in r.get('name', '')]
        segs_done = [r for r in all_results if 'segment' in r.get('name', '')]
        print(f"   Rooms done: {len(rooms_done)}")
        print(f"   Segments done: {len(segs_done)}")
        if rooms_done:
            print(f"   Avg coverage (rooms): {np.mean([r['coverage'] for r in rooms_done]):.1f}%")
            print(f"   Avg confidence (rooms): {np.mean([r['avg_confidence'] for r in rooms_done]):.3f}")
    
    # Save summary
    with open(os.path.join(OUTPUT_ROOT, "mapping_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Results saved to: {OUTPUT_ROOT}")
    print("="*70)

if __name__ == "__main__":
    main()