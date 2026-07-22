# ============================================================
# TEST POINTCNN FINAL MODEL ON AREA 5 (S3DIS)
# ============================================================

import sys
import os
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import torch.serialization
torch.serialization.add_safe_globals([np.ndarray, np.dtype, np.generic])

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json

# Import PointCNN from local model
from pointcnn_model import get_model

# ============================================================
# CONFIGURATION - Auto-detect paths
# ============================================================

# Try multiple path patterns
def find_data_root():
    """Auto-detect data root directory"""
    candidates = [
        "/compute_home/slurmdang12/datasets/mapped_pcd",
        "/home/slurmdang12/datasets/mapped_pcd",
        os.path.expanduser("~/datasets/mapped_pcd"),
        "./datasets/mapped_pcd",
        "../datasets/mapped_pcd",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

def find_model_path():
    """Auto-detect final model path"""
    candidates = [
        "/compute_home/slurmdang12/datasets/pointcnn_refinement/pointcnn_refinement_final.pth",
        "/home/slurmdang12/datasets/pointcnn_refinement/pointcnn_refinement_final.pth",
        os.path.expanduser("~/datasets/pointcnn_refinement/pointcnn_refinement_final.pth"),
        "./datasets/pointcnn_refinement/pointcnn_refinement_final.pth",
        "../datasets/pointcnn_refinement/pointcnn_refinement_final.pth",
        "./pointcnn_refinement_final.pth",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None

ROOT_DIR = find_data_root()
if not ROOT_DIR:
    print("\n❌ ERROR: Could not find mapped_pcd directory!")
    print("Expected in one of:")
    print("  - /compute_home/slurmdang12/datasets/mapped_pcd")
    print("  - /home/slurmdang12/datasets/mapped_pcd")
    print("\nPlease check paths and try again.")
    sys.exit(1)

print(f"✓ Found data root: {ROOT_DIR}")

GT_ROOT = os.path.join(os.path.dirname(ROOT_DIR), "Stanford3dDataset_v1.2_Aligned_Version")
if not os.path.exists(GT_ROOT):
    gt_candidates = [
        "/compute_home/slurmdang12/datasets/Stanford3dDataset_v1.2_Aligned_Version",
        "/home/slurmdang12/datasets/Stanford3dDataset_v1.2_Aligned_Version",
    ]
    for path in gt_candidates:
        if os.path.exists(path):
            GT_ROOT = path
            break

print(f"✓ Found GT root: {GT_ROOT}")

FINAL_MODEL_PATH = find_model_path()
if not FINAL_MODEL_PATH:
    print("\n❌ ERROR: Could not find final model!")
    print("Expected in one of:")
    print("  - /compute_home/slurmdang12/datasets/pointcnn_refinement/pointcnn_refinement_final.pth")
    print("  - /home/slurmdang12/datasets/pointcnn_refinement/pointcnn_refinement_final.pth")
    print("\nPlease check paths and try again.")
    sys.exit(1)

print(f"✓ Found model: {FINAL_MODEL_PATH}")

OUTPUT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "pointcnn_refinement")
TEST_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "test_results")

os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# Hyperparameters
NUM_CLASSES = 12
NUM_POINTS = 8192
BATCH_SIZE = 2
K_NEIGHBORS = 16
RANDOM_SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Detected GPU total memory: {gpu_mem_gb:.1f} GB")
    if gpu_mem_gb >= 80:
        NUM_POINTS = max(NUM_POINTS, 12288)
    elif gpu_mem_gb >= 48:
        NUM_POINTS = max(NUM_POINTS, 10240)

# ============================================================
# CLASSES MAPPING
# ============================================================

CLASSES_12 = [
    "whiteboard", "beam", "column", "bookcase",
    "door", "window", "table", "chair",
    "sofa", "wall", "floor", "ceiling"
]

CLASS_MAP = {
    "board": 0, "whiteboard": 0,
    "beam": 1, "column": 2, "bookcase": 3,
    "door": 4, "window": 5, "table": 6,
    "chair": 7, "sofa": 8, "wall": 9,
    "floor": 10, "ceiling": 11,
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_point_index(points, tolerance=1e-4):
    point_dict = {}
    rounded_points = np.round(points[:, :3] / tolerance) * tolerance
    
    for idx, point_key in enumerate(rounded_points):
        key = tuple(point_key)
        if key not in point_dict:
            point_dict[key] = idx
    
    return point_dict, rounded_points

def find_matching_indices(points, gt_xyz, tolerance=1e-4):
    point_dict, rounded_points = build_point_index(points, tolerance)
    gt_dict, gt_rounded = build_point_index(gt_xyz, tolerance)
    
    matched_indices = np.full(len(points), -1)
    
    for i, point_key in enumerate(rounded_points):
        key = tuple(point_key)
        if key in gt_dict:
            matched_indices[i] = gt_dict[key]
    
    valid_mask = matched_indices != -1
    return matched_indices, valid_mask

# ============================================================
# TEST DATASET - ONLY AREA 5
# ============================================================

class TestDataset(Dataset):
    def __init__(self, root_dir, num_points=8192, k_neighbors=16, test_area='Area_5', seed=42):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.k_neighbors = k_neighbors
        self.test_area = test_area
        
        self.gt_cache = {}
        self.data_cache = {}
        self.samples = []
        
        print(f"\n{'='*80}")
        print(f"SCANNING FOR TEST DATASET - {test_area}")
        print(f"{'='*80}")
        print(f"Root directory: {root_dir}")
        
        # Find all samples in Area_5 only
        area_path = self.root_dir / test_area
        if not area_path.exists():
            print(f"ERROR: {area_path} does not exist!")
            return
        
        print(f"\nScanning: {area_path}")
        
        # First, list all rooms in Area_5
        rooms_found = {}
        all_points_files = list(area_path.rglob("*_points.npy"))
        print(f"\nTotal _points.npy files found: {len(all_points_files)}")
        
        if len(all_points_files) > 0:
            print("\nDetailed scan of each room:")
            for points_file in sorted(all_points_files):
                room_path = points_file.parent
                room_name = room_path.name
                base = points_file.stem.replace("_points", "")
                
                labels_file = room_path / f"{base}_labels.npy"
                conf_file = room_path / f"{base}_confidences.npy"
                
                # Track what files exist in each room
                if room_name not in rooms_found:
                    rooms_found[room_name] = {"points": False, "labels": False, "conf": False}
                
                rooms_found[room_name]["points"] = points_file.exists()
                rooms_found[room_name]["labels"] = labels_file.exists()
                rooms_found[room_name]["conf"] = conf_file.exists()
                
                # Only add if all 3 files exist
                if labels_file.exists() and conf_file.exists():
                    area_name = room_path.parent.name
                    
                    self.samples.append({
                        "points": points_file,
                        "labels": labels_file,
                        "conf": conf_file,
                        "area": area_name,
                        "room": room_name,
                        "cache_key": f"{area_name}/{room_name}"
                    })
                    print(f"  ✓ {room_name:<30} [points: {points_file.exists()}, labels: {labels_file.exists()}, conf: {conf_file.exists()}]")
                else:
                    missing = []
                    if not labels_file.exists():
                        missing.append("labels")
                    if not conf_file.exists():
                        missing.append("conf")
                    print(f"  ✗ {room_name:<30} [MISSING: {', '.join(missing)}]")
        
        print(f"\n{'='*80}")
        print(f"SCAN COMPLETE")
        print(f"{'='*80}")
        print(f"Total rooms in {test_area}: {len(rooms_found)}")
        print(f"Valid samples (with all 3 files): {len(self.samples)}")
        
        if len(self.samples) == 0:
            print(f"\n⚠️  WARNING: No valid samples found in {test_area}!")
            print("Rooms without required files:")
            for room_name, files in rooms_found.items():
                if not (files['points'] and files['labels'] and files['conf']):
                    print(f"  - {room_name}: points={files['points']}, labels={files['labels']}, conf={files['conf']}")
        else:
            print(f"\n✓ Successfully loaded {len(self.samples)} samples:")
            for sample in self.samples:
                print(f"  - {sample['room']}")
    
    def __len__(self):
        return len(self.samples)
    
    def compute_density(self, xyz):
        if len(xyz) > 4096:
            chunk_size = 4096
            densities = []
            for i in range(0, len(xyz), chunk_size):
                chunk = xyz[i:i+chunk_size]
                nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(chunk))).fit(chunk)
                distances, _ = nbrs.kneighbors(chunk)
                density = 1.0 / (np.mean(distances, axis=1) + 1e-6)
                densities.append(density)
            return np.concatenate(densities)
        else:
            nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(xyz)
            distances, _ = nbrs.kneighbors(xyz)
            density = 1.0 / (np.mean(distances, axis=1) + 1e-6)
            return density
    
    def load_room_gt_labels(self, area_name, room_name):
        if "_segment_" in room_name:
            base_room = room_name.split("_segment_")[0]
        else:
            base_room = room_name
        
        anno_dir = Path(GT_ROOT) / area_name / base_room / "Annotations"
        
        if not anno_dir.exists():
            print(f"    WARNING: Annotations folder not found: {anno_dir}")
            return None, None
        
        all_gt_xyz = []
        all_gt_labels = []
        
        for txt_file in anno_dir.glob("*.txt"):
            class_name = txt_file.stem.split("_")[0]
            if class_name not in CLASS_MAP:
                continue
            
            label_id = CLASS_MAP[class_name]
            
            try:
                object_data = np.loadtxt(txt_file)
            except:
                continue
            
            if len(object_data.shape) != 2 or object_data.shape[1] < 3:
                continue
            
            object_xyz = object_data[:, :3]
            object_labels = np.full(len(object_xyz), label_id)
            
            all_gt_xyz.append(object_xyz)
            all_gt_labels.append(object_labels)
        
        if len(all_gt_xyz) == 0:
            print(f"    WARNING: No valid annotation files in {anno_dir}")
            return None, None
        
        gt_xyz = np.concatenate(all_gt_xyz, axis=0)
        gt_labels = np.concatenate(all_gt_labels, axis=0)
        
        return gt_xyz, gt_labels
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        sample_key = (str(sample["points"]), str(sample["labels"]), str(sample["conf"]))
        if sample_key not in self.data_cache:
            self.data_cache[sample_key] = {
                "points": np.load(sample["points"], mmap_mode="r"),
                "pred_labels": np.load(sample["labels"], mmap_mode="r"),
                "confidences": np.load(sample["conf"], mmap_mode="r")
            }

        points = self.data_cache[sample_key]["points"]
        pred_labels = self.data_cache[sample_key]["pred_labels"]
        confidences = self.data_cache[sample_key]["confidences"]
        
        room_name = sample["room"]
        area_name = sample["area"]
        cache_key = sample["cache_key"]
        
        if cache_key not in self.gt_cache:
            gt_xyz, gt_labels = self.load_room_gt_labels(area_name, room_name)
            self.gt_cache[cache_key] = (gt_xyz, gt_labels)
        else:
            gt_xyz, gt_labels = self.gt_cache[cache_key]
        
        if gt_xyz is None:
            # Return dummy data to avoid errors
            dummy_features = np.random.randn(self.num_points, 22).astype(np.float32)
            dummy_labels = np.full(self.num_points, -1, dtype=np.int64)
            return {
                "features": torch.FloatTensor(dummy_features),
                "labels": torch.LongTensor(dummy_labels),
                "room_name": room_name,
                "area_name": area_name
            }
        
        matched_indices, valid_mask = find_matching_indices(points, gt_xyz, tolerance=1e-4)
        
        if not np.any(valid_mask):
            dummy_features = np.random.randn(self.num_points, 22).astype(np.float32)
            dummy_labels = np.full(self.num_points, -1, dtype=np.int64)
            return {
                "features": torch.FloatTensor(dummy_features),
                "labels": torch.LongTensor(dummy_labels),
                "room_name": room_name,
                "area_name": area_name
            }
        
        points_filtered = points[valid_mask]
        pred_labels_filtered = pred_labels[valid_mask]
        confidences_filtered = confidences[valid_mask]
        
        gt_indices = matched_indices[valid_mask]
        gt_labels_filtered = gt_labels[gt_indices]
        
        N = len(points_filtered)
        
        pred_onehot = np.eye(NUM_CLASSES)[pred_labels_filtered]
        conf = confidences_filtered[:, None]
        entropy = 1.0 - confidences_filtered
        entropy = entropy[:, None]
        
        z = points_filtered[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)
        z_norm = z_norm[:, None]
        
        density = self.compute_density(points_filtered[:, :3])
        density = density[:, None]
        
        features = np.concatenate([
            points_filtered[:, :3],
            points_filtered[:, 3:6],
            pred_onehot,
            conf,
            entropy,
            z_norm,
            density
        ], axis=1)
        
        if N >= self.num_points:
            idxs = np.random.choice(N, self.num_points, replace=False)
        else:
            idxs = np.random.choice(N, self.num_points, replace=True)
        
        features = features[idxs]
        gt_labels_filtered = gt_labels_filtered[idxs]
        
        gt_labels_filtered = gt_labels_filtered.astype(np.int64)
        gt_labels_filtered[gt_labels_filtered < 0] = -1
        gt_labels_filtered[gt_labels_filtered >= NUM_CLASSES] = -1
        
        return {
            "features": torch.FloatTensor(features),
            "labels": torch.LongTensor(gt_labels_filtered),
            "room_name": room_name,
            "area_name": area_name
        }

# ============================================================
# MODEL
# ============================================================

class PointCNNRefinement(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.model = get_model(num_classes)
    
    def forward(self, x):
        pred = self.model(x)
        return pred

# ============================================================
# METRICS COMPUTATION
# ============================================================

def compute_iou(pred_labels, true_labels, num_classes):
    ious = []
    for class_id in range(num_classes):
        pred_mask = (pred_labels == class_id)
        true_mask = (true_labels == class_id)
        
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious

def compute_precision(pred_labels, true_labels, num_classes):
    precisions = []
    for class_id in range(num_classes):
        pred_mask = (pred_labels == class_id)
        true_mask = (true_labels == class_id)
        
        tp = np.logical_and(pred_mask, true_mask).sum()
        fp = np.logical_and(pred_mask, ~true_mask).sum()
        
        if (tp + fp) == 0:
            precisions.append(float('nan'))
        else:
            precisions.append(tp / (tp + fp))
    return precisions

def compute_recall(pred_labels, true_labels, num_classes):
    recalls = []
    for class_id in range(num_classes):
        pred_mask = (pred_labels == class_id)
        true_mask = (true_labels == class_id)
        
        tp = np.logical_and(pred_mask, true_mask).sum()
        fn = np.logical_and(~pred_mask, true_mask).sum()
        
        if (tp + fn) == 0:
            recalls.append(float('nan'))
        else:
            recalls.append(tp / (tp + fn))
    return recalls

def compute_f1(precisions, recalls):
    f1_scores = []
    for prec, recall in zip(precisions, recalls):
        if np.isnan(prec) or np.isnan(recall):
            f1_scores.append(float('nan'))
        elif (prec + recall) == 0:
            f1_scores.append(float('nan'))
        else:
            f1_scores.append(2 * (prec * recall) / (prec + recall))
    return f1_scores

# ============================================================
# TESTING FUNCTION
# ============================================================

def test(model, test_loader, criterion, device, num_classes=12):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    print("\n" + "="*80)
    print("TESTING ON AREA 5")
    print("="*80 + "\n")
    
    total_batches = len(test_loader)
    print(f"Total batches to process: {total_batches}\n")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_num = batch_idx + 1
            print(f"[{batch_num}/{total_batches}] Processing batch...", end=" ")
            
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            
            print(f"(features shape: {features.shape}, labels shape: {labels.shape})", end=" ")
            
            pred = model(features)  # [B, C, N]
            loss = criterion(pred, labels)
            total_loss += loss.item()
            
            pred_labels = torch.argmax(pred, dim=1)  # [B, N]
            
            pred_flat = pred_labels.reshape(-1)
            labels_flat = labels.reshape(-1)
            
            valid_mask = labels_flat != -1
            num_valid = valid_mask.sum().item()
            print(f"valid_points: {num_valid}, loss: {loss.item():.4f}")
            
            if valid_mask.any():
                all_preds.append(pred_flat[valid_mask].cpu().numpy())
                all_labels.append(labels_flat[valid_mask].cpu().numpy())
    
    total_points = sum(len(p) for p in all_preds)
    print(f"\n✓ Processed {total_points} total valid points across all batches")
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    print(f"\nComputing metrics...")
    print(f"  - Accuracy...", end=" ", flush=True)
    accuracy = accuracy_score(all_labels, all_preds)
    print("✓")
    
    print(f"  - IoU...", end=" ", flush=True)
    ious = compute_iou(all_preds, all_labels, num_classes)
    mean_iou = np.nanmean(ious)
    print("✓")
    
    print(f"  - Precision...", end=" ", flush=True)
    precisions = compute_precision(all_preds, all_labels, num_classes)
    print("✓")
    
    print(f"  - Recall...", end=" ", flush=True)
    recalls = compute_recall(all_preds, all_labels, num_classes)
    print("✓")
    
    print(f"  - F1-Score...", end=" ", flush=True)
    f1_scores = compute_f1(precisions, recalls)
    print("✓")
    
    mean_precision = np.nanmean(precisions)
    mean_recall = np.nanmean(recalls)
    mean_f1 = np.nanmean(f1_scores)
    
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'ious': ious,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'all_preds': all_preds,
        'all_labels': all_labels
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LOADING TEST DATASET (AREA 5)")
    print("="*80)
    
    test_dataset = TestDataset(
        ROOT_DIR,
        num_points=NUM_POINTS,
        k_neighbors=K_NEIGHBORS,
        test_area='Area_5',
        seed=RANDOM_SEED
    )
    
    if len(test_dataset) == 0:
        print("\n❌ ERROR: No test data found in Area 5!")
        print("Please check:")
        print(f"  1. Directory path: {ROOT_DIR}")
        print(f"  2. Area folder: Area_5")
        print(f"  3. Required files: *_points.npy, *_labels.npy, *_confidences.npy")
        sys.exit(1)
    
    print(f"\n✓ Dataset loaded successfully with {len(test_dataset)} samples")
    
    print(f"\n{'='*80}")
    print(f"CREATING DATALOADER")
    print(f"{'='*80}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Num workers: 4")
    print(f"Pin memory: True")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"Total batches: {len(test_loader)}")
    print(f"✓ DataLoader created successfully")
    
    print(f"\n{'='*80}")
    print(f"LOADING TRAINED MODEL")
    print(f"{'='*80}")
    print(f"Model path: {FINAL_MODEL_PATH}")
    print(f"Model class: PointCNNRefinement")
    print(f"Num classes: {NUM_CLASSES}")
    
    if not os.path.exists(FINAL_MODEL_PATH):
        print(f"\n❌ ERROR: Model file not found at {FINAL_MODEL_PATH}")
        sys.exit(1)
    
    print(f"File size: {os.path.getsize(FINAL_MODEL_PATH) / (1024**2):.2f} MB")
    
    print(f"\nInitializing model...", end=" ", flush=True)
    model = PointCNNRefinement(num_classes=NUM_CLASSES).to(DEVICE)
    print("✓")
    
    print(f"Loading checkpoint...", end=" ", flush=True)
    try:
        checkpoint = torch.load(FINAL_MODEL_PATH, map_location=DEVICE, weights_only=False)
        
        # Handle multiple checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Format 1: Wrapped checkpoint with metadata
                model.load_state_dict(checkpoint["model_state_dict"])
                checkpoint_info = checkpoint
            elif "state_dict" in checkpoint:
                # Format 2: Alternative wrapper
                model.load_state_dict(checkpoint["state_dict"])
                checkpoint_info = checkpoint
            else:
                # Format 3: Direct state dict (likely from model.state_dict())
                model.load_state_dict(checkpoint)
                checkpoint_info = {}
        else:
            # Shouldn't happen, but handle it
            print("\n❌ ERROR: Unexpected checkpoint format")
            sys.exit(1)
        
        print("✓")
        
        if checkpoint_info and "epoch" in checkpoint_info:
            print(f"Model info:")
            print(f"  - Trained epoch: {checkpoint_info['epoch'] + 1}")
            if "val_miou" in checkpoint_info:
                print(f"  - Best validation mIoU: {checkpoint_info['val_miou']:.4f}")
        else:
            print(f"Model info:")
            print(f"  - (No epoch info in checkpoint)")
    
    except KeyError as e:
        print(f"\n❌ ERROR: Checkpoint format issue - {e}")
        print(f"Available keys in checkpoint: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dict'}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load checkpoint - {e}")
        sys.exit(1)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    print(f"\n✓ Model loaded successfully to {DEVICE}")
    
    # Test
    print(f"\n{'='*80}")
    print(f"STARTING TEST EVALUATION")
    print(f"{'='*80}")
    test_metrics = test(model, test_loader, criterion, DEVICE, NUM_CLASSES)
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS - AREA 5")
    print("="*80)
    
    print(f"\nOverall Metrics:")
    print(f"  Loss:          {test_metrics['loss']:.4f}")
    print(f"  Accuracy:      {test_metrics['accuracy']:.4f}")
    print(f"  mIoU:          {test_metrics['mean_iou']:.4f}")
    print(f"  mPrecision:    {test_metrics['mean_precision']:.4f}")
    print(f"  mRecall:       {test_metrics['mean_recall']:.4f}")
    print(f"  mF1-Score:     {test_metrics['mean_f1']:.4f}")
    
    print(f"\n{'Class':<15} {'IoU':>8} {'Precision':>10} {'Recall':>8} {'F1-Score':>10}")
    print("-" * 60)
    for i, class_name in enumerate(CLASSES_12):
        iou = test_metrics['ious'][i]
        prec = test_metrics['precisions'][i]
        rec = test_metrics['recalls'][i]
        f1 = test_metrics['f1_scores'][i]
        
        iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
        prec_str = f"{prec:.4f}" if not np.isnan(prec) else "N/A"
        rec_str = f"{rec:.4f}" if not np.isnan(rec) else "N/A"
        f1_str = f"{f1:.4f}" if not np.isnan(f1) else "N/A"
        
        print(f"{class_name:<15} {iou_str:>8} {prec_str:>10} {rec_str:>8} {f1_str:>10}")
    
    # Save detailed results to JSON
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    
    results_json = {
        "overall": {
            "loss": float(test_metrics['loss']),
            "accuracy": float(test_metrics['accuracy']),
            "mean_iou": float(test_metrics['mean_iou']),
            "mean_precision": float(test_metrics['mean_precision']),
            "mean_recall": float(test_metrics['mean_recall']),
            "mean_f1": float(test_metrics['mean_f1']),
            "total_samples": len(test_metrics['all_labels'])
        },
        "per_class": {}
    }
    
    for i, class_name in enumerate(CLASSES_12):
        results_json["per_class"][class_name] = {
            "iou": float(test_metrics['ious'][i]) if not np.isnan(test_metrics['ious'][i]) else None,
            "precision": float(test_metrics['precisions'][i]) if not np.isnan(test_metrics['precisions'][i]) else None,
            "recall": float(test_metrics['recalls'][i]) if not np.isnan(test_metrics['recalls'][i]) else None,
            "f1_score": float(test_metrics['f1_scores'][i]) if not np.isnan(test_metrics['f1_scores'][i]) else None
        }
    
    results_path = os.path.join(TEST_OUTPUT_DIR, "test_results_area5.json")
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Saving JSON results...", end=" ", flush=True)
    print("✓")
    print(f"Output path: {results_path}")
    
    print("\n" + "="*80)
    print("✓ TESTING COMPLETE")
    print("="*80 + "\n")
