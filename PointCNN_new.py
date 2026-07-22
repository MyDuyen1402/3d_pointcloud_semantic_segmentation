# ============================================================
# POINTCNN REFINEMENT WITH VALIDATION - LAZY LOADING
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
# Đặt global cho tất cả torch.load
import torch.serialization
torch.serialization.add_safe_globals([np.ndarray, np.dtype, np.generic])

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

# Import PointCNN from local model
from pointcnn_model import get_model

# ============================================================
# CONFIGURATION - S3DIS MULTI-AREA SETUP
# ============================================================

# ROOT_DIR should point to parent folder containing all Area folders
# Structure: ROOT_DIR/Area_1/office_1/*_points.npy
#                    /Area_2/office_1/*_points.npy
#                    /Area_3/...
#                    /...
#                    /Area_6/...
# The code will AUTOMATICALLY scan and find all areas/rooms recursively!

# ROOT_DIR = r"E:\\MyDuyen\\KLTN\\mapped_pcd"  # ← CHANGE THIS to your mapped_pcd folder
# GT_ROOT = r"E:\\MyDuyen\\KLTN\\Stanford3dDataset_v1.2_Aligned_Version"  # ← CHANGE THIS (optional, leave empty if no GT)
# OUTPUT_DIR = r"E:\\MyDuyen\\KLTN\\results\\pointcnn_refinement"
ROOT_DIR = "/compute_home/slurmdang12/datasets/mapped_pcd"
GT_ROOT = "/compute_home/slurmdang12/datasets/Stanford3dDataset_v1.2_Aligned_Version"
OUTPUT_DIR = "/compute_home/slurmdang12/datasets/pointcnn_refinement"

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparameters
NUM_CLASSES = 12  # S3DIS: ceiling, floor, wall, beam, column, window, door, table, chair, sofa, bookcase, board
NUM_POINTS = 8192  # Points per sample
BATCH_SIZE = 2  # Batch size
EPOCHS = 150  # Total training epochs
LEARNING_RATE = 1e-4  # Adam optimizer learning rate
K_NEIGHBORS = 16  # For density feature computation
SAVE_EVERY = 5  # Save checkpoint every N epochs
VAL_SPLIT_RATIO = 0.2  # 20% of rooms for validation, 80% for training
RANDOM_SEED = 42  # For reproducibility
NUM_WORKERS = 4  # DataLoader worker count - fixed at 4 for stability

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
if DEVICE == "cuda":
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Detected GPU total memory: {gpu_mem_gb:.1f} GB")
    if gpu_mem_gb >= 80:
        BATCH_SIZE = max(BATCH_SIZE, 4)
        NUM_POINTS = max(NUM_POINTS, 12288)
    elif gpu_mem_gb >= 48:
        BATCH_SIZE = max(BATCH_SIZE, 4)
        NUM_POINTS = max(NUM_POINTS, 10240)
    elif gpu_mem_gb >= 24:
        BATCH_SIZE = max(BATCH_SIZE, 4)
    print(f"Auto-adjusted NUM_POINTS={NUM_POINTS}, BATCH_SIZE={BATCH_SIZE}")

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
# HELPER FUNCTIONS FOR EXACT COORDINATE MATCHING
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
# DATASET WITH LAZY LOADING AND CACHE
# ============================================================

class RefinementDataset(Dataset):
    def __init__(self, root_dir, num_points=8192, k_neighbors=16, 
                 split='train', val_ratio=0.2, seed=42):
        self.root_dir = Path(root_dir)
        self.num_points = num_points
        self.k_neighbors = k_neighbors
        self.split = split
        
        self.gt_cache = {}
        self.data_cache = {}
        all_samples = []
        
        print(f"Scanning {root_dir} for samples...")
        # Recursively find all *_points.npy files across ALL Area/Room folders
        # This automatically discovers all 6 areas and their rooms!
        for points_file in self.root_dir.rglob("*_points.npy"):
            base = points_file.stem.replace("_points", "")
            labels_file = points_file.parent / f"{base}_labels.npy"
            conf_file = points_file.parent / f"{base}_confidences.npy"
            
            # Check that all 3 required files exist (points, labels, confidences)
            if labels_file.exists() and conf_file.exists():
                area_name = points_file.parent.parent.name  # e.g., "Area_1", "Area_2"
                room_name = points_file.parent.name  # e.g., "office_1", "conferenceRoom_1"
                
                all_samples.append({
                    "points": points_file,
                    "labels": labels_file,
                    "conf": conf_file,
                    "area": area_name,
                    "room": room_name,
                    "cache_key": f"{area_name}/{room_name}"
                })
        
        np.random.seed(seed)
        indices = np.random.permutation(len(all_samples))
        val_size = int(len(all_samples) * val_ratio)
        
        # Automatic train/val split:
        # E.g., if found 24 rooms: 19 for training, 5 for validation (80/20 split)
        if split == 'train':
            self.samples = [all_samples[i] for i in indices[val_size:]]  # 80% rooms
        else:
            self.samples = [all_samples[i] for i in indices[:val_size]]  # 20% rooms
        
        print(f"\n{'='*50}")
        print(f"SPLIT: {split}")
        print(f"Total rooms: {len(all_samples)}")
        print(f"Rooms in this split: {len(self.samples)}")
        print(f"{'='*50}\n")
    
    def __len__(self):
        return len(self.samples)
    
    def compute_density(self, xyz):
        # Use batch processing for memory efficiency
        if len(xyz) > 4096:
            # Process in chunks if too many points
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
        
        print(f"    Loaded {len(gt_xyz)} labeled points from {base_room}")
        
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
            print(f"  Loading GT for {cache_key}...")
            gt_xyz, gt_labels = self.load_room_gt_labels(area_name, room_name)
            self.gt_cache[cache_key] = (gt_xyz, gt_labels)
        else:
            gt_xyz, gt_labels = self.gt_cache[cache_key]
        
        if gt_xyz is None:
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        
        matched_indices, valid_mask = find_matching_indices(points, gt_xyz, tolerance=1e-4)
        
        if not np.any(valid_mask):
            return self.__getitem__(np.random.randint(0, len(self.samples)))
        
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
# MODEL - PointCNN for Semantic Segmentation
# ============================================================

class PointCNNRefinement(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.model = get_model(num_classes)
    
    def forward(self, x):
        # Input: [B, N, 22]
        # PointCNN internally handles permutation and outputs [B, C, N]
        pred = self.model(x)
        # Output: [B, 12, N] - ready for CrossEntropyLoss
        return pred

# ============================================================
# EVALUATION FUNCTIONS
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

def evaluate(model, val_loader, criterion, device, num_classes=12):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                
                pred = model(features)  # [B, C, N]
                
                # pred is already [B, C, N] for loss
                loss = criterion(pred, labels)
                total_loss += loss.item()
                
                # Get predictions: argmax over class dimension (dim=1)
                pred_labels = torch.argmax(pred, dim=1)  # [B, N]
                
                pred_flat = pred_labels.reshape(-1)
                labels_flat = labels.reshape(-1)
                
                valid_mask = labels_flat != -1
                if valid_mask.any():
                    all_preds.append(pred_flat[valid_mask].cpu().numpy())
                    all_labels.append(labels_flat[valid_mask].cpu().numpy())
                
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n⚠️  GPU OUT OF MEMORY during validation!")
            print("Try reducing: NUM_POINTS, BATCH_SIZE, or K_NEIGHBORS")
            if device == "cuda":
                torch.cuda.empty_cache()
            raise
        else:
            raise
    
    if len(all_preds) == 0:
        return {
            'loss': total_loss / len(val_loader) if len(val_loader) > 0 else 0,
            'accuracy': 0.0,
            'mean_iou': 0.0,
            'ious': [float('nan')] * num_classes
        }
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    accuracy = accuracy_score(all_labels, all_preds)
    ious = compute_iou(all_preds, all_labels, num_classes)
    mean_iou = np.nanmean(ious)
    
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'ious': ious
    }

# ============================================================
# CREATE DATASETS AND DATALOADERS
# ============================================================

print("\n" + "="*60)
print("CREATING DATASETS")
print("="*60)

train_dataset = RefinementDataset(
    ROOT_DIR,
    num_points=NUM_POINTS,
    k_neighbors=K_NEIGHBORS,
    split='train',
    val_ratio=VAL_SPLIT_RATIO,
    seed=RANDOM_SEED
)

val_dataset = RefinementDataset(
    ROOT_DIR,
    num_points=NUM_POINTS,
    k_neighbors=K_NEIGHBORS,
    split='val',
    val_ratio=VAL_SPLIT_RATIO,
    seed=RANDOM_SEED
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=False
)

print(f"\nTrain batches per epoch: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")

# ============================================================
# MODEL, OPTIMIZER, CRITERION
# ============================================================

print("\n" + "="*60)
print("INITIALIZING MODEL")
print("="*60)

model = PointCNNRefinement(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
criterion = nn.CrossEntropyLoss(ignore_index=-1)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================
# RESUME FROM CHECKPOINT
# ============================================================

start_epoch = 0
best_val_iou = 0
latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")

if os.path.exists(latest_ckpt):
    print("\nLoading checkpoint...")
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_iou = checkpoint.get("best_val_iou", 0)
    
    print(f"Resumed from epoch {start_epoch}")
    print(f"Best validation mIoU so far: {best_val_iou:.4f}")

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

for epoch in range(start_epoch, EPOCHS):
    # ========== TRAINING PHASE ==========
    model.train()
    total_train_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        features = batch["features"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        pred = model(features)  # [B, C, N]
        loss = criterion(pred, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] "
                  f"Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
    
    # ========== SAVE CHECKPOINT ==========
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_iou": best_val_iou,
        "train_loss": avg_train_loss,
    }
    
    torch.save(checkpoint, latest_ckpt)
    print(f"  💾 Saved checkpoint")
    
    # ========== VALIDATION PHASE ==========
    if len(val_loader) > 0:
        print(f"\nEpoch {epoch+1} - Validating...")
        
        # Memory check before validation
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            mem_allocated = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  GPU Memory - Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")
        
        try:
            val_metrics = evaluate(model, val_loader, criterion, DEVICE, NUM_CLASSES)
            
            scheduler.step(val_metrics['loss'])
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1} SUMMARY")
            print(f"{'='*60}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val mIoU: {val_metrics['mean_iou']:.4f}")
            print(f"Learning rate: {current_lr:.8f}")
            
            print("\nPer-class IoU on Validation Set:")
            for i, iou in enumerate(val_metrics['ious']):
                if not np.isnan(iou):
                    print(f"  {CLASSES_12[i]:12s}: {iou:.4f}")
                else:
                    print(f"  {CLASSES_12[i]:12s}: N/A")
            
            checkpoint["val_loss"] = val_metrics['loss']
            checkpoint["val_miou"] = val_metrics['mean_iou']
            
            if val_metrics['mean_iou'] > best_val_iou:
                best_val_iou = val_metrics['mean_iou']
                checkpoint["best_val_iou"] = best_val_iou
                best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
                torch.save(checkpoint, best_model_path)
                print(f"\n🎉 New best model! mIoU: {best_val_iou:.4f}")
            
            torch.save(checkpoint, latest_ckpt)
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
    else:
        print(f"\nEpoch {epoch+1} - No validation data, skipping validation...")
    
    if (epoch + 1) % SAVE_EVERY == 0:
        save_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    print(f"{'='*60}\n")

# ============================================================
# FINAL EVALUATION
# ============================================================

print("\n" + "="*60)
print("TRAINING COMPLETED")
print("="*60)

best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
if os.path.exists(best_model_path) and len(val_loader) > 0:
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Best model was from epoch {checkpoint['epoch']+1} with mIoU: {checkpoint['val_miou']:.4f}")

if len(val_loader) > 0:
    print("\nRunning final evaluation on validation set...")
    final_metrics = evaluate(model, val_loader, criterion, DEVICE, NUM_CLASSES)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Validation Loss: {final_metrics['loss']:.4f}")
    print(f"Validation Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Validation Mean IoU: {final_metrics['mean_iou']:.4f}")
    print("\nFinal Per-class IoU:")
    for i, iou in enumerate(final_metrics['ious']):
        if not np.isnan(iou):
            print(f"  {CLASSES_12[i]:12s}: {iou:.4f}")
    
    results_file = os.path.join(LOG_DIR, "validation_results.txt")
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("POINTCNN REFINEMENT - FINAL RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Validation Loss: {final_metrics['loss']:.4f}\n")
        f.write(f"Validation Accuracy: {final_metrics['accuracy']:.4f}\n")
        f.write(f"Validation Mean IoU: {final_metrics['mean_iou']:.4f}\n\n")
        f.write("Per-class IoU:\n")
        for i, iou in enumerate(final_metrics['ious']):
            if not np.isnan(iou):
                f.write(f"  {CLASSES_12[i]:12s}: {iou:.4f}\n")
        f.write("\n" + "="*60 + "\n")
        f.write(f"Best validation mIoU during training: {best_val_iou:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")
else:
    print("\nNo validation data available for final evaluation.")

# Save final model
final_model_path = os.path.join(OUTPUT_DIR, "pointcnn_refinement_final.pth")
torch.save(model.state_dict(), final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print("\n✅ Training and validation completed successfully!")