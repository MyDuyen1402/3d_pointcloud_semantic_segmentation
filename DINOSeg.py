# 1. Cài đặt thư viện (chạy 1 lần)
# !pip install groundingdino-py sam2 torch torchvision opencv-python pillow matplotlib numpy tqdm

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import os
from pathlib import Path
import json
from tqdm import tqdm
from torchvision.ops import box_convert, nms
from collections import Counter
import sys

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN (THAY ĐỔI THEO MÁY CỦA BẠN)
# ============================================================
GROUNDING_DINO_PATH = r"D:\Thu\KhoaLuan\GroundingDINO"
GROUNDING_DINO_CONFIG = os.path.join(GROUNDING_DINO_PATH, "groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT = r"D:\Thu\KhoaLuan\GroundingDINO\weights\groundingdino_swint_ogc.pth"

SAM2_CHECKPOINT = r"D:\Thu\KhoaLuan\sam2\checkpoints\sam2_hiera_large.pt"
SAM2_CONFIG = r"D:\Thu\KhoaLuan\sam2\sam2\configs\sam2\sam2_hiera_l.yaml"

# Thêm path Grounding DINO
sys.path.insert(0, GROUNDING_DINO_PATH)

# Import Grounding DINO
from groundingdino.util.inference import load_model, load_image, predict

# ============================================================
# CẤU HÌNH XỬ LÝ
# ============================================================
capture_root = r"D:\Thu\KhoaLuan\captures\Standford\Area_5"
output_root = r"D:\Thu\KhoaLuan\results\Standford_grounding_sam2\Area_5"
os.makedirs(output_root, exist_ok=True)

# 12 classes cần detect
CLASSES_12 = [
    "whiteboard", "beam", "column", "bookcase", "door", "window",
    "table", "chair", "sofa", "wall", "floor", "ceiling"
]

# Priority cho mask merging (ưu tiên small objects cao hơn)
PRIORITY_MAP = {
    "beam": 120,      # structural - cao nhất
    "column": 115,
    "window": 110,
    "door": 105,
    "whiteboard": 100,
    "chair": 95,
    "table": 90,
    "sofa": 85,
    "bookcase": 80,
    "wall": 30,
    "floor": 20,
    "ceiling": 10
}

TEXT_PROMPT = " . ".join(CLASSES_12)

# Thresholds (đã tăng lên để giảm noise)
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.30

# ============================================================
# HIỂN THỊ CẤU HÌNH
# ============================================================
print("="*70)
print("🔍 GROUNDING DINO + SAM 2 - ZERO-SHOT SEMANTIC SEGMENTATION")
print("="*70)
print(f"Capture root: {capture_root}")
print(f"Output root: {output_root}")
print(f"Classes: {CLASSES_12}")
print(f"Text prompt: {TEXT_PROMPT}")
print(f"Box threshold: {BOX_THRESHOLD}, Text threshold: {TEXT_THRESHOLD}")
print("="*70)

# ============================================================
# TÌM TẤT CẢ CÁC THƯ MỤC CHỨA ẢNH
# ============================================================
print("\n📁 Scanning for images in capture folders...")

image_folders = []
room_info = []

for root, dirs, files in os.walk(capture_root):
    png_files = [f for f in files if f.lower().endswith('.png') and 'view_' in f.lower()]
    if png_files:
        rel_path = os.path.relpath(root, capture_root)
        image_folders.append(root)
        
        is_segment = 'segment_' in rel_path or 'seg_' in rel_path
        
        poses_file = os.path.join(root, "camera_poses.json")
        num_views = len(png_files)
        room_name = rel_path.replace('\\', '_').replace('/', '_')
        
        if os.path.exists(poses_file):
            try:
                with open(poses_file, 'r') as f:
                    poses_data = json.load(f)
                    num_views = poses_data.get('num_views', len(png_files))
                    room_name = poses_data.get('room_name', room_name)
            except:
                pass
        
        room_info.append({
            'folder': root,
            'room_name': room_name,
            'num_images': num_views,
            'rel_path': rel_path,
            'is_segment': is_segment
        })
        
        segment_tag = " [SEGMENT]" if is_segment else ""
        print(f"   📂 Found: {rel_path}{segment_tag} -> {num_views} images")

print(f"\n✅ Found {len(image_folders)} image folders")
print(f"   Total images to process: {sum(r['num_images'] for r in room_info)}")

# ============================================================
# KIỂM TRA FILE MODEL
# ============================================================
print("\n📁 Checking model files...")

if not os.path.exists(GROUNDING_DINO_CONFIG):
    print(f"❌ Config not found: {GROUNDING_DINO_CONFIG}")
    sys.exit()

if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
    print(f"❌ Checkpoint not found: {GROUNDING_DINO_CHECKPOINT}")
    sys.exit()

if not os.path.exists(SAM2_CHECKPOINT):
    print(f"❌ SAM2 checkpoint not found: {SAM2_CHECKPOINT}")
    sys.exit()

# ============================================================
# LOAD GROUNDING DINO
# ============================================================
print("\n📦 Loading Grounding DINO model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")

model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=device
)
model.eval()
print("✅ Grounding DINO loaded!")

# ============================================================
# LOAD SAM 2
# ============================================================
print("\n📦 Loading SAM 2 model...")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)
sam_predictor = SAM2ImagePredictor(sam2_model)
print("✅ SAM 2 loaded!")

# ============================================================
# CLASS MAPPING & COLORS
# ============================================================
class_to_id = {name: idx for idx, name in enumerate(CLASSES_12)}
id_to_class = {idx: name for idx, name in enumerate(CLASSES_12)}

CLASS_COLORS = {
    'ceiling': [200, 200, 200],
    'floor': [139, 69, 19],
    'wall': [200, 180, 150],
    'beam': [255, 0, 0],       # Màu đỏ nổi bật
    'column': [150, 150, 150],
    'window': [135, 206, 235],
    'door': [205, 133, 63],
    'table': [255, 165, 0],
    'chair': [0, 200, 0],
    'sofa': [255, 105, 180],
    'bookcase': [160, 100, 60],
    'whiteboard': [100, 100, 200],
}

# ============================================================
# HÀM FILL UNLABELED TỪ BOUNDING BOX (CÓ ĐIỀU KIỆN)
# ============================================================
def fill_unlabeled_from_boxes(
    semantic_mask,
    boxes_xyxy,
    labels,
    scores,
    class_to_id,
    h,
    w,
    min_score=0.35
):
    """
    Fill unlabeled pixels only inside confident boxes
    Chỉ fill nếu unlabeled chiếm dưới 60% diện tích box
    """
    filled_count = 0
    
    for box, label, score in zip(boxes_xyxy, labels, scores):
        # Bỏ qua box có score thấp
        if score < min_score:
            continue
        
        if label not in class_to_id:
            continue
        
        class_id = class_to_id[label]
        x1, y1, x2, y2 = map(int, box)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            continue
        
        region = semantic_mask[y1:y2, x1:x2]
        unlabeled = (region == -1)
        
        # Chỉ fill nếu unlabeled không quá 60% box
        if unlabeled.size > 0:
            ratio = np.sum(unlabeled) / unlabeled.size
            if ratio < 0.6:
                region[unlabeled] = class_id
                filled_count += np.sum(unlabeled)
    
    return semantic_mask, filled_count

# ============================================================
# HÀM DỰ ĐOÁN CHÍNH (GROUNDING DINO + SAM)
# ============================================================
def predict_grounding_sam(image_path, text_prompt, box_threshold, text_threshold):
    """
    Run Grounding DINO + SAM on a single image
    Returns: (image_source, masks_list, labels_list, scores_list, boxes_list, (h, w))
    """
    # Load image
    image_source, image = load_image(image_path)
    h, w = image_source.shape[:2]
    
    # Grounding DINO prediction
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    
    if len(boxes) == 0:
        return image_source, [], [], [], [], (h, w)
    
    # Convert boxes from cxcywh normalized to xyxy absolute
    boxes_xyxy_norm = box_convert(boxes.cpu(), in_fmt="cxcywh", out_fmt="xyxy")
    boxes_xyxy = boxes_xyxy_norm * torch.tensor([w, h, w, h])
    
    # ✅ NMS để loại bỏ box chồng nhau
    keep = nms(boxes_xyxy, logits, iou_threshold=0.4)
    boxes_xyxy = boxes_xyxy[keep]
    logits = logits[keep]
    phrases = [phrases[i] for i in keep]
    
    # SAM prediction
    sam_predictor.set_image(image_source)
    
    all_masks = []
    all_labels = []
    all_scores = []
    all_boxes = []
    
    for box, logit, phrase in zip(boxes_xyxy, logits, phrases):
        box_np = box.cpu().numpy()
        
        # SAM với multimask_output=False cho kết quả nhất quán
        sam_masks, sam_scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box_np[None, :],
            multimask_output=False,
        )
        
        if sam_masks.shape[0] > 0:
            single_mask = sam_masks[0]
            single_score = sam_scores[0]
            
            # ✅ Area filtering - bỏ mask quá nhỏ hoặc quá lớn
            mask_area = np.sum(single_mask)
            total_area = h * w
            
            if mask_area < 200:      # quá nhỏ
                continue
            if mask_area > total_area * 0.7:  # quá lớn
                continue
            
            # Xác định class từ phrase
            phrase_lower = phrase.lower()
            detected_class = None
            
            # Ưu tiên các class đặc biệt
            if "beam" in phrase_lower:
                detected_class = "beam"
            elif "whiteboard" in phrase_lower:
                detected_class = "whiteboard"
            elif "column" in phrase_lower or "pillar" in phrase_lower:
                detected_class = "column"
            elif "bookcase" in phrase_lower or "bookshelf" in phrase_lower:
                detected_class = "bookcase"
            else:
                for class_name in CLASSES_12:
                    if class_name.lower() in phrase_lower:
                        detected_class = class_name
                        break
            
            if detected_class is not None:
                all_masks.append(single_mask)
                all_labels.append(detected_class)
                all_scores.append(float(logit * single_score))
                all_boxes.append(box_np)
    
    return image_source, all_masks, all_labels, all_scores, all_boxes, (h, w)

# ============================================================
# HÀM TẠO SEMANTIC MASK TỪ CÁC MASKS
# ============================================================
def create_semantic_mask(masks, labels, scores, h, w):
    """
    Combine multiple masks into single semantic mask với priority
    """
    semantic_mask = np.full((h, w), -1, dtype=np.int32)
    confidence_map = np.zeros((h, w))
    
    if len(masks) == 0:
        return semantic_mask, confidence_map
    
    # Tạo danh sách các mask với priority và score
    items = []
    for mask, label, score in zip(masks, labels, scores):
        priority = PRIORITY_MAP.get(label, 50)
        items.append((mask, label, score, priority))
    
    # Sort theo priority giảm dần, score giảm dần
    items.sort(key=lambda x: (x[3], x[2]), reverse=True)
    
    # Gán mask theo thứ tự ưu tiên
    for mask, label, score, priority in items:
        class_id = class_to_id[label]
        mask_indices = (mask == True) & (semantic_mask == -1)
        semantic_mask[mask_indices] = class_id
        confidence_map[mask_indices] = score
    
    return semantic_mask, confidence_map

# ============================================================
# XỬ LÝ TỪNG ROOM/SEGMENT
# ============================================================
all_results_summary = []
total_images_processed = 0

print("\n" + "="*70)
print("🚀 STARTING PROCESSING")
print("="*70)

for room_idx, room in enumerate(tqdm(room_info, desc="Processing rooms/segments")):
    room_folder = room['folder']
    room_name = room['room_name']
    num_images = room['num_images']
    rel_path = room['rel_path']
    is_segment = room['is_segment']
    
    segment_tag = " [SEGMENT]" if is_segment else ""
    print(f"\n{'='*60}")
    print(f"[{room_idx+1}/{len(room_info)}] Processing: {room_name}{segment_tag}")
    print(f"   Path: {rel_path}")
    print(f"   Images: {num_images}")
    print(f"{'='*60}")
    
    # Tạo output folder cho room này
    room_output_dir = os.path.join(output_root, rel_path)
    os.makedirs(room_output_dir, exist_ok=True)
    os.makedirs(os.path.join(room_output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(room_output_dir, "visualizations"), exist_ok=True)
    
    # Lấy tất cả ảnh trong room folder
    image_files = sorted([f for f in os.listdir(room_folder) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                          and 'view_' in f.lower()])
    
    room_results = []
    
    for img_file in tqdm(image_files, desc=f"   {room_name}", leave=False):
        image_path = os.path.join(room_folder, img_file)
        
        try:
            # Run Grounding DINO + SAM
            image_source, masks_list, labels_list, scores_list, boxes_list, img_size = predict_grounding_sam(
                image_path, TEXT_PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD
            )
            
            h, w = img_size
            
            if len(masks_list) == 0:
                # Không có detection nào - để nguyên unlabeled
                semantic_mask = np.full((h, w), -1, dtype=np.int32)
                confidence_map = np.zeros((h, w))
            else:
                semantic_mask, confidence_map = create_semantic_mask(masks_list, labels_list, scores_list, h, w)
            
            # Fill unlabeled pixels từ bounding box (có điều kiện)
            if len(boxes_list) > 0:
                semantic_mask, filled_count = fill_unlabeled_from_boxes(
                    semantic_mask, boxes_list, labels_list, scores_list, class_to_id, h, w, min_score=0.35
                )
            
            # ✅ QUAN TRỌNG: KHÔNG fill toàn ảnh bằng wall/floor/ceiling
            # Unlabeled pixels giữ nguyên giá trị -1
            
            # Lưu mask (unlabeled = 255 để phân biệt)
            mask_filename = f"mask_{Path(img_file).stem}.png"
            mask_path = os.path.join(room_output_dir, "masks", mask_filename)
            semantic_to_save = semantic_mask.copy()
            semantic_to_save[semantic_to_save == -1] = 255
            mask_image = Image.fromarray(semantic_to_save.astype(np.uint8))
            mask_image.save(mask_path)
            
            # Thống kê
            beam_count = np.sum(semantic_mask == class_to_id["beam"])
            wall_count = np.sum(semantic_mask == class_to_id["wall"])
            unlabeled_count = np.sum(semantic_mask == -1)
            
            # ============================================================
            # VISUALIZATION
            # ============================================================
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # 1. Original image + bounding boxes
            axes[0].imshow(image_source)
            for mask, label, score in zip(masks_list, labels_list, scores_list):
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    color = 'red' if label == 'beam' else 'green'
                    rect = plt.Rectangle((x, y), bw, bh, fill=False, edgecolor=color, linewidth=2)
                    axes[0].add_patch(rect)
                    axes[0].text(x, y-5, f"{label} ({score:.2f})", color=color, fontsize=8,
                               bbox=dict(facecolor='white', alpha=0.7))
            axes[0].set_title(f"Detections: {len(masks_list)} objects", fontweight='bold')
            axes[0].axis('off')
            
            # 2. Semantic mask (unlabeled = black)
            colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
            for class_name, class_id in class_to_id.items():
                colored_mask[semantic_mask == class_id] = CLASS_COLORS[class_name]
            colored_mask[semantic_mask == -1] = [0, 0, 0]  # unlabeled = black
            
            axes[1].imshow(colored_mask)
            axes[1].set_title(f"Semantic Mask\nBeam: {beam_count} px | Unlabeled: {unlabeled_count} px", fontweight='bold')
            axes[1].axis('off')
            
            # 3. Overlay
            overlay = image_source * 0.5 + colored_mask * 0.5
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title("Overlay", fontweight='bold')
            axes[2].axis('off')
            
            # Legend
            legend_patches = []
            for class_name in CLASSES_12:
                patch = mpatches.Patch(color=np.array(CLASS_COLORS[class_name])/255.0, label=class_name)
                legend_patches.append(patch)
            legend_patches.append(mpatches.Patch(color='black', label='unlabeled'))
            
            fig.legend(handles=legend_patches, loc='center right', 
                      bbox_to_anchor=(0.98, 0.5), title="CLASSES", 
                      title_fontsize='12', fontsize='9')
            
            plt.tight_layout(rect=[0, 0, 0.92, 1])
            viz_path = os.path.join(room_output_dir, "visualizations", f"result_{Path(img_file).stem}.png")
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Lưu kết quả
            room_results.append({
                'image': img_file,
                'is_segment': is_segment,
                'num_objects': len(masks_list),
                'beam_pixels': int(beam_count),
                'wall_pixels': int(wall_count),
                'unlabeled_pixels': int(unlabeled_count),
                'classes_detected': list(set(labels_list)) if labels_list else []
            })
            
            total_images_processed += 1
            
        except Exception as e:
            print(f"\n   ❌ Error on {img_file}: {e}")
            continue
    
    # Lưu kết quả cho room
    with open(os.path.join(room_output_dir, "detection_summary.json"), 'w') as f:
        json.dump(room_results, f, indent=2)
    
    all_results_summary.extend([{**r, 'room': room_name, 'rel_path': rel_path} for r in room_results])
    
    print(f"\n   ✅ Completed {room_name}: {len(room_results)}/{num_images} images processed")

# ============================================================
# TỔNG KẾT TOÀN BỘ
# ============================================================
print("\n" + "="*70)
print("📊 FINAL SUMMARY")
print("="*70)
print(f"Total rooms/segments processed: {len(room_info)}")
print(f"  - Normal rooms: {sum(1 for r in room_info if not r['is_segment'])}")
print(f"  - Segments: {sum(1 for r in room_info if r['is_segment'])}")
print(f"Total images processed: {total_images_processed}")

# Lưu tổng kết toàn bộ
summary_path = os.path.join(output_root, "full_detection_summary.json")
with open(summary_path, 'w') as f:
    json.dump(all_results_summary, f, indent=2)

# Tạo report tổng hợp
report = os.path.join(output_root, "full_report.txt")
with open(report, 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("GROUNDING DINO + SAM - FIXED VERSION REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("CONFIGURATION:\n")
    f.write(f"  Capture root: {capture_root}\n")
    f.write(f"  Output root: {output_root}\n")
    f.write(f"  Classes: {', '.join(CLASSES_12)}\n")
    f.write(f"  Text prompt: {TEXT_PROMPT}\n")
    f.write(f"  Box threshold: {BOX_THRESHOLD}\n")
    f.write(f"  Text threshold: {TEXT_THRESHOLD}\n\n")
    
    f.write("CHANGES APPLIED:\n")
    f.write("  1. Removed global wall/floor/ceiling fill\n")
    f.write("  2. Added NMS for bounding boxes\n")
    f.write("  3. Increased thresholds (0.35/0.30)\n")
    f.write("  4. Added area filtering for masks\n")
    f.write("  5. Changed to multimask_output=False\n")
    f.write("  6. Updated priority map for small objects\n")
    f.write("  7. Fill unlabeled only inside confident boxes (ratio < 0.6)\n\n")
    
    f.write("PROCESSING RESULTS:\n")
    f.write(f"  Number of rooms/segments: {len(room_info)}\n")
    f.write(f"  Total images processed: {total_images_processed}\n\n")
    
    # Thống kê theo class
    all_detected = []
    total_beam_pixels = 0
    total_unlabeled_pixels = 0
    
    for r in all_results_summary:
        all_detected.extend(r.get('classes_detected', []))
        total_beam_pixels += r.get('beam_pixels', 0)
        total_unlabeled_pixels += r.get('unlabeled_pixels', 0)
    
    class_counts = Counter(all_detected)
    
    f.write("DETECTION STATISTICS:\n")
    for class_name in CLASSES_12:
        count = class_counts.get(class_name, 0)
        if total_images_processed > 0:
            f.write(f"  {class_name:12s}: detected in {count:3d}/{total_images_processed} images ({100*count/total_images_processed:.1f}%)\n")
        else:
            f.write(f"  {class_name:12s}: detected in {count} images\n")
    
    f.write(f"\nPIXEL STATISTICS:\n")
    f.write(f"  Total beam pixels: {total_beam_pixels}\n")
    f.write(f"  Total unlabeled pixels: {total_unlabeled_pixels}\n")
    if total_beam_pixels + total_unlabeled_pixels > 0:
        f.write(f"  Unlabeled ratio: {100*total_unlabeled_pixels/(total_beam_pixels+total_unlabeled_pixels):.2f}%\n")

print(f"\n✅ Report saved: {report}")
print(f"✅ Full summary saved: {summary_path}")
print(f"\n📂 All results saved to: {output_root}")
print("="*70)