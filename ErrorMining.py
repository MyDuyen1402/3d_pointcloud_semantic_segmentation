"""
=====================================================================
GIAI ĐOẠN 7 (THỰC TẾ): VERIFY MLP PREDICTION + ERROR MINING
=====================================================================

[ĐÃ SỬA 3 LỖI LOGIC]
1. Bật Mahalanobis làm lưới an toàn cho window/whiteboard/table
2. Lớp 1 ceiling/floor dùng RANSAC plane-distance thay vì |z deviation|
3. Mở rộng Z theo lỗ hổng trên tường (wall gap extension)
=====================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import warnings
from scipy.stats import chi2
from sklearn.cluster import DBSCAN

# =====================================================
# PATHS - chỉnh lại cho đúng máy của bạn
# =====================================================

#PLY_PATH = Path(r"D:\Thu\KhoaLuan\conferenceRoom_1_mlp_refined.ply")
PROTOTYPE_PATH = Path("/compute_home/slurmdang12/datasets/Prototype/joint_prototype.pkl")
OUTPUT_DIR = Path("/compute_home/slurmdang12/datasets/ErrorMining")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# CHECKPOINT CONFIG
# =====================================================
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_INTERVAL = 1  # Lưu checkpoint sau mỗi N room
RESUME_FROM_CHECKPOINT = True  # Có tiếp tục từ checkpoint không

# Bảng màu phải KHỚP CHÍNH XÁC với bảng màu dùng khi xuất PLY từ MLP
CLASS_COLORS_CONSISTENT = {
    'whiteboard': [1.000, 1.000, 0.000],      # 255,255,0
    'beam':       [1.000, 0.502, 0.000],      # 255,128,0
    'column':     [0.502, 0.000, 1.000],      # 128,0,255
    'bookcase':   [1.000, 0.000, 1.000],      # 255,0,255
    'door':       [0.000, 1.000, 1.000],      # 0,255,255
    'window':     [0.000, 0.502, 1.000],      # 0,128,255
    'table':      [1.000, 0.000, 0.000],      # 255,0,0
    'chair':      [0.000, 1.000, 0.000],      # 0,255,0
    'sofa':       [1.000, 0.502, 0.502],      # 255,128,128
    'wall':       [0.502, 0.502, 0.502],      # 128,128,128
    'floor':      [0.588, 0.294, 0.000],      # 150,75,0
    'ceiling':    [0.706, 0.863, 1.000],      # 180,220,255
}

# Các class hình thành 1 mảng lớn liên tục, room-level thật sự
ROOM_LEVEL_CLASSES = {"floor", "ceiling"}

# =====================================================
# LOCKED CLASSES - KHÔNG verify, KHÔNG relabel
# =====================================================
LOCKED_CLASSES = {"floor", "ceiling"}

# =====================================================
# QUAN TRỌNG: Mapping nhãn số (model 2D) -> tên class
# Bạn PHẢI xác nhận thứ tự này khớp với model 2D của bạn!
# =====================================================
LABEL2D_CLASS_ORDER = [
    'ceiling', 'floor', 'wall', 'beam', 'column',
    'window', 'door', 'table', 'chair', 'sofa',
    'bookcase', 'whiteboard', 'clutter'
]
NUM_CLASSES_2D = len(LABEL2D_CLASS_ORDER)

# =====================================================
# THAM SỐ CHO FUSION & SO SÁNH 2D vs MLP
# =====================================================
CLASSES_2D_TRUSTED = {'window', 'whiteboard', 'table'}
SIMILARITY_THRESHOLD = 0.85

# =====================================================
# THAM SỐ FUSION - CÂN BẰNG 2D + MLP + BBOX
# =====================================================
MIN_MLP_POINTS_FOR_FUSION = 1000  # Nếu MLP < 1000 điểm → ưu tiên 2D
BBOX_MARGIN = 0.02  # 2cm - mở rộng bbox rất ít để lấp lỗ hổng
MIN_MLP_POINTS_FOR_Z_LOCK = 20  # Số điểm MLP tối thiểu để tin z-range MLP
LOW_MLP_2D_BOTTOM_CUT_RATIO = 0.2  # Cắt 20% điểm thấp nhất của 2D khi MLP quá ít
Z_RANGE_PERCENTILE = 2.0  # Dùng percentile 2-98% thay vì min/max thô để chống outlier
MLP_OUTLIER_DBSCAN_EPS = 0.25  # eps để tách cụm MLP chính, loại điểm nhiễu lẻ tẻ
MLP_OUTLIER_DBSCAN_MIN_SAMPLES = 15

# =====================================================
# THAM SỐ TÁCH 2D THÀNH TỪNG INSTANCE (MỚI)
# =====================================================
INSTANCE_2D_DBSCAN_EPS = 0.5
INSTANCE_2D_DBSCAN_MIN_SAMPLES = 15
INSTANCE_LOCAL_XY_MARGIN = 0.3

# =====================================================
# THAM SỐ KIỂM TRA HÌNH CHỮ NHẬT (chống wall bị nhận lầm thành whiteboard)
# =====================================================
RECTANGULARITY_CHECK_CLASSES = {"whiteboard"}
RECTANGULARITY_MIN_FILL_RATIO = 0.55
RECTANGULARITY_MAX_WIDTH_STD_RATIO = 0.35
RECTANGULARITY_DBSCAN_EPS = 0.3
RECTANGULARITY_DBSCAN_MIN_SAMPLES = 20

# =====================================================
# THAM SỐ CHỐNG WHITEBOARD "LƠ LỬNG GIỮA KHÔNG KHÍ"
# =====================================================
WHITEBOARD_MAX_DIST_TO_WALL = 0.25
WHITEBOARD_MIN_AREA_M2 = 0.15

# =====================================================
# THAM SỐ NGƯỜNG DIỆN TÍCH TỐI ĐA CHO WHITEBOARD
# =====================================================
WHITEBOARD_MAX_AREA_TOLERANCE_MULTIPLIER = 2.5
WHITEBOARD_MAX_AREA_FALLBACK_M2 = 2.0

# =====================================================
# THAM SỐ DBSCAN / TÁCH INSTANCE
# =====================================================
FLOOR_CEILING_PERCENTILE = 2.0
WALL_BOUNDARY_MARGIN = 0.15
DBSCAN_EPS = 0.35
DBSCAN_MIN_SAMPLES = 20
WALL_COORD_CLUSTER_EPS = 0.35
WALL_MIN_SAMPLES = 30
CHI2_CONFIDENCE = 0.99
AUTO_CORRECT_WALL = False
EXPORT_CORRECTED_PLY = True

# =====================================================
# THAM SỐ MỞ RỘNG Z THEO LỖ HỔNG TRÊN TƯỜNG (MỚI - SỬA LỖI #3)
# =====================================================
WALL_GAP_Z_BIN_SIZE = 0.05
WALL_GAP_DENSITY_RATIO_THRESHOLD = 0.3
WALL_GAP_MAX_EXTENSION = 1.5
WALL_GAP_SOLID_REFERENCE_MARGIN = 0.5

# =====================================================
# Các class furniture TUYỆT ĐỐI không cho relabel/ghi đè
# =====================================================
FURNITURE_CLASSES_PROTECTED = {"chair", "sofa", "table", "bookcase"}
PLANAR_CLASSES_CLAMP_Z_ONLY = {"window", "whiteboard"}

# =====================================================
# THAM SỐ KIỂM TRA CEILING/FLOOR (ĐÃ SỬA - LỖI #2)
# =====================================================
CEILING_FLOOR_Z_CHECK_CLASSES = {"ceiling", "floor"}
CEILING_FLOOR_PERPOINT_MAX_Z_DEVIATION = 0.15
CEILING_FLOOR_MAX_Z_DEVIATION = 0.15
CEILING_FLOOR_MAX_THICKNESS = 0.25
CEILING_FLOOR_DBSCAN_EPS = 0.3
CEILING_FLOOR_DBSCAN_MIN_SAMPLES = 20

# =====================================================
# THAM SỐ RANSAC - TÌM MẶT PHẲNG CHIẾM ĐA SỐ ĐIỂM
# =====================================================
RANSAC_PLANE_DISTANCE_THRESHOLD = 0.1
RANSAC_PLANE_NUM_ITERATIONS = 1000
RANSAC_PLANE_MIN_INLIER_RATIO = 0.5
RANSAC_PLANE_MIN_POINTS = 20
RANSAC_NORMAL_AXIS_DOMINANCE_RATIO = 1.3

# =====================================================
# THAM SỐ ĐỘ DÀY MẶT PHẲNG (tự động theo std)
# =====================================================
PLANE_THICKNESS_STD_MULTIPLIER = 3.0
PLANE_THICKNESS_MIN_THRESHOLD = 0.02
PLANAR_CLAMP_MARGIN = 0.05

# =====================================================
# THAM SỐ DBSCAN SUB-CLUSTER
# =====================================================
SUBCLUSTER_DBSCAN_EPS = 0.25
SUBCLUSTER_DBSCAN_MIN_SAMPLES = 15
SUBCLUSTER_MIN_POINTS_TO_CHECK = 15

# =====================================================
# CHECKPOINT FUNCTIONS
# =====================================================

def save_checkpoint(room_name: str, 
                   points, corrected_classes, original_mlp_classes,
                   labels_2d, confs_2d, labels_2d_names,
                   room_geom_ref, processed_rooms: list,
                   total_instances: int, total_suspect: int, total_relabeled: int,
                   writer_state=None):
    """
    Lưu checkpoint cho từng room
    """
    import pickle
    
    checkpoint_data = {
        'room_name': room_name,
        'points': points,
        'corrected_classes': corrected_classes,
        'original_mlp_classes': original_mlp_classes,
        'labels_2d': labels_2d,
        'confs_2d': confs_2d,
        'labels_2d_names': labels_2d_names,
        'room_geom_ref': room_geom_ref,
        'processed_rooms': processed_rooms,
        'total_instances': total_instances,
        'total_suspect': total_suspect,
        'total_relabeled': total_relabeled,
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    }
    
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{room_name}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    print(f"   💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(room_name: str) -> dict:
    """
    Load checkpoint cho 1 room
    """
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{room_name}.pkl"
    if not checkpoint_path.exists():
        return None
    
    import pickle
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   🔄 Loaded checkpoint from: {checkpoint_path}")
    return data


def save_global_checkpoint(processed_rooms: list, 
                          total_instances: int, total_suspect: int, total_relabeled: int):
    """
    Lưu checkpoint tổng thể
    """
    import pickle
    
    checkpoint_data = {
        'processed_rooms': processed_rooms,
        'total_instances': total_instances,
        'total_suspect': total_suspect,
        'total_relabeled': total_relabeled,
        'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    }
    
    checkpoint_path = CHECKPOINT_DIR / "global_checkpoint.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    return checkpoint_path


def load_global_checkpoint() -> dict:
    """
    Load checkpoint tổng thể
    """
    checkpoint_path = CHECKPOINT_DIR / "global_checkpoint.pkl"
    if not checkpoint_path.exists():
        return None
    
    import pickle
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"   🔄 Loaded global checkpoint: {len(data['processed_rooms'])} rooms processed")
    return data


def cleanup_checkpoint(room_name: str):
    """
    Xóa checkpoint sau khi xử lý xong room
    """
    checkpoint_path = CHECKPOINT_DIR / f"checkpoint_{room_name}.pkl"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"   🗑️  Removed checkpoint: {checkpoint_path}")


# =====================================================
# 2D CACHE CONFIG
# =====================================================
CACHE_2D_DIR = Path(r"/compute_home/slurmdang12/datasets/ErrorMining_output/cache_2d")
CACHE_2D_DIR.mkdir(parents=True, exist_ok=True)
USE_2D_CACHE = True  # Bật cache để tăng tốc
AUTO_CLEANUP_CACHE = True  # Tự động xóa cache sau khi xử lý xong
KEEP_2D_CACHE = False

def get_memory_usage():
    """Lấy memory usage hiện tại (MB)"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory(label="Memory"):
    """In memory usage với label"""
    print(f"   📊 {label}: {get_memory_usage():.1f} MB")
def get_2d_cache_path(room_name: str) -> Path:
    """Lấy đường dẫn file cache cho 1 room"""
    return CACHE_2D_DIR / f"{room_name}_2d_merged.npz"


def load_2d_from_cache(room_name: str) -> dict:
    """Load dữ liệu 2D từ cache nếu có"""
    cache_path = get_2d_cache_path(room_name)
    if cache_path.exists():
        print(f"   📦 Load 2D from cache: {cache_path.name}")
        data = np.load(cache_path, allow_pickle=True)
        return {
            'points': data['points'],
            'labels': data['labels'],
            'confidences': data['confidences'],
        }
    return None

def save_2d_to_cache(room_name: str, points: np.ndarray, labels: np.ndarray, confs: np.ndarray):
    """Lưu dữ liệu 2D đã gộp vào cache"""
    cache_path = get_2d_cache_path(room_name)
    print(f"   💾 Saving 2D to cache: {cache_path.name}")
    np.savez_compressed(
        cache_path,
        points=points,
        labels=labels,
        confidences=confs
    )
    print(f"   ✅ Cache saved: {len(points):,} points")


def cleanup_2d_cache(room_name: str = None):
    """Xóa cache 2D"""
    if room_name:
        cache_path = get_2d_cache_path(room_name)
        if cache_path.exists():
            cache_path.unlink()
            print(f"   🗑️  Removed cache: {cache_path.name}")
    else:
        # Xóa tất cả cache
        import shutil
        if CACHE_2D_DIR.exists():
            shutil.rmtree(CACHE_2D_DIR)
            CACHE_2D_DIR.mkdir(parents=True, exist_ok=True)
            print(f"   🗑️  Removed all 2D cache")

def load_2d_data(room_name: str, force_reload: bool = False):
    """
    Load dữ liệu 2D - có cache để tăng tốc
    """
    import gc
    import tempfile
    import shutil
    from collections import defaultdict
    
    room_name = room_name.replace("_prediction", "")
    
    # ===== KIỂM TRA CACHE =====
    if USE_2D_CACHE and not force_reload:
        cached_data = load_2d_from_cache(room_name)
        if cached_data is not None:
            return cached_data
    
    search_dirs = [
        Path(r"/compute_home/slurmdang12/datasets/mapped_pcd/Area_5"),
        # Path(r"D:\Thu\KhoaLuan\results\mapped_pcd\Area_1"),
        # Path(r"D:\Thu\KhoaLuan\results\mapped_pcd"),
    ]

    all_segments = []
    found_any = False

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        
        print(f"   🔍 Đang tìm 2D trong: {base_dir}")

        # CÁCH 1: Thư mục trực tiếp (đã gộp sẵn)
        direct_dir = base_dir / room_name
        if direct_dir.exists() and direct_dir.is_dir():
            print(f"   📁 Kiểm tra thư mục trực tiếp: {direct_dir}")
            
            points_file = direct_dir / f"{room_name}_points.npy"
            labels_file = direct_dir / f"{room_name}_labels.npy"
            confs_file = direct_dir / f"{room_name}_confidences.npy"
            
            if not points_file.exists():
                points_candidates = list(direct_dir.glob("*points*.npy"))
                if points_candidates:
                    points_file = points_candidates[0]
                    base_name = points_file.stem.replace('_points', '')
                    labels_file = direct_dir / f"{base_name}_labels.npy"
                    confs_file = direct_dir / f"{base_name}_confidences.npy"
            
            if points_file.exists() and labels_file.exists() and confs_file.exists():
                print(f"   ✅ Tìm thấy dữ liệu 2D đã gộp trong {direct_dir}")
                pts = np.load(points_file, allow_pickle=True)
                if pts.shape[1] > 3:
                    pts = pts[:, :3]
                
                result = {
                    'points': pts,
                    'labels': np.load(labels_file, allow_pickle=True),
                    'confidences': np.load(confs_file, allow_pickle=True),
                }
                
                # Lưu vào cache để lần sau dùng
                if USE_2D_CACHE:
                    save_2d_to_cache(room_name, result['points'], result['labels'], result['confidences'])
                
                return result
        
        # CÁCH 2: Tìm các segment (cần gộp)
        segment_dirs = sorted(base_dir.glob(f"{room_name}_segment_*"))
        if segment_dirs:
            print(f"   📁 Tìm thấy {len(segment_dirs)} segment(s)")
            for seg_dir in segment_dirs:
                seg_name = seg_dir.name
                points_file = seg_dir / f"{seg_name}_points.npy"
                labels_file = seg_dir / f"{seg_name}_labels.npy"
                confs_file = seg_dir / f"{seg_name}_confidences.npy"
                
                if points_file.exists() and labels_file.exists() and confs_file.exists():
                    all_segments.append({
                        'points_file': points_file,
                        'labels_file': labels_file,
                        'confs_file': confs_file,
                        'name': seg_name
                    })
                    found_any = True
                    print(f"   ✅ Tìm thấy segment: {seg_name}")
                else:
                    print(f"   ⚠️  Thiếu file trong {seg_dir}")
        
        if found_any:
            break

    if not found_any or not all_segments:
        warnings.warn(f"Không tìm thấy file 2D cho {room_name}")
        return None

    # ===== GỘP CÁC SEGMENT =====
    print(f"   🔄 Đang gộp {len(all_segments)} segments...")
    
    all_points = []
    all_labels = []
    all_confs = []
    
    for seg in all_segments:
        pts = np.load(seg['points_file'], allow_pickle=True)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        lbl = np.load(seg['labels_file'], allow_pickle=True)
        cf = np.load(seg['confs_file'], allow_pickle=True)
        
        all_points.append(pts)
        all_labels.append(lbl)
        all_confs.append(cf)
        
        gc.collect()
    
    # Gộp
    print(f"   🔄 Concatenating...")
    merged_points = np.concatenate(all_points, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_confs = np.concatenate(all_confs, axis=0)
    
    del all_points, all_labels, all_confs
    gc.collect()
    
    # Khử trùng
    if len(all_segments) > 1:
        print(f"   🔄 Deduplicating {len(merged_points):,} points...")
        merged_points, merged_labels, merged_confs = _dedupe_by_confidence_batch(
            merged_points, merged_labels, merged_confs
        )
        print(f"   -> Sau khử trùng: {len(merged_points):,} points")
    
    print(f"   ✅ Gộp thành công: {len(merged_points):,} points")
    
    result = {
        'points': merged_points,
        'labels': merged_labels,
        'confidences': merged_confs,
    }
    
    # Lưu vào cache
    if USE_2D_CACHE:
        save_2d_to_cache(room_name, merged_points, merged_labels, merged_confs)
    
    return result

# =====================================================
# HÀM GỘP CÁC SEGMENT 2D THÀNH ROOM (GIỐNG MLP)
# =====================================================

def group_2d_segments_by_room(area_path: Path, area_name: str) -> dict:
    """
    Gộp các segment 2D thành room dựa trên tên thư mục
    Tương tự như group_samples_by_room trong MLP
    """
    from pathlib import Path
    
    # Tìm tất cả thư mục segment trong area
    segment_dirs = sorted(area_path.glob("*_segment_*"))
    
    if not segment_dirs:
        print(f"   ❌ Không tìm thấy segment nào trong {area_path}")
        return {}
    
    room_groups = {}
    
    for seg_dir in segment_dirs:
        seg_name = seg_dir.name
        # Tách tên room từ segment (ví dụ: "conferenceRoom_1_segment_0" -> "conferenceRoom_1")
        base_room = seg_name.split('_segment_')[0]
        
        if base_room not in room_groups:
            room_groups[base_room] = {
                'base_room': base_room,
                'display_name': base_room,
                'segments': []
            }
        room_groups[base_room]['segments'].append(seg_dir)
    
    # Sắp xếp segments theo thứ tự
    for room in room_groups.values():
        room['segments'] = sorted(room['segments'])
    
    return room_groups


def load_and_merge_2d_segments(segment_dirs: list) -> dict:
    """
    Load và gộp dữ liệu 2D từ nhiều segment thành 1 room
    Tương tự như load_and_merge_group trong MLP
    """
    all_points = []
    all_labels = []
    all_confs = []
    
    for seg_dir in segment_dirs:
        seg_name = seg_dir.name
        points_file = seg_dir / f"{seg_name}_points.npy"
        labels_file = seg_dir / f"{seg_name}_labels.npy"
        confs_file = seg_dir / f"{seg_name}_confidences.npy"
        
        if points_file.exists() and labels_file.exists() and confs_file.exists():
            pts = np.load(points_file)
            lbl = np.load(labels_file)
            cf = np.load(confs_file)
            
            all_points.append(pts)
            all_labels.append(lbl)
            all_confs.append(cf)
        else:
            print(f"   ⚠️  Thiếu file trong {seg_dir}")
    
    if not all_points:
        return None
    
    # Gộp tất cả
    merged_points = np.concatenate(all_points, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_confs = np.concatenate(all_confs, axis=0)
    
    # Khử trùng điểm (giống MLP)
    merged_points, merged_labels, merged_confs = _dedupe_by_confidence(
        merged_points, merged_labels, merged_confs
    )
    
    return {
        'points': merged_points,
        'labels': merged_labels,
        'confidences': merged_confs,
    }


def process_2d_room(room_group: dict, ply_path: Path) -> dict:
    """
    Xử lý 1 room 2D đã gộp
    """
    room_name = room_group['base_room']
    segment_dirs = room_group['segments']
    
    print(f"\n   📁 Room: {room_name} ({len(segment_dirs)} segments)")
    
    # Load và gộp dữ liệu 2D
    merged_data = load_and_merge_2d_segments(segment_dirs)
    if merged_data is None:
        print(f"   ❌ Không load được 2D cho {room_name}")
        return None
    
    print(f"   ✅ Đã gộp {len(segment_dirs)} segments → {len(merged_data['points'])} points")
    
    return merged_data


def load_2d_data_with_grouping(ply_path: Path, area_path: Path, area_name: str) -> dict:
    """
    Hàm chính: load dữ liệu 2D đã gộp theo room
    Thay thế cho hàm load_2d_data cũ
    """
    room_name = ply_path.stem.replace('_mlp_refined', '').replace('_prediction', '')
    
    # Gộp các segment 2D thành room
    room_groups = group_2d_segments_by_room(area_path, area_name)
    
    if not room_groups:
        print(f"   ❌ Không tìm thấy segment 2D nào cho {room_name}")
        return None
    
    # Tìm room group tương ứng
    if room_name not in room_groups:
        # Thử tìm kiếm fuzzy (nếu có prefix khác)
        found = False
        for key in room_groups.keys():
            if room_name in key or key in room_name:
                room_name = key
                found = True
                break
        if not found:
            print(f"   ❌ Không tìm thấy room '{room_name}' trong 2D segments")
            return None
    
    room_group = room_groups[room_name]
    
    # Xử lý room đã gộp
    merged_data = process_2d_room(room_group, ply_path)
    
    return merged_data

# =====================================================
# STEP 1: Load PLY + map màu -> class
# =====================================================

def load_ply(ply_path: Path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def color_to_class_batch(colors: np.ndarray, palette: dict, eps: float = 0.05):
    palette_names = list(palette.keys())
    palette_arr = np.array([palette[c] for c in palette_names])

    dists = np.linalg.norm(colors[:, None, :] - palette_arr[None, :, :], axis=2)
    best_idx = np.argmin(dists, axis=1)
    best_dist = dists[np.arange(len(colors)), best_idx]

    labels = np.array([palette_names[i] for i in best_idx], dtype=object)
    labels[best_dist >= eps] = None
    return labels


def label_to_class_name(label_int: int) -> str:
    if label_int < 0 or label_int >= NUM_CLASSES_2D:
        return None
    return LABEL2D_CLASS_ORDER[label_int]


def compute_class_similarity(predicted_cls: str, label_2d_cls: str, conf_2d: float) -> float:
    if predicted_cls == label_2d_cls:
        return float(conf_2d)
    return 0.0


# =====================================================
# STEP 2: TÁCH INSTANCE
# =====================================================

def geometric_room_reference(all_points: np.ndarray):
    z = all_points[:, 2]
    floor_z = float(np.percentile(z, FLOOR_CEILING_PERCENTILE))
    ceiling_z = float(np.percentile(z, 100 - FLOOR_CEILING_PERCENTILE))

    x, y = all_points[:, 0], all_points[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    near_boundary = (
        (x - xmin < WALL_BOUNDARY_MARGIN) | (xmax - x < WALL_BOUNDARY_MARGIN) |
        (y - ymin < WALL_BOUNDARY_MARGIN) | (ymax - y < WALL_BOUNDARY_MARGIN)
    )
    wall_points = all_points[near_boundary]

    floor_points = all_points[np.abs(z - floor_z) < (ceiling_z - floor_z) * 0.05]
    ceiling_points = all_points[np.abs(z - ceiling_z) < (ceiling_z - floor_z) * 0.05]

    return {
        "floor_z": floor_z,
        "ceiling_z": ceiling_z,
        "wall": wall_points,
        "floor": floor_points,
        "ceiling": ceiling_points,
    }


def estimate_normals(points):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def split_wall_instances(points: np.ndarray, coord_threshold: float = 0.1) -> list:
    n = len(points)
    if n < 100:
        return [np.ones(n, dtype=bool)]

    x = points[:, 0]
    y = points[:, 1]

    x_bins = np.arange(x.min(), x.max() + coord_threshold, coord_threshold)
    x_digitized = np.digitize(x, x_bins)
    x_centers = []
    for i in range(1, len(x_bins)):
        mask = (x_digitized == i)
        if np.sum(mask) > 50:
            center = np.mean(x[mask])
            x_centers.append((center, np.sum(mask)))

    y_bins = np.arange(y.min(), y.max() + coord_threshold, coord_threshold)
    y_digitized = np.digitize(y, y_bins)
    y_centers = []
    for i in range(1, len(y_bins)):
        mask = (y_digitized == i)
        if np.sum(mask) > 50:
            center = np.mean(y[mask])
            y_centers.append((center, np.sum(mask)))

    x_centers = sorted(x_centers, key=lambda t: t[1], reverse=True)[:2]
    y_centers = sorted(y_centers, key=lambda t: t[1], reverse=True)[:2]

    instances = []
    for x_val, _ in x_centers:
        mask = np.abs(x - x_val) < coord_threshold
        if np.sum(mask) > 50:
            instances.append(mask)

    for y_val, _ in y_centers:
        mask = np.abs(y - y_val) < coord_threshold
        if np.sum(mask) > 50:
            instances.append(mask)

    if len(instances) < 4:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        margin = 0.15
        instances = []
        for mask in [
            x < x_min + margin,
            x > x_max - margin,
            y < y_min + margin,
            y > y_max - margin,
        ]:
            if np.sum(mask) > 50:
                instances.append(mask)

    if len(instances) < 2:
        return [np.ones(n, dtype=bool)]

    return instances


def split_into_instances(points: np.ndarray, cls: str):
    n = len(points)

    if cls == "wall":
        return split_wall_instances(points)

    if cls in ROOM_LEVEL_CLASSES:
        return [np.ones(n, dtype=bool)]

    if n < DBSCAN_MIN_SAMPLES:
        return [np.ones(n, dtype=bool)] if n >= 10 else []

    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=1).fit(points)
    labels = clustering.labels_

    instances = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        instances.append(labels == lbl)

    return instances


# =====================================================
# STEP 3: FEATURE EXTRACTION
# =====================================================

def eigen_features(points):
    if len(points) < 20:
        return 0.0, 0.0, 0.0, 0.0
    cov = np.cov(points.T)
    eigvals = np.sort(np.linalg.eigh(cov)[0])[::-1]
    l1, l2, l3 = eigvals + 1e-12
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    roughness = l3 / l1
    curvature = l3 / (l1 + l2 + l3)
    return float(linearity), float(planarity), float(roughness), float(curvature)


def mesh_features(points):
    if len(points) < 4:
        return 0.0, 0.0
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        hull, _ = pcd.compute_convex_hull()
        return float(hull.get_surface_area()), float(hull.get_volume())
    except Exception:
        return 0.0, 0.0


def shape_features(points):
    mins, maxs = points.min(axis=0), points.max(axis=0)
    dims = maxs - mins
    width, depth, height = float(dims[0]), float(dims[1]), float(dims[2])
    volume_bbox = float(np.prod(dims))
    longest, shortest = float(np.max(dims)), float(np.min(dims)) + 1e-12
    elongation = longest / shortest
    compactness = volume_bbox / ((dims.sum()) ** 3 + 1e-12)
    return width, depth, height, compactness, elongation


def sphericity(area, volume):
    if area <= 0 or volume <= 0:
        return 0.0
    return float((np.pi ** (1/3)) * ((6 * volume) ** (2/3)) / area)


def normal_variance(pcd):
    normals = np.asarray(pcd.normals)
    return float(np.var(normals)) if len(normals) else 0.0


def normal_entropy(pcd, bins=16):
    normals = np.asarray(pcd.normals)
    if len(normals) == 0:
        return 0.0
    hist, _ = np.histogramdd(normals, bins=bins, range=[[-1, 1]] * 3)
    hist = hist.flatten()
    hist = hist[hist > 0]
    prob = hist / hist.sum()
    return float(-np.sum(prob * np.log2(prob + 1e-12)))


def extract_geometry(points: np.ndarray) -> dict:
    pcd = estimate_normals(points)
    linearity, planarity, roughness, curvature = eigen_features(points)
    surface_area, volume = mesh_features(points)
    width, depth, height, compactness, elongation = shape_features(points)

    return {
        "NumPoints": len(points),
        "SurfaceArea": surface_area,
        "Volume": volume,
        "Height": height,
        "Width": width,
        "Depth": depth,
        "HeightAboveFloor": np.nan,
        "HeightClearance": np.nan,
        "Planarity": planarity,
        "Linearity": linearity,
        "Roughness": roughness,
        "Curvature": curvature,
        "NormalEntropy": normal_entropy(pcd),
        "NormalVariance": normal_variance(pcd),
        "Compactness": compactness,
        "Elongation": elongation,
        "Sphericity": sphericity(surface_area, volume),
    }


def extract_spatial(points: np.ndarray, room_geom_ref: dict) -> dict:
    from scipy.spatial import cKDTree

    def min_dist_to(key):
        target = room_geom_ref.get(key)
        if target is None or len(target) == 0:
            return np.nan
        tree = cKDTree(target)
        dists, _ = tree.query(points, k=1)
        return float(np.min(dists))

    def normal_alignment_to(key):
        target = room_geom_ref.get(key)
        if target is None or len(target) < 20:
            return np.nan
        try:
            pcd_a = estimate_normals(points)
            pcd_b = estimate_normals(target)
            na = np.asarray(pcd_a.normals).mean(axis=0)
            nb = np.asarray(pcd_b.normals).mean(axis=0)
            norm_a, norm_b = np.linalg.norm(na), np.linalg.norm(nb)
            if norm_a < 1e-9 or norm_b < 1e-9:
                return np.nan
            return float(abs(np.dot(na / norm_a, nb / norm_b)))
        except Exception:
            return np.nan

    return {
        "DistanceToWall": min_dist_to("wall"),
        "DistanceToFloor": min_dist_to("floor"),
        "DistanceToCeiling": min_dist_to("ceiling"),
        "NormalAlignmentWall": normal_alignment_to("wall"),
        "NormalAlignmentFloor": normal_alignment_to("floor"),
        "NormalAlignmentCeiling": normal_alignment_to("ceiling"),
    }


def fill_height_from_floor(geom: dict, points: np.ndarray, floor_z: float):
    if floor_z is None or not np.isfinite(floor_z):
        return geom
    min_z, max_z = points[:, 2].min(), points[:, 2].max()
    geom["HeightAboveFloor"] = float(min_z - floor_z)
    geom["HeightClearance"] = float(max_z - floor_z)
    return geom


# =====================================================
# STEP 4: FUSION - Cân bằng 2D + MLP + BBOX
# =====================================================

def robust_mlp_z_range(ply_points: np.ndarray, mlp_mask: np.ndarray):
    pts_mlp = ply_points[mlp_mask]
    if len(pts_mlp) < MLP_OUTLIER_DBSCAN_MIN_SAMPLES:
        return None, None

    try:
        clustering = DBSCAN(
            eps=MLP_OUTLIER_DBSCAN_EPS,
            min_samples=MLP_OUTLIER_DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit(pts_mlp)
        labels = clustering.labels_
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            main_pts = pts_mlp
        else:
            biggest_label = np.bincount(valid_labels).argmax()
            main_pts = pts_mlp[labels == biggest_label]
            n_dropped = len(pts_mlp) - len(main_pts)
            if n_dropped > 0:
                print(f"    [robust_z] Loại {n_dropped} điểm MLP nhiễu/lẻ tẻ")
    except Exception:
        main_pts = pts_mlp

    z_vals = main_pts[:, 2]
    z_min = float(np.percentile(z_vals, Z_RANGE_PERCENTILE))
    z_max = float(np.percentile(z_vals, 100 - Z_RANGE_PERCENTILE))
    return z_min, z_max


def rectangularity_fill_ratio(points: np.ndarray) -> float:
    if len(points) < 20:
        return 1.0

    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u_axis, v_axis = eigvecs[:, 0], eigvecs[:, 1]
    proj_u = centered @ u_axis
    proj_v = centered @ v_axis

    bbox_area = (proj_u.max() - proj_u.min()) * (proj_v.max() - proj_v.min())
    if bbox_area < 1e-6:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(np.column_stack([proj_u, proj_v]))
        hull_area = hull.volume
    except Exception:
        return 0.0

    return float(hull_area / bbox_area)


def rectangularity_width_std_ratio(points: np.ndarray, n_bins: int = 10) -> float:
    if len(points) < 30:
        return 0.0

    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u_axis, v_axis = eigvecs[:, 0], eigvecs[:, 1]
    proj_u = centered @ u_axis
    proj_v = centered @ v_axis

    v_min, v_max = proj_v.min(), proj_v.max()
    if v_max - v_min < 1e-6:
        return 0.0

    bin_edges = np.linspace(v_min, v_max, n_bins + 1)
    widths = []
    for i in range(n_bins):
        mask = (proj_v >= bin_edges[i]) & (proj_v < bin_edges[i + 1])
        if np.sum(mask) < 5:
            continue
        u_vals = proj_u[mask]
        widths.append(u_vals.max() - u_vals.min())

    if len(widths) < 3:
        return 0.0

    widths = np.array(widths)
    mean_w = widths.mean()
    if mean_w < 1e-6:
        return 0.0
    return float(widths.std() / mean_w)


def is_valid_rectangular_shape(points: np.ndarray) -> bool:
    fill_ratio = rectangularity_fill_ratio(points)
    width_std_ratio = rectangularity_width_std_ratio(points)

    is_ok = (fill_ratio >= RECTANGULARITY_MIN_FILL_RATIO) and \
            (width_std_ratio <= RECTANGULARITY_MAX_WIDTH_STD_RATIO)

    print(f"    [rectangularity] fill_ratio={fill_ratio:.2f}, "
          f"width_std_ratio={width_std_ratio:.2f} -> {'OK' if is_ok else 'REJECT'}")
    return is_ok


def is_attached_to_wall_and_large_enough(points: np.ndarray, wall_points,
                                         whiteboard_max_area: float = None) -> bool:
    if len(points) < 4:
        return False

    if wall_points is None or len(wall_points) == 0:
        return False

    from scipy.spatial import cKDTree
    tree = cKDTree(wall_points)
    dists, _ = tree.query(points, k=1)
    mean_dist_to_wall = float(np.mean(dists))
    is_attached = mean_dist_to_wall <= WHITEBOARD_MAX_DIST_TO_WALL

    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    u_axis, v_axis = eigvecs[:, 0], eigvecs[:, 1]
    proj_u = centered @ u_axis
    proj_v = centered @ v_axis
    approx_area = float((proj_u.max() - proj_u.min()) * (proj_v.max() - proj_v.min()))
    is_large_enough = approx_area >= WHITEBOARD_MIN_AREA_M2

    is_not_too_large = True
    if whiteboard_max_area is not None:
        is_not_too_large = approx_area <= whiteboard_max_area

    is_ok = is_attached and is_large_enough and is_not_too_large

    print(f"    [wall_attach] dist_to_wall={mean_dist_to_wall*100:.1f}cm, "
          f"approx_area={approx_area:.3f}m² -> {'OK' if is_ok else 'REJECT'}")
    return is_ok


def filter_non_rectangular_components(ply_points: np.ndarray,
                                      final_mask: np.ndarray,
                                      cls: str,
                                      wall_points=None,
                                      whiteboard_max_area: float = None) -> np.ndarray:
    if cls not in RECTANGULARITY_CHECK_CLASSES:
        return final_mask

    idx_all = np.where(final_mask)[0]
    if len(idx_all) < RECTANGULARITY_DBSCAN_MIN_SAMPLES:
        return final_mask

    pts = ply_points[idx_all]
    try:
        clustering = DBSCAN(
            eps=RECTANGULARITY_DBSCAN_EPS,
            min_samples=RECTANGULARITY_DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit(pts)
    except Exception:
        return final_mask

    labels = clustering.labels_
    new_mask = np.zeros_like(final_mask)

    for lbl in set(labels):
        if lbl == -1:
            continue
        comp_idx = idx_all[labels == lbl]
        comp_pts = ply_points[comp_idx]

        passes_rectangularity = is_valid_rectangular_shape(comp_pts)
        if not passes_rectangularity:
            continue

        passes_wall_and_size = is_attached_to_wall_and_large_enough(
            comp_pts, wall_points, whiteboard_max_area=whiteboard_max_area
        )
        if not passes_wall_and_size:
            continue

        new_mask[comp_idx] = True

    n_rejected = np.sum(final_mask) - np.sum(new_mask)
    if n_rejected > 0:
        print(f"  ⚠️ Đã loại {n_rejected} điểm '{cls}' do không đạt kiểm tra")

    return new_mask


def fuse_with_2d_mlp_bbox(ply_points: np.ndarray,
                          mlp_classes: np.ndarray,
                          labels_2d: np.ndarray,
                          confs_2d: np.ndarray,
                          cls: str,
                          floor_z: float,
                          ceiling_z: float,
                          mask_2d_subset: np.ndarray = None,
                          wall_points=None,
                          whiteboard_max_area: float = None) -> tuple:
    cls_label_idx = LABEL2D_CLASS_ORDER.index(cls) if cls in LABEL2D_CLASS_ORDER else -1
    if cls_label_idx == -1:
        return None, None

    mask_2d = mask_2d_subset if mask_2d_subset is not None else (labels_2d == cls_label_idx)
    mlp_mask_global = (mlp_classes == cls)
    locked_points = np.isin(mlp_classes, list(LOCKED_CLASSES))

    if np.sum(mask_2d) < 10:
        return None, None

    pts_2d = ply_points[mask_2d]
    mins_2d = pts_2d.min(axis=0)
    maxs_2d = pts_2d.max(axis=0)
    margin = BBOX_MARGIN

    local_xy = (
        (ply_points[:, 0] >= mins_2d[0] - INSTANCE_LOCAL_XY_MARGIN) &
        (ply_points[:, 0] <= maxs_2d[0] + INSTANCE_LOCAL_XY_MARGIN) &
        (ply_points[:, 1] >= mins_2d[1] - INSTANCE_LOCAL_XY_MARGIN) &
        (ply_points[:, 1] <= maxs_2d[1] + INSTANCE_LOCAL_XY_MARGIN)
    )
    mlp_mask = mlp_mask_global & local_xy

    mlp_count = np.sum(mlp_mask)

    if mlp_count >= MIN_MLP_POINTS_FOR_Z_LOCK:
        z_min_mlp, z_max_mlp = robust_mlp_z_range(ply_points, mlp_mask)

        if z_min_mlp is None:
            z_sorted = np.sort(pts_2d[:, 2])
            cutoff_idx = int(len(z_sorted) * LOW_MLP_2D_BOTTOM_CUT_RATIO)
            z_min_final = z_sorted[cutoff_idx] - margin
            z_max_final = maxs_2d[2] + margin
        else:
            z_min_final = max(z_min_mlp, mins_2d[2]) - margin
            z_max_final = min(z_max_mlp, maxs_2d[2]) + margin

            if z_min_final >= z_max_final:
                z_min_final = z_min_mlp - margin
                z_max_final = z_max_mlp + margin
    else:
        z_sorted = np.sort(pts_2d[:, 2])
        cutoff_idx = int(len(z_sorted) * LOW_MLP_2D_BOTTOM_CUT_RATIO)
        z_min_final = z_sorted[cutoff_idx] - margin
        z_max_final = maxs_2d[2] + margin

    mins_bbox = np.array([mins_2d[0] - margin, mins_2d[1] - margin, z_min_final])
    maxs_bbox = np.array([maxs_2d[0] + margin, maxs_2d[1] + margin, z_max_final])

    bbox_mask = (ply_points[:, 0] >= mins_bbox[0]) & (ply_points[:, 0] <= maxs_bbox[0]) & \
                (ply_points[:, 1] >= mins_bbox[1]) & (ply_points[:, 1] <= maxs_bbox[1]) & \
                (ply_points[:, 2] >= mins_bbox[2]) & (ply_points[:, 2] <= maxs_bbox[2])

    floor_threshold = floor_z + 0.1
    ceiling_threshold = ceiling_z - 0.1
    not_floor = ply_points[:, 2] > floor_threshold
    not_ceiling = ply_points[:, 2] < ceiling_threshold
    not_floor_ceiling = not_floor & not_ceiling

    valid_mask = bbox_mask & not_floor_ceiling & (~locked_points) & local_xy

    both_correct = mask_2d & mlp_mask
    only_2d = mask_2d & ~mlp_mask
    only_mlp = mlp_mask & ~mask_2d

    use_2d_only = mlp_count < MIN_MLP_POINTS_FOR_FUSION

    if use_2d_only:
        final_mask = valid_mask & mask_2d
    else:
        final_mask = valid_mask & (both_correct | only_2d | only_mlp)

        if np.sum(final_mask) < 10:
            final_mask = valid_mask & mask_2d & (~locked_points)

    final_mask = filter_non_rectangular_components(
        ply_points, final_mask, cls, wall_points=wall_points,
        whiteboard_max_area=whiteboard_max_area,
    )

    conf_2d = np.mean(confs_2d[mask_2d]) if np.any(mask_2d) else 0.5

    bbox_info = {
        'x_min': mins_bbox[0], 'x_max': maxs_bbox[0],
        'y_min': mins_bbox[1], 'y_max': maxs_bbox[1],
        'z_min': mins_bbox[2], 'z_max': maxs_bbox[2],
        'num_points': np.sum(final_mask),
        'confidence': conf_2d,
        'use_2d_only': use_2d_only,
        'both_correct': np.sum(both_correct),
        'only_2d': np.sum(only_2d),
        'only_mlp': np.sum(only_mlp),
    }

    return final_mask, bbox_info


def estimate_whiteboard_max_area(joint_prototype: dict) -> float:
    try:
        proto = joint_prototype.get("whiteboard")
        if proto is None:
            raise KeyError("whiteboard không có trong joint_prototype")

        mu = proto["mu"]
        feats = proto["features"]

        def get_mu_value(name):
            idx = feats.index(name)
            return float(mu[idx])

        if "SurfaceArea" in feats:
            base_area = get_mu_value("SurfaceArea")
        elif "Width" in feats and "Height" in feats:
            base_area = get_mu_value("Width") * get_mu_value("Height")
        else:
            raise KeyError("Thiếu cả SurfaceArea và Width/Height trong prototype")

        if not np.isfinite(base_area) or base_area <= 0:
            raise ValueError(f"base_area không hợp lệ: {base_area}")

        max_area = base_area * WHITEBOARD_MAX_AREA_TOLERANCE_MULTIPLIER
        return max_area

    except Exception as e:
        return WHITEBOARD_MAX_AREA_FALLBACK_M2


def apply_fusion_to_class(points: np.ndarray,
                          predicted_classes: np.ndarray,
                          labels_2d: np.ndarray,
                          confs_2d: np.ndarray,
                          cls: str,
                          joint_prototype: dict,
                          room_geom_ref: dict) -> tuple:
    if cls in LOCKED_CLASSES:
        return predicted_classes, False, "locked_class"

    floor_z = room_geom_ref.get('floor_z', 0)
    ceiling_z = room_geom_ref.get('ceiling_z', 3.0)
    wall_points = room_geom_ref.get('wall', None)

    whiteboard_max_area = None
    if cls == "whiteboard":
        whiteboard_max_area = estimate_whiteboard_max_area(joint_prototype)

    cls_label_idx = LABEL2D_CLASS_ORDER.index(cls) if cls in LABEL2D_CLASS_ORDER else -1
    if cls_label_idx == -1:
        return predicted_classes, False, "no_2d_label"

    mask_2d_all = (labels_2d == cls_label_idx)
    if np.sum(mask_2d_all) < 10:
        return predicted_classes, False, "fusion_failed"

    idx_2d_all = np.where(mask_2d_all)[0]
    pts_2d_all = points[idx_2d_all]

    try:
        cluster_labels = DBSCAN(
            eps=INSTANCE_2D_DBSCAN_EPS,
            min_samples=INSTANCE_2D_DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit(pts_2d_all).labels_
    except Exception:
        cluster_labels = np.zeros(len(pts_2d_all), dtype=int)

    unique_labels = sorted(set(cluster_labels) - {-1})
    if not unique_labels:
        unique_labels = [0]
        cluster_labels = np.zeros(len(pts_2d_all), dtype=int)

    corrected_classes = predicted_classes.copy()
    any_applied = False
    reasons = []

    for cl_lbl in unique_labels:
        comp_idx = idx_2d_all[cluster_labels == cl_lbl]
        if len(comp_idx) < 10:
            continue

        mask_2d_instance = np.zeros(len(points), dtype=bool)
        mask_2d_instance[comp_idx] = True

        final_mask, bbox_info = fuse_with_2d_mlp_bbox(
            points, corrected_classes, labels_2d, confs_2d,
            cls, floor_z, ceiling_z, mask_2d_subset=mask_2d_instance,
            wall_points=wall_points, whiteboard_max_area=whiteboard_max_area,
        )

        if final_mask is None or bbox_info is None:
            reasons.append(f"instance{cl_lbl}_fusion_failed")
            continue

        if bbox_info['num_points'] < 10:
            reasons.append(f"instance{cl_lbl}_too_few_points")
            continue

        corrected_classes[final_mask] = cls
        any_applied = True
        reasons.append(f"instance{cl_lbl}_conf_{bbox_info['confidence']:.2f}")

    if not any_applied:
        return predicted_classes, False, "fusion_too_few_points"

    return corrected_classes, True, ";".join(reasons)


# =====================================================
# STEP 5: MAHALANOBIS + RELABEL ENGINE
# =====================================================

def mahalanobis_distance(x: np.ndarray, mu: np.ndarray, sigma_inv: np.ndarray) -> float:
    diff = x - mu
    return float(np.sqrt(diff @ sigma_inv @ diff.T))


def verify_instance(feat: dict, predicted_cls: str, joint_prototype: dict):
    if predicted_cls not in joint_prototype:
        return None, None, None

    proto = joint_prototype[predicted_cls]
    feats = proto["features"]

    try:
        x = np.array([feat[f] for f in feats], dtype=float)
    except KeyError:
        return None, None, None

    if not np.all(np.isfinite(x)):
        return None, None, None

    d = mahalanobis_distance(x, proto["mu"], proto["sigma_inv"])
    k = len(feats)
    p_value = float(1 - chi2.cdf(d ** 2, df=k))

    return d, p_value, k


def error_mining_threshold(n_dims: int, confidence: float = CHI2_CONFIDENCE) -> float:
    return float(np.sqrt(chi2.ppf(confidence, df=n_dims)))


def compute_class_scores(feat: dict, joint_prototype: dict) -> dict:
    scores = {}
    for cls in joint_prototype.keys():
        d, p_value, k = verify_instance(feat, cls, joint_prototype)
        if p_value is None:
            continue
        scores[cls] = p_value
    return scores


def spatial_prior(feat: dict, cls: str) -> float:
    score = 1.0

    if cls == "window":
        score *= 1.5 if feat.get("DistanceToWall", np.inf) < 0.2 else 0.3
    elif cls == "chair":
        score *= 1.5 if feat.get("DistanceToFloor", np.inf) < 0.3 else 0.5
    elif cls == "wall":
        score *= 1.5 if feat.get("Planarity", 0) > 0.6 else 0.3
    elif cls == "ceiling":
        score *= 1.5 if feat.get("HeightClearance", 0) > 2.5 else 0.4

    return score


def final_scores(feat: dict, joint_prototype: dict) -> dict:
    proto_scores = compute_class_scores(feat, joint_prototype)
    return {cls: s * spatial_prior(feat, cls) for cls, s in proto_scores.items()}


def relabel_instance(feat: dict, predicted_cls: str, joint_prototype: dict,
                     min_margin: float = 1.3):
    scores = final_scores(feat, joint_prototype)
    if not scores:
        return predicted_cls, False

    best_cls = max(scores, key=scores.get)
    best_score = scores[best_cls]

    sorted_scores = sorted(scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0

    confidence_gap = best_score / (second_best + 1e-9)

    if best_cls != predicted_cls and confidence_gap > min_margin:
        return best_cls, True

    return predicted_cls, False


# =====================================================
# STEP 6: EXPORT CORRECTED PLY
# =====================================================

def export_corrected_ply(points: np.ndarray, corrected_classes: np.ndarray, room_name: str) -> Path:
    import open3d as o3d

    colors = np.full((len(points), 3), 0.2)
    for cls, rgb in CLASS_COLORS_CONSISTENT.items():
        mask = (corrected_classes == cls)
        if np.any(mask):
            colors[mask] = rgb

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    out_path = OUTPUT_DIR / f"{room_name}_corrected.ply"
    o3d.io.write_point_cloud(str(out_path), pcd)
    return out_path


# =====================================================
# STEP 7: MAIN PIPELINE
# =====================================================

import csv
import gc

# ROOM_PLY_PATHS = [
#     Path(r"D:\Thu\KhoaLuan\conferenceRoom_1_mlp_refined.ply"),
# ]

GEOMETRY_FEATURE_COLS = [
    "NumPoints", "SurfaceArea", "Volume", "Height", "Width", "Depth",
    "HeightAboveFloor", "HeightClearance", "Planarity", "Linearity",
    "Roughness", "Curvature", "NormalEntropy", "NormalVariance",
    "Compactness", "Elongation", "Sphericity",
]
SPATIAL_FEATURE_COLS = [
    "DistanceToWall", "DistanceToFloor", "DistanceToCeiling",
    "NormalAlignmentWall", "NormalAlignmentFloor", "NormalAlignmentCeiling",
]
META_COLS = [
    "RoomName", "InstanceID", "PredictedClass", "FinalClass", "Relabeled", "RelabelSuggested",
    "NumPoints", "MahalanobisDistance", "PValue", "NDims", "Threshold", "Suspect",
    "Avg2DSimilarity", "DecisionReason",
]

ALL_COLS = META_COLS + [c for c in (GEOMETRY_FEATURE_COLS + SPATIAL_FEATURE_COLS) if c != "NumPoints"]

TEST_OUTPUT_DIR = Path(r"/compute_home/slurmdang12/datasets/MLPpredictions")

def resolve_room_paths():
    """
    Tìm tất cả file PLY trong MLPpredictions directory
    Nếu không tìm thấy hoặc directory không tồn tại, return empty list
    """
    if not TEST_OUTPUT_DIR.exists():
        print(f"   ⚠️  MLPpredictions directory không tồn tại: {TEST_OUTPUT_DIR}")
        print(f"      Sẽ chỉ xử lý các room từ checkpoint nếu có")
        return []
    
    paths = sorted(TEST_OUTPUT_DIR.glob("*.ply"))
    
    if not paths:
        print(f"   ⚠️  Không tìm thấy file PLY nào trong {TEST_OUTPUT_DIR}")
        print(f"      Hãy kiểm tra lại đường dẫn hoặc tên file")
        return []
    
    print(f"   ✅ Tìm thấy {len(paths)} file PLY")
    for p in paths:
        print(f"      - {p.name}")
    
    return paths


def _dedupe_by_confidence(points, labels, confs, decimals=4):
    keys = [tuple(p) for p in np.round(points, decimals)]
    best = {}
    for i, k in enumerate(keys):
        if k not in best or confs[i] > best[k][0]:
            best[k] = (confs[i], i)

    keep_idx = np.array([v[1] for v in best.values()])
    return points[keep_idx], labels[keep_idx], confs[keep_idx]

def _dedupe_by_confidence_batch(points, labels, confs, decimals=4, batch_size=100000):
    """
    Khử trùng theo batch để tránh memory spike
    """
    from collections import defaultdict
    
    n = len(points)
    unique_dict = {}
    
    print(f"   🔄 Deduplicating {n:,} points in batches of {batch_size:,}...")
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_points = points[start:end]
        batch_confs = confs[start:end]
        
        for i in range(len(batch_points)):
            key = tuple(np.round(batch_points[i], decimals))
            if key not in unique_dict or batch_confs[i] > unique_dict[key][0]:
                unique_dict[key] = (batch_confs[i], start + i)
        
        # Giải phóng batch
        del batch_points, batch_confs
        gc.collect()
        
        if (start // batch_size) % 5 == 0:
            print(f"      Processed {min(end, n):,}/{n:,} points...")
    
    keep_idx = np.array([v[1] for v in unique_dict.values()])
    print(f"   -> Kept {len(keep_idx):,} unique points")
    
    return points[keep_idx], labels[keep_idx], confs[keep_idx]
    
def load_2d_data(room_name: str):
    """
    Load dữ liệu 2D - tối ưu memory cho các room lớn (hallway)
    """
    import gc
    import tempfile
    import shutil
    from collections import defaultdict
    
    room_name = room_name.replace("_prediction", "")
    
    search_dirs = [
        Path(r"/compute_home/slurmdang12/datasets/mapped_pcd/Area_5"),
        # Path(r"/compute_home/slurmdang12/datasets/mapped_pcd/Area_1"),
        # Path(r"/compute_home/slurmdang12/datasets/mapped_pcd"),
    ]

    all_segments = []
    found_any = False

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        
        print(f"   🔍 Đang tìm 2D trong: {base_dir}")

        # CÁCH 1: Thư mục trực tiếp
        direct_dir = base_dir / room_name
        if direct_dir.exists() and direct_dir.is_dir():
            print(f"   📁 Kiểm tra thư mục trực tiếp: {direct_dir}")
            
            points_file = direct_dir / f"{room_name}_points.npy"
            labels_file = direct_dir / f"{room_name}_labels.npy"
            confs_file = direct_dir / f"{room_name}_confidences.npy"
            
            if not points_file.exists():
                points_candidates = list(direct_dir.glob("*points*.npy"))
                if points_candidates:
                    points_file = points_candidates[0]
                    base_name = points_file.stem.replace('_points', '')
                    labels_file = direct_dir / f"{base_name}_labels.npy"
                    confs_file = direct_dir / f"{base_name}_confidences.npy"
            
            if points_file.exists() and labels_file.exists() and confs_file.exists():
                print(f"   ✅ Tìm thấy dữ liệu 2D trong {direct_dir}")
                # Load với allow_pickle=True vì file có object data
                pts = np.load(points_file, allow_pickle=True)
                if pts.shape[1] > 3:
                    pts = pts[:, :3]
                return {
                    'points': pts,
                    'labels': np.load(labels_file, allow_pickle=True),
                    'confidences': np.load(confs_file, allow_pickle=True),
                }
        
        # CÁCH 2: Tìm các segment
        segment_dirs = sorted(base_dir.glob(f"{room_name}_segment_*"))
        if segment_dirs:
            print(f"   📁 Tìm thấy {len(segment_dirs)} segment(s)")
            for seg_dir in segment_dirs:
                seg_name = seg_dir.name
                points_file = seg_dir / f"{seg_name}_points.npy"
                labels_file = seg_dir / f"{seg_name}_labels.npy"
                confs_file = seg_dir / f"{seg_name}_confidences.npy"
                
                if points_file.exists() and labels_file.exists() and confs_file.exists():
                    all_segments.append({
                        'points_file': points_file,
                        'labels_file': labels_file,
                        'confs_file': confs_file,
                        'name': seg_name
                    })
                    found_any = True
                    print(f"   ✅ Tìm thấy segment: {seg_name}")
                else:
                    print(f"   ⚠️  Thiếu file trong {seg_dir}")
        
        if found_any:
            break

    if not found_any or not all_segments:
        warnings.warn(f"Không tìm thấy file 2D cho {room_name}")
        return None

    print(f"   🔄 Đang gộp {len(all_segments)} segments...")

    # ============================================================
    # PHẦN GỘP SEGMENTS - TỐI ƯU MEMORY
    # ============================================================
    
    # BƯỚC 1: Đếm tổng số points (load bình thường, không dùng mmap)
    total_points = 0
    for seg in all_segments:
        pts = np.load(seg['points_file'], allow_pickle=True)
        total_points += pts.shape[0]
        del pts
        gc.collect()
    
    print(f"   Total points before dedup: {total_points:,}")
    
    # Nếu quá lớn, giảm số points
    MAX_POINTS = 5_000_000  # 5 triệu points tối đa
    sample_rate = 1.0
    if total_points > MAX_POINTS:
        sample_rate = MAX_POINTS / total_points
        print(f"   ⚠️  Quá lớn! Sampling {sample_rate*100:.1f}% points...")
    
    # BƯỚC 2: Load và gộp từng segment (không dùng memmap để tránh lỗi pickle)
    all_points = []
    all_labels = []
    all_confs = []
    
    for seg in all_segments:
        pts = np.load(seg['points_file'], allow_pickle=True)
        if pts.shape[1] > 3:
            pts = pts[:, :3]
        lbl = np.load(seg['labels_file'], allow_pickle=True)
        cf = np.load(seg['confs_file'], allow_pickle=True)
        
        n = pts.shape[0]
        
        # Sampling nếu cần
        if sample_rate < 1.0:
            indices = np.random.choice(n, int(n * sample_rate), replace=False)
            pts = pts[indices]
            lbl = lbl[indices]
            cf = cf[indices]
        
        all_points.append(pts)
        all_labels.append(lbl)
        all_confs.append(cf)
        
        # Giải phóng ngay
        gc.collect()
    
    # BƯỚC 3: Gộp tất cả
    print(f"   🔄 Concatenating...")
    merged_points = np.concatenate(all_points, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_confs = np.concatenate(all_confs, axis=0)
    
    # Giải phóng list
    del all_points, all_labels, all_confs
    gc.collect()
    
    # BƯỚC 4: Khử trùng
    if len(all_segments) > 1:
        print(f"   🔄 Deduplicating {len(merged_points):,} points...")
        merged_points, merged_labels, merged_confs = _dedupe_by_confidence_batch(
            merged_points, merged_labels, merged_confs
        )
        print(f"   -> Sau khử trùng: {len(merged_points):,} points")
    
    print(f"   ✅ Load thành công: {len(merged_points):,} points 2D")
    
    return {
        'points': merged_points,
        'labels': merged_labels,
        'confidences': merged_confs,
    }

def map_2d_labels_to_ply(ply_points: np.ndarray,
                         mapped_data: dict) -> tuple:
    """
    Map nhãn 2D lên PLY points - tối ưu cho số lượng lớn
    """
    mapped_points = mapped_data['points']
    mapped_labels = mapped_data['labels']
    mapped_confs = mapped_data['confidences']

    n_ply = len(ply_points)
    n_mapped = len(mapped_points)

    # Kiểm tra số chiều
    if mapped_points.shape[1] > 3:
        print(f"   ⚠️  Mapped points có {mapped_points.shape[1]} chiều, chỉ lấy 3 chiều đầu (x,y,z)")
        mapped_points = mapped_points[:, :3]

    # Nếu số lượng bằng nhau, copy trực tiếp
    if n_ply == n_mapped:
        labels_2d = mapped_labels.copy()
        confs_2d = mapped_confs.copy()
        return labels_2d, confs_2d

    # Nếu khác nhau, dùng KDTree
    print(f"   🔄 Mapping {n_ply:,} PLY points to {n_mapped:,} 2D points...")
    
    from scipy.spatial import cKDTree
    import gc
    
    # Xây dựng KDTree
    tree = cKDTree(mapped_points)
    
    # Query theo batch để tránh memory spike
    BATCH_SIZE = 200000
    labels_2d = np.full(n_ply, -1, dtype=np.int64)
    confs_2d = np.full(n_ply, 0.0, dtype=np.float32)
    
    for start in range(0, n_ply, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_ply)
        batch_points = ply_points[start:end]
        
        distances, indices = tree.query(batch_points, k=1)
        valid = distances < 0.1
        
        labels_2d[start:end][valid] = mapped_labels[indices[valid]]
        confs_2d[start:end][valid] = mapped_confs[indices[valid]]
        
        # Giải phóng batch
        del batch_points, distances, indices, valid
        if start % (BATCH_SIZE * 2) == 0:
            gc.collect()
    
    # Giải phóng tree
    del tree
    gc.collect()
    
    print(f"   ✅ Mapped: {np.sum(labels_2d >= 0):,} points có nhãn")
    
    return labels_2d, confs_2d


# =====================================================
# PHẦN POST-PROCESSING (SỬA 3 LỖI)
# =====================================================

def _fit_plane_ransac(points: np.ndarray,
                      distance_threshold: float = RANSAC_PLANE_DISTANCE_THRESHOLD,
                      num_iterations: int = RANSAC_PLANE_NUM_ITERATIONS):
    n = len(points)
    if n < RANSAC_PLANE_MIN_POINTS:
        return None, None

    rng = np.random.default_rng(42)
    best_inlier_count = -1
    best_normal = None
    best_inlier_mask = None

    for _ in range(num_iterations):
        sample_idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[sample_idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-9:
            continue
        normal = normal / norm_len

        dist = np.abs((points - p0) @ normal)
        inlier_mask = dist <= distance_threshold
        inlier_count = int(np.sum(inlier_mask))

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_inlier_mask = inlier_mask

    if best_normal is None:
        return None, None

    inlier_ratio = best_inlier_count / n
    if inlier_ratio < RANSAC_PLANE_MIN_INLIER_RATIO:
        return None, None

    inlier_pts = points[best_inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    cov = np.cov((inlier_pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    refined_normal = eigvecs[:, np.argmin(eigvals)]

    dist_refined = np.abs((points - centroid) @ refined_normal)
    refined_inlier_mask = dist_refined <= distance_threshold

    return refined_normal, refined_inlier_mask


def _determine_fixed_axis(original_mlp_mask_instance: np.ndarray,
                          points: np.ndarray):
    pts_mlp_orig = points[original_mlp_mask_instance]
    if len(pts_mlp_orig) < RANSAC_PLANE_MIN_POINTS:
        return None, None, None, None, None, None

    normal, inlier_mask = _fit_plane_ransac(pts_mlp_orig)
    if normal is None or inlier_mask is None:
        return None, None, None, None, None, None

    n_inlier = int(np.sum(inlier_mask))
    print(f"    [determine_axis] RANSAC: {n_inlier}/{len(pts_mlp_orig)} điểm là inlier")

    inlier_pts = pts_mlp_orig[inlier_mask]
    mlp_mins = inlier_pts.min(axis=0)
    mlp_maxs = inlier_pts.max(axis=0)

    centroid = inlier_pts.mean(axis=0)
    dist_inlier = (inlier_pts - centroid) @ normal
    plane_std = float(np.std(dist_inlier))
    plane_thickness_threshold = max(
        PLANE_THICKNESS_STD_MULTIPLIER * plane_std,
        PLANE_THICKNESS_MIN_THRESHOLD,
    )
    print(f"    [determine_axis] Độ dày mặt phẳng: std={plane_std*100:.1f}cm "
          f"-> ngưỡng loại = {plane_thickness_threshold*100:.1f}cm")

    abs_nx, abs_ny = abs(float(normal[0])), abs(float(normal[1]))
    abs_nx_safe = max(abs_nx, 1e-9)
    abs_ny_safe = max(abs_ny, 1e-9)

    x_dominant = abs_nx / abs_ny_safe >= RANSAC_NORMAL_AXIS_DOMINANCE_RATIO
    y_dominant = abs_ny / abs_nx_safe >= RANSAC_NORMAL_AXIS_DOMINANCE_RATIO

    if x_dominant and not y_dominant:
        print(f"    [determine_axis] trục cố định = X")
        return "x", mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold
    if y_dominant and not x_dominant:
        print(f"    [determine_axis] trục cố định = Y")
        return "y", mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold

    print(f"    [determine_axis] KHÔNG áp clamp - không trục nào chiếm ưu thế")
    return None, mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold


def _wall_points_on_same_plane(room_wall_points, normal, centroid, low, high, axis_fixed,
                               distance_threshold=RANSAC_PLANE_DISTANCE_THRESHOLD):
    if room_wall_points is None or len(room_wall_points) == 0:
        return None
    dist_to_plane = np.abs((room_wall_points - centroid) @ normal)
    coplanar = dist_to_plane <= distance_threshold
    if axis_fixed == "x":
        in_range = (room_wall_points[:, 1] >= low) & (room_wall_points[:, 1] <= high)
    else:
        in_range = (room_wall_points[:, 0] >= low) & (room_wall_points[:, 0] <= high)
    mask = coplanar & in_range
    return room_wall_points[mask] if np.any(mask) else None


def extend_planar_instance_z_to_wall_gap(points, corrected_classes, original_mlp_classes,
                                         cls, instance_point_idx, room_wall_points,
                                         axis_fixed, mlp_mins, mlp_maxs, normal, centroid,
                                         plane_thickness_threshold):
    empty_extra = np.array([], dtype=int)
    if len(instance_point_idx) == 0 or room_wall_points is None:
        return corrected_classes, empty_extra

    inst_points = points[instance_point_idx]
    if axis_fixed == "x":
        low, high = mlp_mins[1] - PLANAR_CLAMP_MARGIN, mlp_maxs[1] + PLANAR_CLAMP_MARGIN
    else:
        low, high = mlp_mins[0] - PLANAR_CLAMP_MARGIN, mlp_maxs[0] + PLANAR_CLAMP_MARGIN

    wall_pts = _wall_points_on_same_plane(room_wall_points, normal, centroid, low, high, axis_fixed)
    if wall_pts is None or len(wall_pts) < 30:
        print(f"    [extend_wall_gap] '{cls}': không đủ wall_points cùng mặt phẳng -> bỏ qua")
        return corrected_classes, empty_extra

    z_wall = wall_pts[:, 2]
    z_inst_min, z_inst_max = float(inst_points[:, 2].min()), float(inst_points[:, 2].max())

    far_above = z_wall > (z_inst_max + WALL_GAP_SOLID_REFERENCE_MARGIN)
    far_below = z_wall < (z_inst_min - WALL_GAP_SOLID_REFERENCE_MARGIN)
    ref_z = z_wall[far_above | far_below]
    if len(ref_z) < 10:
        print(f"    [extend_wall_gap] '{cls}': không đủ điểm wall tham chiếu -> bỏ qua")
        return corrected_classes, empty_extra

    n_ref_bins = max(1, int((ref_z.max() - ref_z.min()) / WALL_GAP_Z_BIN_SIZE))
    solid_density = len(ref_z) / n_ref_bins
    if solid_density < 1e-6:
        return corrected_classes, empty_extra
    density_threshold = solid_density * WALL_GAP_DENSITY_RATIO_THRESHOLD

    step = WALL_GAP_Z_BIN_SIZE
    z_max_new, extended_up = z_inst_max, 0.0
    while extended_up < WALL_GAP_MAX_EXTENSION:
        bin_count = int(np.sum((z_wall >= z_max_new) & (z_wall < z_max_new + step)))
        if bin_count >= density_threshold:
            break
        z_max_new += step
        extended_up += step

    z_min_new, extended_down = z_inst_min, 0.0
    while extended_down < WALL_GAP_MAX_EXTENSION:
        bin_count = int(np.sum((z_wall >= z_min_new - step) & (z_wall < z_min_new)))
        if bin_count >= density_threshold:
            break
        z_min_new -= step
        extended_down += step

    if z_max_new <= z_inst_max + 1e-6 and z_min_new >= z_inst_min - 1e-6:
        print(f"    [extend_wall_gap] '{cls}': không cần mở rộng")
        return corrected_classes, empty_extra

    print(f"    [extend_wall_gap] '{cls}': mở rộng Z [{z_inst_min:.2f},{z_inst_max:.2f}] "
          f"-> [{z_min_new:.2f},{z_max_new:.2f}] (mật độ tường={solid_density:.1f}/bin)")

    if axis_fixed == "x":
        in_horiz = (points[:, 1] >= low) & (points[:, 1] <= high)
    else:
        in_horiz = (points[:, 0] >= low) & (points[:, 0] <= high)
    in_ext_z = (points[:, 2] >= z_min_new) & (points[:, 2] <= z_max_new)
    on_plane = np.abs((points - centroid) @ normal) <= plane_thickness_threshold
    protected_or_locked = np.isin(original_mlp_classes,
                                   list(FURNITURE_CLASSES_PROTECTED) + list(LOCKED_CLASSES))
    candidate_mask = in_horiz & in_ext_z & on_plane & (~protected_or_locked)

    corrected_classes = corrected_classes.copy()
    extra_idx = np.where(candidate_mask & (corrected_classes != cls))[0]
    corrected_classes[candidate_mask] = cls
    if len(extra_idx) > 0:
        print(f"    [extend_wall_gap] '{cls}': lấp thêm {len(extra_idx)} điểm")

    return corrected_classes, extra_idx


def clamp_planar_instance_no_horizontal_overflow(points: np.ndarray,
                                                 corrected_classes: np.ndarray,
                                                 original_mlp_classes: np.ndarray,
                                                 cls: str,
                                                 instance_point_idx: np.ndarray):
    if len(instance_point_idx) == 0:
        return corrected_classes, None, None, None, None, None, None

    inst_points = points[instance_point_idx]
    inst_mins = inst_points.min(axis=0)
    inst_maxs = inst_points.max(axis=0)

    near_margin = 0.3
    near_bbox = (
        (points[:, 0] >= inst_mins[0] - near_margin) & (points[:, 0] <= inst_maxs[0] + near_margin) &
        (points[:, 1] >= inst_mins[1] - near_margin) & (points[:, 1] <= inst_maxs[1] + near_margin)
    )
    original_mlp_mask_instance = (original_mlp_classes == cls) & near_bbox

    axis_fixed, mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold = \
        _determine_fixed_axis(original_mlp_mask_instance, points)

    if axis_fixed is None or normal is None:
        print(f"    [clamp_planar] '{cls}' instance: không xác định được trục cố định -> KHÔNG áp clamp")
        return corrected_classes, None, None, None, None, None, None

    corrected_classes = corrected_classes.copy()
    current_idx = instance_point_idx
    current_points = inst_points

    # LỚP 1: chặn ngang
    margin = PLANAR_CLAMP_MARGIN
    if axis_fixed == "x":
        y_low, y_high = mlp_mins[1] - margin, mlp_maxs[1] + margin
        out_of_bound = (current_points[:, 1] < y_low) | (current_points[:, 1] > y_high)
        print(f"    [clamp_planar][Lớp 1] chặn Y trong [{y_low:.2f}, {y_high:.2f}]")
    else:
        x_low, x_high = mlp_mins[0] - margin, mlp_maxs[0] + margin
        out_of_bound = (current_points[:, 0] < x_low) | (current_points[:, 0] > x_high)
        print(f"    [clamp_planar][Lớp 1] chặn X trong [{x_low:.2f}, {x_high:.2f}]")

    n_out_l1 = int(np.sum(out_of_bound))
    if n_out_l1 > 0:
        out_global_idx = current_idx[out_of_bound]
        corrected_classes[out_global_idx] = original_mlp_classes[out_global_idx]

    keep_l1 = ~out_of_bound
    current_idx = current_idx[keep_l1]
    current_points = current_points[keep_l1]

    if len(current_idx) == 0:
        return corrected_classes, axis_fixed, mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold

    # LỚP 2: loại điểm nhô ra khỏi mặt phẳng
    dist_to_plane = np.abs((current_points - centroid) @ normal)
    protrude_mask = dist_to_plane > plane_thickness_threshold

    n_out_l2 = int(np.sum(protrude_mask))
    if n_out_l2 > 0:
        protrude_global_idx = current_idx[protrude_mask]
        corrected_classes[protrude_global_idx] = original_mlp_classes[protrude_global_idx]
        print(f"    [clamp_planar][Lớp 2] loại {n_out_l2} điểm nhô ra khỏi mặt phẳng")

    return corrected_classes, axis_fixed, mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold


def fill_unmatched_subcluster_with_original_mlp(points: np.ndarray,
                                                corrected_classes: np.ndarray,
                                                original_mlp_classes: np.ndarray,
                                                joint_prototype: dict,
                                                cls: str,
                                                instance_point_idx: np.ndarray) -> np.ndarray:
    if len(instance_point_idx) < SUBCLUSTER_MIN_POINTS_TO_CHECK:
        return corrected_classes

    inst_points = points[instance_point_idx]

    try:
        sub_labels = DBSCAN(
            eps=SUBCLUSTER_DBSCAN_EPS,
            min_samples=SUBCLUSTER_DBSCAN_MIN_SAMPLES,
            n_jobs=-1,
        ).fit(inst_points).labels_
    except Exception:
        return corrected_classes

    unique_sub = sorted(set(sub_labels) - {-1})
    if len(unique_sub) <= 1:
        return corrected_classes

    corrected_classes = corrected_classes.copy()

    for sub_lbl in unique_sub:
        sub_mask = (sub_labels == sub_lbl)
        n_sub = int(np.sum(sub_mask))
        if n_sub < SUBCLUSTER_MIN_POINTS_TO_CHECK:
            continue

        sub_global_idx = instance_point_idx[sub_mask]
        sub_pts = points[sub_global_idx]

        geom = extract_geometry(sub_pts)
        d, p_value, k = verify_instance(geom, cls, joint_prototype)
        if d is None or k is None:
            continue

        threshold = error_mining_threshold(k)
        is_suspect = d > threshold

        if is_suspect:
            corrected_classes[sub_global_idx] = original_mlp_classes[sub_global_idx]
            print(f"    [subcluster_check] '{cls}' sub-cluster ({n_sub} điểm): "
                  f"d={d:.2f} > threshold={threshold:.2f} -> trả về MLP gốc")

    return corrected_classes


def fix_ceiling_floor_overflow_by_z(points: np.ndarray,
                                    corrected_classes: np.ndarray,
                                    original_mlp_classes: np.ndarray,
                                    room_floor_z,
                                    room_ceiling_z) -> np.ndarray:
    """
    [ĐÃ SỬA - LỖI #2] Lớp 1 dùng RANSAC plane-distance thay vì |z deviation|
    """
    result = corrected_classes.copy()

    for cls in CEILING_FLOOR_Z_CHECK_CLASSES:
        room_z_ref = room_ceiling_z if cls == "ceiling" else room_floor_z
        if room_z_ref is None:
            continue

        cls_mask = (result == cls)
        n_cls_points_before = int(np.sum(cls_mask))
        if n_cls_points_before < CEILING_FLOOR_DBSCAN_MIN_SAMPLES:
            continue

        cls_indices_before = np.where(cls_mask)[0]
        cls_points_before = points[cls_indices_before]

        # LỚP 1: RANSAC plane-distance
        normal, inlier_mask = _fit_plane_ransac(cls_points_before)

        if normal is not None and inlier_mask is not None:
            centroid_plane = cls_points_before[inlier_mask].mean(axis=0)
            dist_to_plane = np.abs((cls_points_before - centroid_plane) @ normal)
            per_point_out = dist_to_plane > CEILING_FLOOR_PERPOINT_MAX_Z_DEVIATION
            print(f"\nClass '{cls}': LỚP 1 (RANSAC plane-distance, "
                  f"ngưỡng {CEILING_FLOOR_PERPOINT_MAX_Z_DEVIATION*100:.0f}cm, "
                  f"{int(np.sum(inlier_mask))}/{len(cls_points_before)} điểm là inlier)")
        else:
            # Fallback
            z_vals_all = cls_points_before[:, 2]
            per_point_out = np.abs(z_vals_all - room_z_ref) > CEILING_FLOOR_PERPOINT_MAX_Z_DEVIATION
            print(f"\nClass '{cls}': LỚP 1 (fallback |z-z_chuẩn|, "
                  f"RANSAC không tìm được mặt phẳng đủ tin cậy)")

        n_out_l1 = int(np.sum(per_point_out))
        if n_out_l1 > 0:
            out_idx_l1 = cls_indices_before[per_point_out]
            result[out_idx_l1] = original_mlp_classes[out_idx_l1]
            print(f"    [Lớp 1] '{cls}': loại {n_out_l1} điểm lệch khỏi mặt phẳng")
        else:
            print(f"    [Lớp 1] '{cls}': không có điểm nào vượt ngưỡng")

        # LỚP 2: DBSCAN trên phần còn lại
        cls_mask_after_l1 = (result == cls)
        n_cls_points_after_l1 = int(np.sum(cls_mask_after_l1))
        if n_cls_points_after_l1 < CEILING_FLOOR_DBSCAN_MIN_SAMPLES:
            continue

        cls_indices = np.where(cls_mask_after_l1)[0]
        cls_points = points[cls_indices]

        try:
            instance_labels = DBSCAN(
                eps=CEILING_FLOOR_DBSCAN_EPS,
                min_samples=CEILING_FLOOR_DBSCAN_MIN_SAMPLES,
                n_jobs=-1,
            ).fit(cls_points).labels_
        except Exception:
            continue

        unique_inst = sorted(set(instance_labels) - {-1})
        if not unique_inst:
            continue

        for inst_lbl in unique_inst:
            inst_mask_local = (instance_labels == inst_lbl)
            instance_point_idx = cls_indices[inst_mask_local]
            inst_pts = points[instance_point_idx]

            z_vals = inst_pts[:, 2]
            z_median = float(np.median(z_vals))
            z_thickness = float(z_vals.max() - z_vals.min())
            z_deviation = abs(z_median - room_z_ref)

            is_z_off = z_deviation > CEILING_FLOOR_MAX_Z_DEVIATION
            is_too_thick = z_thickness > CEILING_FLOOR_MAX_THICKNESS

            if is_z_off or is_too_thick:
                result[instance_point_idx] = original_mlp_classes[instance_point_idx]
                reasons = []
                if is_z_off:
                    reasons.append(f"lệch median Z {z_deviation*100:.1f}cm")
                if is_too_thick:
                    reasons.append(f"dày {z_thickness*100:.1f}cm")
                print(f"    [Lớp 2] '{cls}' cụm ({len(instance_point_idx)} điểm): "
                      f"{'; '.join(reasons)} -> loại")

    return result


def postprocess_protect_furniture_and_clamp_planar(points: np.ndarray,
                                                   corrected_classes: np.ndarray,
                                                   original_mlp_classes: np.ndarray,
                                                   joint_prototype: dict,
                                                   room_name: str,
                                                   room_floor_z=None,
                                                   room_ceiling_z=None,
                                                   room_wall_points=None) -> np.ndarray:
    print(f"\n{'-'*60}")
    print(f"[POST-PROCESSING] Room '{room_name}' - SỬA 3 LỖI")
    print(f"{'-'*60}")

    result = corrected_classes.copy()

    # (4) Rà soát ceiling/floor - ĐÃ SỬA LỖI #2
    result = fix_ceiling_floor_overflow_by_z(
        points, result, original_mlp_classes, room_floor_z, room_ceiling_z
    )

    # (2) + (3): xử lý window/whiteboard
    for cls in PLANAR_CLASSES_CLAMP_Z_ONLY:
        cls_mask = (result == cls)
        n_cls_points = int(np.sum(cls_mask))
        if n_cls_points < SUBCLUSTER_MIN_POINTS_TO_CHECK:
            continue

        cls_indices = np.where(cls_mask)[0]
        cls_points = points[cls_indices]

        try:
            instance_labels = DBSCAN(
                eps=DBSCAN_EPS,
                min_samples=DBSCAN_MIN_SAMPLES,
                n_jobs=-1,
            ).fit(cls_points).labels_
        except Exception:
            instance_labels = np.zeros(len(cls_points), dtype=int)

        unique_inst = sorted(set(instance_labels) - {-1})
        if not unique_inst:
            unique_inst = [0]
            instance_labels = np.zeros(len(cls_points), dtype=int)

        print(f"\nClass '{cls}': {n_cls_points} điểm -> {len(unique_inst)} instance")

        for inst_lbl in unique_inst:
            inst_mask_local = (instance_labels == inst_lbl)
            if np.sum(inst_mask_local) < SUBCLUSTER_MIN_POINTS_TO_CHECK:
                continue
            instance_point_idx = cls_indices[inst_mask_local]

            # (2) Clamp ngang + lấy thông tin mặt phẳng
            result, axis_fixed, mlp_mins, mlp_maxs, normal, centroid, plane_thickness_threshold = \
                clamp_planar_instance_no_horizontal_overflow(
                    points, result, original_mlp_classes, cls, instance_point_idx
                )

            still_cls_mask = (result[instance_point_idx] == cls)
            instance_point_idx_remaining = instance_point_idx[still_cls_mask]

            # (3) Mở rộng Z theo lỗ hổng trên tường - SỬA LỖI #3
            if axis_fixed is not None:
                result, extra_idx = extend_planar_instance_z_to_wall_gap(
                    points, result, original_mlp_classes, cls,
                    instance_point_idx_remaining, room_wall_points,
                    axis_fixed, mlp_mins, mlp_maxs, normal, centroid,
                    plane_thickness_threshold,
                )
                if len(extra_idx) > 0:
                    instance_point_idx_remaining = np.union1d(instance_point_idx_remaining, extra_idx)

            # (3b) Kiểm tra sub-cluster không khớp prototype
            result = fill_unmatched_subcluster_with_original_mlp(
                points, result, original_mlp_classes, joint_prototype,
                cls, instance_point_idx_remaining
            )

    # (1) Lưới an toàn furniture - LUÔN chạy sau cùng
    furniture_mask = np.isin(original_mlp_classes, list(FURNITURE_CLASSES_PROTECTED))
    n_furniture = int(np.sum(furniture_mask))
    if n_furniture > 0:
        n_would_change = int(np.sum(furniture_mask & (result != original_mlp_classes)))
        result[furniture_mask] = original_mlp_classes[furniture_mask]
        if n_would_change > 0:
            print(f"\n[FURNITURE GUARD] Đã chặn {n_would_change} điểm furniture bị ghi đè")

    return result


def process_one_room(ply_path: Path, joint_prototype: dict, writer: csv.DictWriter, 
                     resume: bool = True):
    """
    Xử lý 1 room PLY với tối ưu memory và checkpoint
    """
    import gc
    import psutil
    import os
    
    def get_memory():
        """Lấy memory usage hiện tại (MB)"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    room_name = ply_path.stem.replace('_mlp_refined', '')
    print(f"\n{'='*60}\nROOM: {room_name}\n{'='*60}")
    print(f"📊 Memory before: {get_memory():.1f} MB")

    # ===== KIỂM TRA CHECKPOINT =====
    if resume:
        checkpoint = load_checkpoint(room_name)
        if checkpoint is not None:
            print(f"   🔄 Resuming from checkpoint for {room_name}")
            return checkpoint.get('total_instances', 0), 0, 0, True

    # ============================================================
    # BƯỚC 1: LOAD PLY
    # ============================================================
    print(f"\n[1] Loading PLY...")
    points, colors = load_ply(ply_path)
    print(f"   -> {len(points):,} points")
    
    # Map màu sang class
    point_classes = color_to_class_batch(colors, CLASS_COLORS_CONSISTENT)
    n_unmatched = int(np.sum(pd.isna(point_classes)))
    if n_unmatched > 0:
        warnings.warn(f"[{room_name}] {n_unmatched} điểm không khớp màu")

    original_mlp_classes = point_classes.copy()
    del colors
    gc.collect()
    print(f"📊 Memory after PLY: {get_memory():.1f} MB")

    # ============================================================
    # BƯỚC 2: LOAD 2D DATA (CÓ CACHE)
    # ============================================================
    print(f"\n[2] Loading 2D data...")
    mapped_data = load_2d_data(room_name)
    
    if mapped_data is None:
        warnings.warn(f"Không có dữ liệu 2D cho {room_name}")
        labels_2d = np.full(len(points), -1, dtype=np.int64)
        confs_2d = np.full(len(points), 0.0, dtype=np.float32)
        has_2d = False
    else:
        labels_2d, confs_2d = map_2d_labels_to_ply(points, mapped_data)
        n_labeled = np.sum(labels_2d >= 0)
        print(f"   -> 2D labels: {n_labeled:,} điểm có nhãn ({n_labeled/len(points)*100:.1f}%)")
        has_2d = True
        del mapped_data
        gc.collect()
    
    labels_2d_names = np.array([label_to_class_name(lbl) for lbl in labels_2d], dtype=object)
    print(f"📊 Memory after 2D: {get_memory():.1f} MB")

    # ============================================================
    # BƯỚC 3: ROOM REFERENCE
    # ============================================================
    print(f"\n[3] Computing room reference...")
    room_geom_ref = geometric_room_reference(points)
    floor_z = room_geom_ref["floor_z"]
    ceiling_z = room_geom_ref["ceiling_z"]
    print(f"   -> floor_z={floor_z:.3f}, ceiling_z={ceiling_z:.3f}")

    # ============================================================
    # BƯỚC 4: FUSION 2D + MLP
    # ============================================================
    if has_2d:
        print(f"\n[4] Applying fusion...")
        for cls in CLASSES_2D_TRUSTED:
            cls_label_idx = LABEL2D_CLASS_ORDER.index(cls) if cls in LABEL2D_CLASS_ORDER else -1
            if cls_label_idx >= 0:
                mask_2d_check = (labels_2d == cls_label_idx)
                if np.sum(mask_2d_check) > 10:
                    print(f"   -> Fusion for {cls}...")
                    point_classes, is_applied, reason = apply_fusion_to_class(
                        points, point_classes, labels_2d, confs_2d,
                        cls, joint_prototype, room_geom_ref
                    )
                    if is_applied:
                        print(f"      ✅ Fusion applied: {reason}")
                    else:
                        print(f"      ⚠️ Fusion skipped: {reason}")
                    gc.collect()

    # ============================================================
    # BƯỚC 5: GOM POINTS THEO CLASS (CHỈ GIỮ INDEX)
    # ============================================================
    print(f"\n[5] Grouping points by class...")
    class_indices = {}
    for cls in set(point_classes):
        if cls is None or (isinstance(cls, float) and np.isnan(cls)):
            continue
        mask = (point_classes == cls)
        class_indices[cls] = np.where(mask)[0]
        print(f"   -> {cls}: {len(class_indices[cls]):,} points")

    corrected_classes = point_classes.copy()
    del point_classes
    gc.collect()
    print(f"📊 Memory after grouping: {get_memory():.1f} MB")

    # ============================================================
    # BƯỚC 6: PROCESS EACH CLASS
    # ============================================================
    print(f"\n[6] Processing instances...")
    n_total = 0
    n_suspect = 0
    n_relabeled = 0
    total_classes = len(class_indices)
    class_counter = 0

    for cls, cls_indices in class_indices.items():
        class_counter += 1
        print(f"\n   [{class_counter}/{total_classes}] Class '{cls}': {len(cls_indices):,} points")
        
        # Lấy points của class này theo index
        cls_points = points[cls_indices]
        
        # ===== LOCKED CLASSES =====
        if cls in LOCKED_CLASSES:
            print(f"      🔒 LOCKED - giữ nguyên nhãn MLP")
            feat = {
                "NumPoints": len(cls_points),
                "SurfaceArea": np.nan, "Volume": np.nan, "Height": np.nan,
                "Width": np.nan, "Depth": np.nan,
                "HeightAboveFloor": np.nan, "HeightClearance": np.nan,
                "Planarity": np.nan, "Linearity": np.nan, "Roughness": np.nan,
                "Curvature": np.nan, "NormalEntropy": np.nan,
                "NormalVariance": np.nan, "Compactness": np.nan,
                "Elongation": np.nan, "Sphericity": np.nan,
                "DistanceToWall": np.nan, "DistanceToFloor": np.nan,
                "DistanceToCeiling": np.nan,
                "NormalAlignmentWall": np.nan,
                "NormalAlignmentFloor": np.nan,
                "NormalAlignmentCeiling": np.nan,
            }
            row = {
                "RoomName": room_name,
                "InstanceID": f"{cls}_locked",
                "PredictedClass": cls,
                "FinalClass": cls,
                "Relabeled": False,
                "RelabelSuggested": False,
                "NumPoints": len(cls_points),
                "MahalanobisDistance": np.nan,
                "PValue": np.nan,
                "NDims": np.nan,
                "Threshold": np.nan,
                "Suspect": False,
                "Avg2DSimilarity": np.nan,
                "DecisionReason": "locked_class",
            }
            row.update(feat)
            writer.writerow(row)
            
            del cls_points
            gc.collect()
            continue

        # ===== TÁCH INSTANCE =====
        instance_masks = split_into_instances(cls_points, cls)
        print(f"      -> {len(instance_masks)} instance(s)")

        for idx, mask in enumerate(instance_masks):
            # Lấy points của instance này
            inst_points = cls_points[mask]
            global_idx = cls_indices[mask]
            
            # Extract features
            geom = extract_geometry(inst_points)
            geom = fill_height_from_floor(geom, inst_points, floor_z)
            spatial = extract_spatial(inst_points, room_geom_ref)
            feat = {**geom, **spatial}

            # Verify với Mahalanobis
            d, p_value, k = verify_instance(feat, cls, joint_prototype)
            threshold = error_mining_threshold(k) if k else None
            is_suspect = (d is not None and threshold is not None and d > threshold)

            avg_sim_2d = np.nan
            decision_reason = "mahalanobis_only"
            final_cls = cls
            is_relabel_applied = False
            is_relabel_suggested = False

            # ===== RELABEL LOGIC =====
            if cls in CLASSES_2D_TRUSTED and has_2d:
                inst_labels_2d = labels_2d_names[global_idx]
                inst_confs_2d = confs_2d[global_idx]

                sims = []
                for lbl_2d, conf in zip(inst_labels_2d, inst_confs_2d):
                    if lbl_2d is not None and not pd.isna(lbl_2d):
                        sim = compute_class_similarity(cls, lbl_2d, conf)
                        sims.append(sim)

                if sims:
                    avg_sim_2d = float(np.mean(sims))

                if avg_sim_2d < SIMILARITY_THRESHOLD:
                    decision_reason = f"2d_low_similarity_{avg_sim_2d:.2f}"
                else:
                    decision_reason = f"2d_similarity_ok_{avg_sim_2d:.2f}"

                if is_suspect:
                    suggested_cls, is_relabel_suggested = relabel_instance(
                        feat, cls, joint_prototype
                    )
                    if is_relabel_suggested and suggested_cls != cls:
                        is_relabel_applied = True
                        final_cls = suggested_cls
                        decision_reason += f"_mahalanobis_override_to_{suggested_cls}"
                        print(f"      ⚠️ [{cls}] 2D đồng thuận nhưng Mahalanobis nghi ngờ "
                              f"(d={d:.2f} > {threshold:.2f}) -> ép trả về '{suggested_cls}'")
            else:
                if is_suspect:
                    suggested_cls, is_relabel_suggested = relabel_instance(feat, cls, joint_prototype)
                    if is_relabel_suggested and (cls != "wall" or AUTO_CORRECT_WALL):
                        is_relabel_applied = True
                        final_cls = suggested_cls
                        decision_reason = "mahalanobis_relabel"
                    else:
                        decision_reason = "mahalanobis_suspect_but_keep"
                else:
                    final_cls = cls
                    decision_reason = "mahalanobis_ok"

            # Áp dụng relabel
            if is_relabel_applied:
                safe_idx = global_idx[
                    ~np.isin(corrected_classes[global_idx], list(LOCKED_CLASSES))
                ]
                corrected_classes[safe_idx] = final_cls

            # GHI TRỰC TIẾP VÀO CSV
            row = {
                "RoomName": room_name,
                "InstanceID": f"{cls}_{idx+1}",
                "PredictedClass": cls,
                "FinalClass": final_cls,
                "Relabeled": is_relabel_applied,
                "RelabelSuggested": is_relabel_suggested,
                "NumPoints": len(inst_points),
                "MahalanobisDistance": d,
                "PValue": p_value,
                "NDims": k,
                "Threshold": threshold,
                "Suspect": is_suspect,
                "Avg2DSimilarity": avg_sim_2d,
                "DecisionReason": decision_reason,
            }
            row.update(feat)
            writer.writerow(row)

            n_total += 1
            if is_suspect:
                n_suspect += 1
            if is_relabel_applied:
                n_relabeled += 1

            # GIẢI PHÓNG INSTANCE DATA
            del inst_points, geom, spatial, feat, row
            if idx % 5 == 0:
                gc.collect()

        # GIẢI PHÓNG CLASS DATA
        del instance_masks, cls_points
        gc.collect()
        
        # Lưu checkpoint sau mỗi class
        save_checkpoint(
            room_name=room_name,
            points=points,
            corrected_classes=corrected_classes,
            original_mlp_classes=original_mlp_classes,
            labels_2d=labels_2d,
            confs_2d=confs_2d,
            labels_2d_names=labels_2d_names,
            room_geom_ref=room_geom_ref,
            processed_rooms=[room_name],
            total_instances=n_total,
            total_suspect=n_suspect,
            total_relabeled=n_relabeled
        )
        
        print(f"📊 Memory after class '{cls}': {get_memory():.1f} MB")

    # ============================================================
    # BƯỚC 7: POST-PROCESSING
    # ============================================================
    print(f"\n[7] Post-processing...")
    print(f"📊 Memory before post-processing: {get_memory():.1f} MB")
    
    room_floor_z = room_geom_ref.get('floor_z', None)
    room_ceiling_z = room_geom_ref.get('ceiling_z', None)
    room_wall_points = room_geom_ref.get('wall', None)

    corrected_classes = postprocess_protect_furniture_and_clamp_planar(
        points=points,
        corrected_classes=corrected_classes,
        original_mlp_classes=original_mlp_classes,
        joint_prototype=joint_prototype,
        room_name=room_name,
        room_floor_z=room_floor_z,
        room_ceiling_z=room_ceiling_z,
        room_wall_points=room_wall_points,
    )

    # ============================================================
    # BƯỚC 8: EXPORT PLY
    # ============================================================
    if EXPORT_CORRECTED_PLY:
        print(f"\n[8] Exporting corrected PLY...")
        out_path = export_corrected_ply(points, corrected_classes, room_name)
        print(f"   ✅ Saved: {out_path}")

    # ============================================================
    # BƯỚC 9: CLEANUP
    # ============================================================
    print(f"\n[9] Cleanup...")
    
    # Xóa checkpoint
    cleanup_checkpoint(room_name)
    
    # Xóa cache 2D nếu cấu hình
    if AUTO_CLEANUP_CACHE and not KEEP_2D_CACHE:
        cleanup_2d_cache(room_name)

    # Giải phóng tất cả
    del points, corrected_classes, original_mlp_classes, labels_2d, confs_2d, labels_2d_names
    del class_indices, room_geom_ref
    gc.collect()
    gc.collect()

    print(f"📊 Memory after cleanup: {get_memory():.1f} MB")
    print(f"\n{'='*60}")
    print(f"✅ Room '{room_name}' completed!")
    print(f"   - Total instances: {n_total}")
    print(f"   - Suspect instances: {n_suspect}")
    print(f"   - Relabeled instances: {n_relabeled}")
    print(f"{'='*60}\n")
    
    return n_total, n_suspect, n_relabeled, False


def build_error_mining_report(feat_csv: Path, report_csv: Path):
    df = pd.read_csv(feat_csv)
    df_valid = df[df["MahalanobisDistance"].notna()].copy()
    df_valid["Suspect"] = df_valid["Suspect"].astype(str).str.lower() == "true"
    df_suspect = df_valid[df_valid["Suspect"]].sort_values("MahalanobisDistance", ascending=False)
    df_suspect.to_csv(report_csv, index=False)
    return len(df_valid), len(df_suspect), df_suspect


def main():
    """
    Main pipeline - xử lý tất cả rooms với checkpoint và cache
    """
    import gc
    import time
    
    start_time = time.time()
    
    print("=" * 70)
    print("ERROR MINING PIPELINE - MLP PREDICTION VERIFICATION")
    print("=" * 70)
    
    # ===== LOAD PROTOTYPE =====
    print("\n[1] Loading Joint Prototype...")
    try:
        with open(PROTOTYPE_PATH, "rb") as f:
            joint_prototype = pickle.load(f)
        print(f"   ✅ Classes có prototype: {sorted(joint_prototype.keys())}")
    except Exception as e:
        print(f"   ❌ Lỗi load prototype: {e}")
        return

    # ===== RESOLVE ROOM PATHS =====
    print("\n[2] Resolving room paths...")
    room_paths = resolve_room_paths()
    
    if room_paths:
        print(f"   ✅ Tìm thấy {len(room_paths)} rooms")
    else:
        print(f"   ⚠️  Không tìm thấy room mới từ MLPpredictions directory")

    # ===== LOAD GLOBAL CHECKPOINT =====
    print("\n[3] Loading global checkpoint...")
    global_checkpoint = load_global_checkpoint() if RESUME_FROM_CHECKPOINT else None
    
    processed_rooms = global_checkpoint.get('processed_rooms', []) if global_checkpoint else []
    total_instances = global_checkpoint.get('total_instances', 0) if global_checkpoint else 0
    total_suspect = global_checkpoint.get('total_suspect', 0) if global_checkpoint else 0
    total_relabeled = global_checkpoint.get('total_relabeled', 0) if global_checkpoint else 0

    if processed_rooms:
        print(f"   🔄 Resuming from checkpoint: {len(processed_rooms)} rooms đã xử lý")
        print(f"      - Total instances: {total_instances:,}")
        print(f"      - Total suspect: {total_suspect:,}")
        print(f"      - Total relabeled: {total_relabeled:,}")
    else:
        print("   ℹ️  No checkpoint found. Starting fresh...")

    # ===== PREPARE CSV =====
    print("\n[4] Preparing output CSV...")
    feat_out = OUTPUT_DIR / "per_instance_features.csv"
    file_exists = feat_out.exists()
    
    # Backup existing file
    if file_exists:
        backup_path = OUTPUT_DIR / f"per_instance_features_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        import shutil
        shutil.copy2(feat_out, backup_path)
        print(f"   📄 Backup existing CSV: {backup_path.name}")
    
    # ===== PROCESS ROOMS =====
    print("\n[5] Processing rooms...")
    print("-" * 70)
    
    total_rooms = len(room_paths)
    remaining_rooms = [p for p in room_paths if p.stem.replace('_mlp_refined', '') not in processed_rooms]
    
    if not remaining_rooms and not processed_rooms:
        print(f"   ⚠️  WARNING: No rooms to process!")
        print(f"      - No rooms found in {TEST_OUTPUT_DIR}")
        print(f"      - No checkpoint found from previous runs")
        print(f"\n   Please check:")
        print(f"      1. MLPpredictions directory path: {TEST_OUTPUT_DIR}")
        print(f"      2. File naming (should be *_mlp_refined.ply)")
        print(f"      3. Or run with checkpoint from previous run")
        return
    
    if not remaining_rooms:
        print("   ✅ All rooms already processed!")
        if processed_rooms:
            print(f"   📊 {len(processed_rooms)} rooms in checkpoint")
    else:
        print(f"   📊 Need to process: {len(remaining_rooms)}/{total_rooms} new rooms")
        if processed_rooms:
            print(f"   📊 Plus {len(processed_rooms)} rooms from checkpoint")
    
    # Memory tracking
    def get_memory():
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    with open(feat_out, "a" if file_exists else "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ALL_COLS)
        if not file_exists:
            writer.writeheader()
            print("   📄 Created new CSV with header")
        
        for idx, ply_path in enumerate(room_paths):
            room_name = ply_path.stem.replace('_mlp_refined', '')
            
            # Check if file exists (in case it was deleted)
            if not ply_path.exists():
                print(f"\n⏭️  [{idx+1}/{total_rooms}] Skipping {room_name} (file deleted: {ply_path.name})")
                if room_name in processed_rooms:
                    # Mark as processed even though file is missing
                    print(f"   ℹ️  Already in processed list, skipping...")
                continue
            
            # Skip processed rooms
            if room_name in processed_rooms:
                print(f"\n⏭️  [{idx+1}/{total_rooms}] Skipping {room_name} (already processed)")
                continue
            
            print(f"\n{'='*70}")
            print(f"[{idx+1}/{total_rooms}] Processing: {room_name}")
            print(f"{'='*70}")
            print(f"📊 Memory before: {get_memory():.1f} MB")
            
            room_start_time = time.time()
            
            try:
                # Process room
                n_total, n_suspect, n_relabeled, was_checkpoint = process_one_room(
                    ply_path, joint_prototype, writer, resume=RESUME_FROM_CHECKPOINT
                )
                
                if was_checkpoint:
                    print(f"   ⏭️  Room was checkpointed, skipping...")
                    continue
                
                # Update statistics
                f.flush()
                total_instances += n_total
                total_suspect += n_suspect
                total_relabeled += n_relabeled
                processed_rooms.append(room_name)
                
                # Save global checkpoint
                save_global_checkpoint(processed_rooms, total_instances, total_suspect, total_relabeled)
                
                room_time = time.time() - room_start_time
                print(f"\n   ✅ Room completed in {room_time:.1f}s")
                print(f"   📊 Memory after: {get_memory():.1f} MB")
                print(f"   📊 Cumulative: {len(processed_rooms)}/{total_rooms} rooms, "
                      f"{total_instances:,} instances, {total_suspect:,} suspect, {total_relabeled:,} relabeled")
                
            except FileNotFoundError as e:
                print(f"\n   ⏭️  SKIPPING {room_name}: File not found or deleted")
                print(f"      Error: {e}")
                # Mark as processed so we don't try again
                if room_name not in processed_rooms:
                    processed_rooms.append(room_name)
                    save_global_checkpoint(processed_rooms, total_instances, total_suspect, total_relabeled)
                gc.collect()
                continue
                
            except Exception as e:
                print(f"\n   ❌ ERROR processing {room_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Emergency save
                save_global_checkpoint(processed_rooms, total_instances, total_suspect, total_relabeled)
                print(f"   💾 Emergency checkpoint saved!")
                
                # Clean up memory
                gc.collect()
                continue

    # ===== GENERATE REPORT =====
    print("\n" + "=" * 70)
    print("[6] Generating final report...")
    print("=" * 70)
    
    try:
        report_out = OUTPUT_DIR / "error_mining_report.csv"
        n_total, n_suspect, df_suspect = build_error_mining_report(feat_out, report_out)
        print(f"   ✅ Saved: {report_out}")
    except Exception as e:
        print(f"   ❌ Error generating report: {e}")
        df_suspect = None
        n_total = 0
        n_suspect = 0

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print(f"\n📊 Processing Statistics:")
    print(f"   - Total rooms processed: {len(processed_rooms)}/{total_rooms}")
    print(f"   - Total instances verified: {total_instances:,}")
    print(f"   - Total suspect instances: {total_suspect:,} ({total_suspect/total_instances*100:.1f}%" if total_instances > 0 else "N/A")
    print(f"   - Total relabeled instances: {total_relabeled:,} ({total_relabeled/total_instances*100:.1f}%" if total_instances > 0 else "N/A")
    print(f"\n⏱️  Total time: {hours}h {minutes}m {seconds}s")
    
    # Memory summary
    import psutil
    import os
    process = psutil.Process(os.getpid())
    final_memory = process.memory_info().rss / 1024 / 1024
    print(f"💾 Final memory usage: {final_memory:.1f} MB")
    
    # Files
    print(f"\n📁 Output files:")
    print(f"   - Features CSV: {feat_out} ({feat_out.stat().st_size / 1024 / 1024:.1f} MB)" if feat_out.exists() else "   - Features CSV: Not found")
    print(f"   - Report CSV: {report_out} ({report_out.stat().st_size / 1024 / 1024:.1f} MB)" if report_out.exists() else "   - Report CSV: Not found")
    
    # Top suspect instances
    if df_suspect is not None and len(df_suspect) > 0:
        print(f"\n🔍 Top 10 most suspicious instances:")
        cols = ["RoomName", "InstanceID", "PredictedClass", "FinalClass",
                "MahalanobisDistance", "PValue", "Avg2DSimilarity", "DecisionReason"]
        print(df_suspect[cols].head(10).to_string(index=False))
        
        # Save detailed summary
        summary_path = OUTPUT_DIR / "summary_stats.txt"
        with open(summary_path, 'w', encoding='utf-8') as sf:
            sf.write("=" * 70 + "\n")
            sf.write("ERROR MINING SUMMARY\n")
            sf.write("=" * 70 + "\n\n")
            sf.write(f"Total rooms processed: {len(processed_rooms)}/{total_rooms}\n")
            sf.write(f"Total instances verified: {total_instances:,}\n")
            sf.write(f"Total suspect instances: {total_suspect:,}\n")
            sf.write(f"Total relabeled instances: {total_relabeled:,}\n")
            sf.write(f"\nTop 10 suspicious instances:\n")
            sf.write(df_suspect[cols].head(10).to_string(index=False))
            sf.write(f"\n\nClass distribution of suspects:\n")
            sf.write(df_suspect['PredictedClass'].value_counts().to_string())
        print(f"\n   📄 Detailed summary saved: {summary_path}")
    
    # ===== CLEANUP =====
    print("\n[7] Final cleanup...")
    
    # Clean up 2D cache if configured
    if AUTO_CLEANUP_CACHE:
        cleanup_2d_cache()
        print("   🧹 Cleaned up 2D cache")
    
    # Force garbage collection
    gc.collect()
    gc.collect()
    
    print("\n" + "=" * 70)
    print("✅ DONE!")
    print("=" * 70)

if __name__ == "__main__":
    main()