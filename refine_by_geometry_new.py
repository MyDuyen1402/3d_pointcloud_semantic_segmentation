"""
=====================================================================
REFINE BY GEOMETRY - LÀM MỊN KẾT QUẢ SAU ERROR MINING
=====================================================================

Input: Tất cả file PLY đã được corrected từ error_mining.py trong thư mục ErrorMining
       File joint_prototype.csv (chứa thống kê các class)

Output: File PLY đã được làm mịn (refined) theo các quy tắc hình học

Các quy tắc làm mịn:
1. Tường, trần, sàn: ép về mặt phẳng chính xác bằng RANSAC
2. Cửa sổ: phải nằm trên 1 mặt phẳng, độ dày <= ngưỡng từ prototype
3. Bảng (whiteboard): phải gần hình chữ nhật, nằm trên tường
4. Tường bị nhầm thành cửa sổ: kiểm tra độ dày, nếu dày hơn prototype -> trả về tường
=====================================================================
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import warnings
from typing import Tuple, Optional, List, Dict
import pandas as pd
import json
from datetime import datetime
import time

# =====================================================
# PATHS
# =====================================================

INPUT_DIR = Path("/compute_home/slurmdang12/datasets/ErrorMining")
PROTOTYPE_CSV_PATH = Path("/compute_home/slurmdang12/datasets/Prototype/joint_prototype.csv")
OUTPUT_DIR = Path("/compute_home/slurmdang12/datasets/ErrorMining_output")
CHECKPOINT_DIR = Path("/compute_home/slurmdang12/datasets/ErrorMining_output/checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# BẢNG MÀU
# =====================================================

CLASS_COLORS = {
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

# =====================================================
# THAM SỐ CHO RANSAC PLANE FITTING
# =====================================================

RANSAC_DISTANCE_THRESHOLD = 0.05  # 5cm
RANSAC_NUM_ITERATIONS = 1000
RANSAC_MIN_INLIER_RATIO = 0.6

# =====================================================
# THAM SỐ CHO TƯỜNG, TRẦN, SÀN
# =====================================================

WALL_PLANE_EPS = 0.05      # Sai số cho phép ép tường vào mặt phẳng
FLOOR_CEILING_EPS = 0.03   # Sai số cho phép ép sàn/trần vào mặt phẳng
CEILING_FLOOR_MAX_PROTRUSION = 0.3  # Điểm lệch mặt phẳng > ngưỡng này -> KHÔNG ép,
                                     # mà coi là khả năng bị gán nhãn sai -> gán lại
                                     # nhãn theo hàng xóm gần nhất (xem vote_relabel_by_neighbors)
RELABEL_KNN_K = 6              # Số hàng xóm dùng để vote gán lại nhãn
RELABEL_MIN_VOTE_RATIO = 0.5   # Tỉ lệ đa số tối thiểu trong k hàng xóm mới gán lại

# =====================================================
# THAM SỐ CHO CỬA SỔ - LẤY TỪ PROTOTYPE
# =====================================================

WINDOW_MIN_POINTS_FOR_PLANE = 20   # Số điểm tối thiểu để kiểm tra mặt phẳng
WINDOW_PLANE_DISTANCE_THRESHOLD = 0.08  # Ngưỡng inlier cho RANSAC khi kiểm tra cửa sổ
WINDOW_MAX_THICKNESS_RATIO = 0.12  # Độ dày tối đa = Height * 0.12 (12% chiều cao)
WINDOW_THICKNESS_PERCENTILE = (5, 95)  # Cắt 5% điểm nhiễu ở mỗi đầu khi đo độ dày

WINDOW_EXTEND_COPLANAR_DIST = 0.08   # Điểm 'wall' lệch mặt phẳng cửa sổ <= ngưỡng này
                                      # mới được coi là "cùng mặt phẳng" để xét mở rộng
WINDOW_EXTEND_BIN_SIZE = 0.05        # Kích thước bin quét theo chiều cao (5cm)
WINDOW_EXTEND_DENSITY_RATIO = 0.35   # Bin có mật độ <= tỉ lệ này so với mật độ tường
                                      # thật (đo xa cửa sổ) -> coi là cửa sổ bị thiếu
WINDOW_EXTEND_U_MARGIN = 0.05        # Biên nới rộng theo chiều ngang khi xét điểm wall

# =====================================================
# THAM SỐ CHO BẢNG (WHITEBOARD) - LẤY TỪ PROTOTYPE
# =====================================================

WHITEBOARD_MIN_FILL_RATIO = 0.8     # Diện tích hull / hình chữ nhật bao NHỎ NHẤT
WHITEBOARD_MAX_WIDTH_STD_RATIO = 0.3 # Độ lệch chiều rộng tối đa
WHITEBOARD_MAX_ASPECT_RATIO = 3.0   # Tỉ lệ dài/rộng tối đa
WHITEBOARD_MAX_DIST_TO_WALL = 0.25  # Khoảng cách tối đa từ bảng đến tường (25cm)
WHITEBOARD_AREA_TOLERANCE = 2.5     # Diện tích cho phép = prototype * tolerance

# =====================================================
# THAM SỐ DBSCAN
# =====================================================

DBSCAN_EPS = 0.3
DBSCAN_MIN_SAMPLES = 15


# =====================================================
# ĐỌC PROTOTYPE TỪ CSV
# =====================================================

def load_prototype_from_csv(csv_path: Path) -> Dict:
    """
    Đọc joint_prototype.csv và trả về dict chứa thông tin các class.
    """
    df = pd.read_csv(csv_path)
    
    # Lấy danh sách các class (cột đầu tiên)
    classes = df['Class'].tolist()
    
    # Lấy danh sách các feature - lấy từ các cột có hậu tố _mean
    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    feature_cols = [col[:-5] for col in mean_cols]  # bỏ _mean
    
    # Lọc bỏ các cột không phải feature
    exclude = ['Class', 'N', 'NumPoints']
    feature_cols = [f for f in feature_cols if f not in exclude]
    
    prototype = {}
    
    for i, cls in enumerate(classes):
        mu = []
        sigma = []
        
        for feature in feature_cols:
            mean_col = f"{feature}_mean"
            std_col = f"{feature}_std"
            
            # Đọc giá trị
            mean_val = df.loc[i, mean_col] if mean_col in df.columns else 0.0
            std_val = df.loc[i, std_col] if std_col in df.columns else 1.0
            
            mu.append(float(mean_val) if not pd.isna(mean_val) else 0.0)
            sigma.append(float(std_val) if not pd.isna(std_val) else 1.0)
        
        prototype[cls] = {
            'features': feature_cols,
            'mu': np.array(mu),
            'sigma': np.array(sigma),
        }
    
    return prototype


# =====================================================
# CHECKPOINT MANAGEMENT
# =====================================================

def save_checkpoint(checkpoint_path: Path, data: Dict):
    """Lưu checkpoint"""
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: Path) -> Dict:
    """Đọc checkpoint"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {}


def is_processed(checkpoint_path: Path, ply_name: str) -> bool:
    """Kiểm tra xem file đã được xử lý chưa"""
    checkpoint = load_checkpoint(checkpoint_path)
    return ply_name in checkpoint.get('processed_files', [])


def mark_processed(checkpoint_path: Path, ply_name: str, stats: Dict):
    """Đánh dấu file đã được xử lý"""
    checkpoint = load_checkpoint(checkpoint_path)
    if 'processed_files' not in checkpoint:
        checkpoint['processed_files'] = []
    if ply_name not in checkpoint['processed_files']:
        checkpoint['processed_files'].append(ply_name)
    
    if 'file_stats' not in checkpoint:
        checkpoint['file_stats'] = {}
    checkpoint['file_stats'][ply_name] = stats
    checkpoint['last_update'] = datetime.now().isoformat()
    
    save_checkpoint(checkpoint_path, checkpoint)


# =====================================================
# HÀM TIỆN ÍCH
# =====================================================

def load_ply(ply_path: Path):
    """Load PLY file"""
    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    return points, colors


def color_to_class_batch(colors: np.ndarray, palette: dict, eps: float = 0.05):
    """Map màu -> class name"""
    palette_names = list(palette.keys())
    palette_arr = np.array([palette[c] for c in palette_names])

    dists = np.linalg.norm(colors[:, None, :] - palette_arr[None, :, :], axis=2)
    best_idx = np.argmin(dists, axis=1)
    best_dist = dists[np.arange(len(colors)), best_idx]

    labels = np.array([palette_names[i] for i in best_idx], dtype=object)
    labels[best_dist >= eps] = None
    return labels


def fit_plane_ransac(points: np.ndarray, 
                     distance_threshold: float = RANSAC_DISTANCE_THRESHOLD,
                     num_iterations: int = RANSAC_NUM_ITERATIONS):
    """
    Tìm mặt phẳng chiếm đa số điểm bằng RANSAC.
    Trả về: (normal, centroid, inlier_mask)
    """
    n = len(points)
    if n < 10:
        return None, None, None

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

    if best_normal is None or best_inlier_count < 10:
        return None, None, None

    inlier_ratio = best_inlier_count / n
    if inlier_ratio < RANSAC_MIN_INLIER_RATIO:
        return None, None, None

    # Tinh chỉnh normal trên tập inlier
    inlier_pts = points[best_inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    cov = np.cov((inlier_pts - centroid).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    refined_normal = eigvecs[:, np.argmin(eigvals)]

    # Đảm bảo normal hướng lên trên (cho sàn/trần) hoặc hướng ra ngoài (cho tường)
    if refined_normal[2] > 0:
        refined_normal = -refined_normal

    return refined_normal, centroid, best_inlier_mask


def project_points_to_plane(points: np.ndarray, normal: np.ndarray, centroid: np.ndarray):
    """Chiếu điểm lên mặt phẳng"""
    d = (points - centroid) @ normal
    return points - d[:, None] * normal[None, :]


def get_plane_thickness(points: np.ndarray, normal: np.ndarray, centroid: np.ndarray):
    """Tính độ dày của tập điểm so với mặt phẳng (dùng cho tường/sàn/trần)"""
    dist = np.abs((points - centroid) @ normal)
    return float(dist.max() - dist.min()), float(np.std(dist))


def get_plane_thickness_robust(points: np.ndarray, normal: np.ndarray, centroid: np.ndarray,
                                percentile: Tuple[float, float] = WINDOW_THICKNESS_PERCENTILE):
    """
    Tính độ dày theo percentile (vd. 5%-95%) thay vì max-min thô.
    """
    dist = np.abs((points - centroid) @ normal)
    lo, hi = percentile
    p_lo = np.percentile(dist, lo)
    p_hi = np.percentile(dist, hi)
    return float(p_hi - p_lo), float(np.std(dist))


def min_bounding_rectangle_fill_ratio(points_2d: np.ndarray) -> Tuple[float, float, float]:
    """
    Tìm hình chữ nhật bao NHỎ NHẤT của convex hull, xoay thử theo từng cạnh
    của hull (rotating calipers đơn giản hoá) — không phụ thuộc trục PCA.
    """
    hull = ConvexHull(points_2d)
    hull_pts = points_2d[hull.vertices]
    hull_area = hull.volume

    n = len(hull_pts)
    min_area = np.inf
    best_w, best_h = 0.0, 0.0

    for i in range(n):
        p1 = hull_pts[i]
        p2 = hull_pts[(i + 1) % n]
        edge = p2 - p1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-9:
            continue
        edge_dir = edge / edge_len
        perp_dir = np.array([-edge_dir[1], edge_dir[0]])

        proj_edge = hull_pts @ edge_dir
        proj_perp = hull_pts @ perp_dir

        w = proj_edge.max() - proj_edge.min()
        h = proj_perp.max() - proj_perp.min()
        area = w * h
        if area < min_area:
            min_area = area
            best_w, best_h = w, h

    if not np.isfinite(min_area) or min_area < 1e-6:
        return 0.0, 0.0, 0.0

    fill_ratio = hull_area / min_area
    return float(fill_ratio), float(best_w), float(best_h)


def compute_rectangularity(points: np.ndarray) -> Tuple[float, float, float]:
    """
    Tính độ vuông vắn của một tập điểm trên mặt phẳng.
    Trả về: (fill_ratio, width_std_ratio, aspect_ratio)
    """
    if len(points) < 20:
        return 0.0, 1.0, 1.0

    # Chiếu lên mặt phẳng PCA (2 trục chính)
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    u_axis, v_axis = eigvecs[:, order[0]], eigvecs[:, order[1]]
    
    proj_u = centered @ u_axis
    proj_v = centered @ v_axis
    points_2d = np.column_stack([proj_u, proj_v])

    try:
        fill_ratio, w, h = min_bounding_rectangle_fill_ratio(points_2d)
    except Exception:
        return 0.0, 1.0, 1.0

    if w < 1e-6 or h < 1e-6:
        return 0.0, 1.0, 1.0

    aspect_ratio = max(w, h) / min(w, h)

    # Độ đồng đều chiều rộng theo chiều cao
    n_bins = 10
    v_min, v_max = proj_v.min(), proj_v.max()
    if v_max - v_min < 1e-6:
        return fill_ratio, 0.0, aspect_ratio

    bin_edges = np.linspace(v_min, v_max, n_bins + 1)
    widths = []
    for i in range(n_bins):
        mask = (proj_v >= bin_edges[i]) & (proj_v < bin_edges[i+1])
        if np.sum(mask) < 5:
            continue
        widths.append(proj_u[mask].max() - proj_u[mask].min())

    if len(widths) < 3:
        return fill_ratio, 0.0, aspect_ratio

    widths = np.array(widths)
    width_std_ratio = widths.std() / (widths.mean() + 1e-6)

    return fill_ratio, width_std_ratio, aspect_ratio


def is_rectangular(points: np.ndarray) -> bool:
    """Kiểm tra xem tập điểm có hình chữ nhật không"""
    fill_ratio, width_std_ratio, aspect_ratio = compute_rectangularity(points)
    return (fill_ratio >= WHITEBOARD_MIN_FILL_RATIO and 
            width_std_ratio <= WHITEBOARD_MAX_WIDTH_STD_RATIO and
            aspect_ratio <= WHITEBOARD_MAX_ASPECT_RATIO)


def is_attached_to_wall(points: np.ndarray, wall_points: np.ndarray) -> bool:
    """Kiểm tra xem tập điểm có nằm sát tường không"""
    if len(wall_points) == 0:
        return False
    
    from scipy.spatial import cKDTree
    tree = cKDTree(wall_points)
    dists, _ = tree.query(points, k=1)
    mean_dist = float(np.mean(dists))
    return mean_dist <= WHITEBOARD_MAX_DIST_TO_WALL


def split_into_instances(points: np.ndarray, eps: float = DBSCAN_EPS, 
                         min_samples: int = DBSCAN_MIN_SAMPLES) -> List[np.ndarray]:
    """Tách tập điểm thành các instance bằng DBSCAN"""
    if len(points) < min_samples:
        return [np.ones(len(points), dtype=bool)]
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit(points)
    labels = clustering.labels_
    
    instances = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        instances.append(labels == lbl)
    
    if len(instances) == 0:
        return [np.ones(len(points), dtype=bool)]
    
    return instances


def vote_relabel_by_neighbors(global_idx: np.ndarray,
                               all_points: np.ndarray,
                               all_classes: np.ndarray,
                               current_class: str,
                               k: int = RELABEL_KNN_K,
                               min_vote_ratio: float = RELABEL_MIN_VOTE_RATIO) -> np.ndarray:
    """
    Dùng cho các điểm bị nghi gán nhãn sai.
    Tìm k điểm gần nhất KHÔNG thuộc current_class trong toàn bộ point cloud,
    gán lại nhãn theo class chiếm đa số trong k điểm đó.
    """
    from scipy.spatial import cKDTree

    other_mask = (all_classes != current_class) & (all_classes != None)
    other_points = all_points[other_mask]
    other_classes = all_classes[other_mask]

    new_labels = all_classes[global_idx].copy()
    if len(other_points) == 0 or len(global_idx) == 0:
        return new_labels

    tree = cKDTree(other_points)
    k_use = min(k, len(other_points))
    _, nn_idx = tree.query(all_points[global_idx], k=k_use)
    if k_use == 1:
        nn_idx = nn_idx[:, None]

    nn_classes = other_classes[nn_idx]

    for i in range(len(global_idx)):
        vals, counts = np.unique(nn_classes[i], return_counts=True)
        best = vals[np.argmax(counts)]
        ratio = counts.max() / k_use
        if ratio >= min_vote_ratio:
            new_labels[i] = best

    return new_labels


# =====================================================
# CÁC HÀM LÀM MỊN CHÍNH
# =====================================================

def refine_wall(points: np.ndarray, indices: np.ndarray, 
                corrected_classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Làm mịn tường: ép các điểm vào mặt phẳng chính xác
    """
    if len(points) < 50:
        return points, indices
    
    # Tìm mặt phẳng chính
    normal, centroid, inlier_mask = fit_plane_ransac(
        points, distance_threshold=RANSAC_DISTANCE_THRESHOLD
    )
    
    if normal is None:
        return points, indices
    
    # Chiếu tất cả điểm lên mặt phẳng
    points = project_points_to_plane(points, normal, centroid)
    
    # Kiểm tra độ dày của tường
    thickness, std = get_plane_thickness(points, normal, centroid)
    
    # Nếu tường quá dày (> 30cm) thì giữ nguyên
    if thickness > 0.3:
        return points, indices
    
    return points, indices


def refine_floor_ceiling(points: np.ndarray, indices: np.ndarray,
                         all_points: np.ndarray, all_classes: np.ndarray,
                         is_ceiling: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Làm mịn sàn/trần: ép vào mặt phẳng nằm ngang.
    """
    if len(points) < 50:
        return points, indices
    
    # Tìm mặt phẳng (ưu tiên mặt phẳng nằm ngang)
    normal, centroid, inlier_mask = fit_plane_ransac(
        points, distance_threshold=RANSAC_DISTANCE_THRESHOLD
    )
    
    if normal is None:
        return points, indices
    
    projected_points = points.copy()
    current_class = 'ceiling' if is_ceiling else 'floor'
    
    if is_ceiling:
        # Trần: lấy z cao nhất của inlier làm chuẩn
        z_ref = points[inlier_mask][:, 2].max()
    else:
        # Sàn: lấy z thấp nhất của inlier làm chuẩn
        z_ref = points[inlier_mask][:, 2].min()
    
    # Điểm ngoài inlier
    outlier_mask = ~inlier_mask
    if np.sum(outlier_mask) > 0:
        outlier_local = np.where(outlier_mask)[0]
        dist_to_plane = np.abs((points[outlier_local] - centroid) @ normal)

        # Lệch trong khoảng cho phép -> ép như cũ (vd. đèn âm trần)
        within_cap = dist_to_plane <= CEILING_FLOOR_MAX_PROTRUSION
        far_local = outlier_local[~within_cap]

        # Lệch quá xa -> nghi gán nhãn sai, GÁN LẠI nhãn theo hàng xóm
        if len(far_local) > 0:
            far_global = indices[far_local]
            new_labels = vote_relabel_by_neighbors(
                far_global, all_points, all_classes, current_class
            )
            changed = int(np.sum(new_labels != current_class))
            all_classes[far_global] = new_labels
            tag = "Ceiling" if is_ceiling else "Floor"
            print(f"  ⚠️ {tag}: {len(far_local)} điểm lệch > "
                  f"{CEILING_FLOOR_MAX_PROTRUSION}m -> gán lại nhãn cho "
                  f"{changed}/{len(far_local)} điểm theo hàng xóm gần nhất")
    
    return points, indices


def get_window_max_thickness(prototype: Dict, cls: str = 'window') -> float:
    """
    Lấy độ dày tối đa cho cửa sổ từ prototype.
    Dùng Height * WINDOW_MAX_THICKNESS_RATIO
    """
    if cls not in prototype:
        return 0.25  # fallback 25cm
    
    try:
        proto = prototype[cls]
        features = proto['features']
        mu = proto['mu']
        
        # Tìm chiều cao (Height) trong features
        if 'Height' in features:
            idx = features.index('Height')
            height = mu[idx]
            if height > 0:
                return height * WINDOW_MAX_THICKNESS_RATIO
    except:
        pass
    
    return 0.25  # fallback 25cm


def get_whiteboard_area_range(prototype: Dict, cls: str = 'board') -> Tuple[float, float]:
    """
    Lấy khoảng diện tích cho phép của whiteboard từ prototype.
    Trả về: (min_area, max_area)
    """
    # Fallback mặc định
    min_area = 0.15
    max_area = 2.5
    
    if cls not in prototype:
        return min_area, max_area
    
    try:
        proto = prototype[cls]
        features = proto['features']
        mu = proto['mu']
        sigma = proto['sigma']
        
        # Tìm Width và Height
        if 'Width' in features and 'Height' in features:
            w_idx = features.index('Width')
            h_idx = features.index('Height')
            width = mu[w_idx]
            height = mu[h_idx]
            
            if width > 0 and height > 0:
                area = width * height
                min_area = max(0.1, area * 0.3)  # 30% của diện tích chuẩn
                max_area = area * WHITEBOARD_AREA_TOLERANCE
    except:
        pass
    
    return min_area, max_area


def extend_window_upward(window_points: np.ndarray,
                          normal: np.ndarray,
                          centroid: np.ndarray,
                          wall_points: np.ndarray,
                          wall_global_indices: np.ndarray,
                          all_classes: np.ndarray) -> int:
    """
    Mở rộng cửa sổ lên phía trên nếu phần đang gán nhãn 'wall' ngay trên nó
    thực ra CÙNG MẶT PHẲNG với cửa sổ và có mật độ điểm thấp bất thường.
    """
    if len(wall_points) == 0 or len(window_points) == 0:
        return 0

    world_up = np.array([0.0, 0.0, 1.0])
    v_axis = world_up - (world_up @ normal) * normal
    v_norm = np.linalg.norm(v_axis)
    if v_norm < 1e-6:
        return 0
    v_axis = v_axis / v_norm
    u_axis = np.cross(normal, v_axis)
    u_axis = u_axis / (np.linalg.norm(u_axis) + 1e-9)

    def to_uv(pts):
        rel = pts - centroid
        return rel @ u_axis, rel @ v_axis

    win_u, win_v = to_uv(window_points)
    u_min, u_max = win_u.min(), win_u.max()
    v_max_window = win_v.max()

    dist_to_plane = np.abs((wall_points - centroid) @ normal)
    coplanar_mask = dist_to_plane <= WINDOW_EXTEND_COPLANAR_DIST
    if not np.any(coplanar_mask):
        return 0

    wp = wall_points[coplanar_mask]
    wp_global_idx = wall_global_indices[coplanar_mask]
    wu, wv = to_uv(wp)

    margin = WINDOW_EXTEND_U_MARGIN
    in_u_range = (wu >= u_min - margin) & (wu <= u_max + margin)
    above_window = wv > v_max_window
    candidate_mask = in_u_range & above_window
    if not np.any(candidate_mask):
        return 0

    cand_v = wv[candidate_mask]
    cand_global_idx = wp_global_idx[candidate_mask]

    # Mật độ tường thật, đo ở xa cửa sổ
    far_mask = (wu < u_min - 0.3) | (wu > u_max + 0.3)
    if np.sum(far_mask) < 30:
        return 0
    far_u, far_v = wu[far_mask], wv[far_mask]
    far_area = (far_v.max() - far_v.min()) * (far_u.max() - far_u.min())
    if far_area < 1e-6:
        return 0
    ref_density = np.sum(far_mask) / far_area

    col_width = (u_max - u_min) + 2 * margin
    bin_edges = np.arange(v_max_window, cand_v.max() + WINDOW_EXTEND_BIN_SIZE,
                           WINDOW_EXTEND_BIN_SIZE)

    relabel_idx = []
    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (cand_v >= lo) & (cand_v < hi)
        n_in_bin = int(np.sum(in_bin))
        bin_area = col_width * WINDOW_EXTEND_BIN_SIZE
        local_density = n_in_bin / max(bin_area, 1e-6)

        if local_density <= ref_density * WINDOW_EXTEND_DENSITY_RATIO:
            relabel_idx.extend(cand_global_idx[in_bin].tolist())
        else:
            break

    if len(relabel_idx) == 0:
        return 0

    relabel_idx = np.array(relabel_idx)
    all_classes[relabel_idx] = 'window'
    return len(relabel_idx)


def refine_window(points: np.ndarray, indices: np.ndarray,
                  corrected_classes: np.ndarray,
                  prototype: Dict,
                  all_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Làm mịn cửa sổ.
    """
    if len(points) < WINDOW_MIN_POINTS_FOR_PLANE:
        return points, indices, True
    
    # Tìm mặt phẳng chính
    normal, centroid, inlier_mask = fit_plane_ransac(
        points, distance_threshold=WINDOW_PLANE_DISTANCE_THRESHOLD
    )
    
    if normal is None:
        return points, indices, True
    
    # Đo độ dày theo percentile
    thickness, std = get_plane_thickness_robust(points, normal, centroid)
    
    # Lấy ngưỡng độ dày từ prototype
    max_thickness = get_window_max_thickness(prototype, 'window')
    print(f"  Window: thickness={thickness:.3f}m, max={max_thickness:.3f}m")
    
    if thickness > max_thickness:
        print(f"  ⚠️ Window: thickness={thickness:.3f}m > {max_thickness:.3f}m, trả về wall")
        return points, indices, False
    
    # Cụm này là cửa sổ thật -> chỉ ép điểm NẰM GẦN mặt phẳng (inlier)
    projected_points = points.copy()
    projected_points[inlier_mask] = project_points_to_plane(
        points[inlier_mask], normal, centroid
    )
    
    # Điểm lệch xa mặt phẳng trong cùng cụm: gán lại nhãn theo hàng xóm
    outlier_mask = ~inlier_mask
    if np.sum(outlier_mask) > 0:
        outlier_local = np.where(outlier_mask)[0]
        far_global = indices[outlier_local]
        new_labels = vote_relabel_by_neighbors(
            far_global, all_points, corrected_classes, 'window'
        )
        changed = int(np.sum(new_labels != 'window'))
        corrected_classes[far_global] = new_labels
        print(f"  Window: {len(outlier_local)} điểm lệch mặt phẳng -> "
              f"gán lại nhãn cho {changed}/{len(outlier_local)} điểm theo hàng xóm")
    
    return points, indices, True


def refine_whiteboard(points: np.ndarray, indices: np.ndarray,
                      corrected_classes: np.ndarray,
                      wall_points: np.ndarray,
                      prototype: Dict,
                      all_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Làm mịn bảng (whiteboard).
    """
    if len(points) < 20:
        return points, indices, True
    
    # 1. Kiểm tra hình chữ nhật
    if not is_rectangular(points):
        print(f"  ⚠️ Whiteboard: không đủ hình chữ nhật, trả về wall")
        return points, indices, False
    
    # 2. Tính diện tích
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    u_axis, v_axis = eigvecs[:, order[0]], eigvecs[:, order[1]]
    proj_u = centered @ u_axis
    proj_v = centered @ v_axis
    area = (proj_u.max() - proj_u.min()) * (proj_v.max() - proj_v.min())
    
    # Lấy ngưỡng diện tích từ prototype
    min_area, max_area = get_whiteboard_area_range(prototype, 'board')
    print(f"  Whiteboard: area={area:.3f}m², range=[{min_area:.3f}, {max_area:.3f}]")
    
    if area < min_area or area > max_area:
        print(f"  ⚠️ Whiteboard: area={area:.3f}m² không hợp lý, trả về wall")
        return points, indices, False
    
    # 3. Kiểm tra nằm trên tường
    if not is_attached_to_wall(points, wall_points):
        print(f"  ⚠️ Whiteboard: không nằm sát tường, trả về wall")
        return points, indices, False
    
    # Tìm mặt phẳng chính để ép
    normal, centroid, inlier_mask = fit_plane_ransac(
        points, distance_threshold=RANSAC_DISTANCE_THRESHOLD
    )
    
    if normal is None:
        return points, indices, True
    
    # Chỉ ép điểm NẰM GẦN mặt phẳng (inlier)
    projected_points = points.copy()
    projected_points[inlier_mask] = project_points_to_plane(
        points[inlier_mask], normal, centroid
    )
    
    # Điểm lệch xa mặt phẳng trong cùng cụm: gán lại nhãn theo hàng xóm
    outlier_mask = ~inlier_mask
    if np.sum(outlier_mask) > 0:
        outlier_local = np.where(outlier_mask)[0]
        far_global = indices[outlier_local]
        new_labels = vote_relabel_by_neighbors(
            far_global, all_points, corrected_classes, 'whiteboard'
        )
        changed = int(np.sum(new_labels != 'whiteboard'))
        corrected_classes[far_global] = new_labels
        print(f"  Whiteboard: {len(outlier_local)} điểm lệch mặt phẳng -> "
              f"gán lại nhãn cho {changed}/{len(outlier_local)} điểm theo hàng xóm")
    
    return points, indices, True


# =====================================================
# HÀM MAIN - XỬ LÝ TOÀN BỘ FILE TRONG THƯ MỤC
# =====================================================

def process_single_file(ply_path: Path, prototype: Dict, output_dir: Path, 
                       checkpoint_path: Path) -> Dict:
    """
    Xử lý một file PLY duy nhất và trả về thống kê
    """
    print(f"\n{'='*60}")
    print(f"REFINE BY GEOMETRY: {ply_path.name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Load PLY
    points, colors = load_ply(ply_path)
    print(f"Loaded {len(points)} points")
    
    # Map màu -> class
    class_labels = color_to_class_batch(colors, CLASS_COLORS)
    unique_classes = set(cls for cls in class_labels if cls is not None)
    print(f"Mapped to {len(unique_classes)} classes: {sorted(unique_classes)}")
    
    # Lấy các class đặc biệt
    wall_mask = (class_labels == 'wall')
    floor_mask = (class_labels == 'floor')
    ceiling_mask = (class_labels == 'ceiling')
    window_mask = (class_labels == 'window')
    whiteboard_mask = (class_labels == 'whiteboard')
    
    # Lấy wall points để kiểm tra whiteboard / mở rộng cửa sổ
    wall_points = points[wall_mask] if np.any(wall_mask) else np.array([])
    wall_global_indices = np.where(wall_mask)[0] if np.any(wall_mask) else np.array([], dtype=int)
    print(f"Wall points: {len(wall_points)}")
    
    # Sao chép điểm để refine
    refined_points = points.copy()
    refined_classes = class_labels.copy()
    
    # === 1. LÀM MỊN TƯỜNG ===
    if np.any(wall_mask):
        print("\n[1] Refining walls...")
        wall_indices = np.where(wall_mask)[0]
        wall_pts = points[wall_indices]
        
        # Tách tường thành các mặt phẳng riêng
        wall_instances = split_into_instances(wall_pts)
        print(f"  Found {len(wall_instances)} wall plane(s)")
        
        for i, inst_mask in enumerate(wall_instances):
            inst_pts = wall_pts[inst_mask]
            inst_indices = wall_indices[inst_mask]
            projected, _ = refine_wall(inst_pts, inst_indices, refined_classes)
            refined_points[inst_indices] = projected
    
    # === 2. LÀM MỊN SÀN ===
    if np.any(floor_mask):
        print("\n[2] Refining floor...")
        floor_indices = np.where(floor_mask)[0]
        floor_pts = points[floor_indices]
        
        # Tách sàn thành các cụm
        floor_instances = split_into_instances(floor_pts, eps=0.5)
        
        for inst_mask in floor_instances:
            inst_pts = floor_pts[inst_mask]
            inst_indices = floor_indices[inst_mask]
            projected, _ = refine_floor_ceiling(
                inst_pts, inst_indices, points, refined_classes, is_ceiling=False
            )
            refined_points[inst_indices] = projected
    
    # === 3. LÀM MỊN TRẦN ===
    if np.any(ceiling_mask):
        print("\n[3] Refining ceiling...")
        ceiling_indices = np.where(ceiling_mask)[0]
        ceiling_pts = points[ceiling_indices]
        
        ceiling_instances = split_into_instances(ceiling_pts, eps=0.5)
        
        for inst_mask in ceiling_instances:
            inst_pts = ceiling_pts[inst_mask]
            inst_indices = ceiling_indices[inst_mask]
            projected, _ = refine_floor_ceiling(
                inst_pts, inst_indices, points, refined_classes, is_ceiling=True
            )
            refined_points[inst_indices] = projected
    
    # === 4. KIỂM TRA CỬA SỔ ===
    if np.any(window_mask):
        print("\n[4] Checking windows...")
        window_indices = np.where(window_mask)[0]
        window_pts = points[window_indices]
        
        window_instances = split_into_instances(window_pts)
        print(f"  Found {len(window_instances)} window instance(s)")
        
        for inst_mask in window_instances:
            inst_pts = window_pts[inst_mask]
            inst_indices = window_indices[inst_mask]
            projected, _, is_valid = refine_window(
                inst_pts, inst_indices, refined_classes, prototype, points
            )
            
            if not is_valid:
                # Trả về tường
                refined_classes[inst_indices] = 'wall'
                print(f"  ⚠️ Window instance trả về wall")
            else:
                refined_points[inst_indices] = projected
                # Thử mở rộng cửa sổ lên trên nếu phần 'wall' phía trên
                # cùng mặt phẳng với cửa sổ
                normal, centroid, _ = fit_plane_ransac(
                    inst_pts, distance_threshold=WINDOW_PLANE_DISTANCE_THRESHOLD
                )
                if normal is not None and len(wall_points) > 0:
                    n_extended = extend_window_upward(
                        inst_pts, normal, centroid,
                        wall_points, wall_global_indices, refined_classes
                    )
                    if n_extended > 0:
                        print(f"  ⬆️ Window: mở rộng lên trên, gán lại "
                              f"{n_extended} điểm wall -> window")
    
    # === 5. KIỂM TRA BẢNG (WHITEBOARD) ===
    if np.any(whiteboard_mask):
        print("\n[5] Checking whiteboards...")
        whiteboard_indices = np.where(whiteboard_mask)[0]
        whiteboard_pts = points[whiteboard_indices]
        
        whiteboard_instances = split_into_instances(whiteboard_pts)
        print(f"  Found {len(whiteboard_instances)} whiteboard instance(s)")
        
        for inst_mask in whiteboard_instances:
            inst_pts = whiteboard_pts[inst_mask]
            inst_indices = whiteboard_indices[inst_mask]
            projected, _, is_valid = refine_whiteboard(
                inst_pts, inst_indices, refined_classes, wall_points, prototype, points
            )
            
            if not is_valid:
                # Trả về tường
                refined_classes[inst_indices] = 'wall'
                print(f"  ⚠️ Whiteboard instance trả về wall")
            else:
                refined_points[inst_indices] = projected
    
    # === 6. XUẤT KẾT QUẢ ===
    print("\n[6] Exporting refined PLY...")
    
    # Tạo màu mới
    refined_colors = np.full((len(refined_points), 3), 0.2)
    for cls, rgb in CLASS_COLORS.items():
        mask = (refined_classes == cls)
        if np.any(mask):
            refined_colors[mask] = rgb
    
    # Tạo point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(refined_points)
    pcd.colors = o3d.utility.Vector3dVector(refined_colors)
    
    # Lưu file
    output_name = ply_path.stem.replace('_corrected', '') + '_refined.ply'
    output_path = output_dir / output_name
    o3d.io.write_point_cloud(str(output_path), pcd)
    
    # Thống kê
    print(f"\n{'='*60}")
    print("REFINEMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Total points: {len(refined_points)}")
    
    # Đếm số điểm mỗi class
    print("\nClass distribution after refinement:")
    class_counts = {}
    valid_classes = sorted(cls for cls in set(refined_classes) if cls is not None)

    for cls in valid_classes:
        count = np.sum(refined_classes == cls)
        class_counts[cls] = int(count)
        print(f"  {cls}: {count} points")
    
    # So sánh với trước khi refine
    print("\nChanges:")
    changes = {}
    all_classes = sorted(
        cls
        for cls in (set(class_labels) | set(refined_classes))
        if cls is not None
    )

    for cls in all_classes:
        before = np.sum(class_labels == cls)
        after = np.sum(refined_classes == cls)

        if before != after:
            diff = after - before
            changes[cls] = {
                'before': int(before),
                'after': int(after),
                'diff': int(diff)
            }
            print(f"  {cls}: {before} -> {after} ({diff:+d})")
    
    elapsed_time = time.time() - start_time
    
    stats = {
        'total_points': int(len(refined_points)),
        'class_counts': class_counts,
        'changes': changes,
        'elapsed_time': elapsed_time
    }
    
    print(f"\nElapsed time: {elapsed_time:.2f} seconds")
    print("DONE.")
    
    return stats


def main():
    """Xử lý tất cả các file PLY trong thư mục ErrorMining"""
    print("="*80)
    print("REFINE BY GEOMETRY - BATCH PROCESSING")
    print("="*80)
    
    # Load prototype từ CSV
    print("\nLoading joint prototype from CSV...")
    if not PROTOTYPE_CSV_PATH.exists():
        print(f"❌ Prototype CSV not found: {PROTOTYPE_CSV_PATH}")
        print("Using default values...")
        prototype = {}
    else:
        try:
            prototype = load_prototype_from_csv(PROTOTYPE_CSV_PATH)
            print(f"  Loaded classes: {sorted(prototype.keys())}")
            for cls, data in prototype.items():
                print(f"    {cls}: {len(data['features'])} features")
        except Exception as e:
            print(f"  ⚠️ Error loading prototype: {e}")
            prototype = {}
    
    # Tìm tất cả file PLY trong thư mục ErrorMining
    ply_files = sorted(INPUT_DIR.glob("*.ply"))
    
    if not ply_files:
        print(f"❌ No PLY files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(ply_files)} PLY file(s) to process")
    
    # Checkpoint path
    checkpoint_path = CHECKPOINT_DIR / "refinement_checkpoint.json"
    
    # Statistics
    all_stats = {}
    processed_count = 0
    skipped_count = 0
    
    # Process each file
    for i, ply_path in enumerate(ply_files, 1):
        print(f"\n{'#'*80}")
        print(f"Processing {i}/{len(ply_files)}: {ply_path.name}")
        print(f"{'#'*80}")
        
        # Check if already processed
        if is_processed(checkpoint_path, ply_path.name):
            print(f"⏭️  File already processed. Skipping...")
            skipped_count += 1
            continue
        
        try:
            # Process file
            stats = process_single_file(ply_path, prototype, OUTPUT_DIR, checkpoint_path)
            all_stats[ply_path.name] = stats
            
            # Mark as processed
            mark_processed(checkpoint_path, ply_path.name, stats)
            processed_count += 1
            
            print(f"✅ Successfully processed: {ply_path.name}")
            
        except Exception as e:
            print(f"❌ Error processing {ply_path.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"Total files: {len(ply_files)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(ply_files) - processed_count - skipped_count}")
    
    if all_stats:
        print("\nDetailed statistics:")
        for filename, stats in all_stats.items():
            print(f"\n  {filename}:")
            print(f"    Total points: {stats['total_points']}")
            print(f"    Time: {stats['elapsed_time']:.2f}s")
            if stats['changes']:
                print(f"    Changes: {len(stats['changes'])} class(es) changed")
                for cls, change in stats['changes'].items():
                    print(f"      {cls}: {change['before']} -> {change['after']} ({change['diff']:+d})")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}")
    print(f"📝 Checkpoint saved to: {checkpoint_path}")
    print("="*80)
    print("DONE.")


if __name__ == "__main__":
    main()