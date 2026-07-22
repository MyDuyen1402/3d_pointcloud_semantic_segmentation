import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
import laspy

# Kiểm tra phiên bản Open3D
print(f"🔄 Open3D version: {o3d.__version__}")

# ====== chọn file ======
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Chọn file point cloud",
    filetypes=[("Point Cloud", "*.ply *.pcd *.las")]
)

if not file_path:
    print("Không chọn file!")
    exit()

# ====== load point cloud ======
ext = os.path.splitext(file_path)[1].lower()

if ext == ".las":
    print("📥 Đang đọc file LAS...")
    
    las = laspy.read(file_path)
    
    # Lấy tọa độ
    points = np.vstack((las.x, las.y, las.z)).transpose()
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Nếu có màu (optional)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        print("🎨 File có màu RGB")
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
        
        # Normalize về [0,1]
        colors = colors / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        print("⚠️ File không có màu, sẽ dùng Z coloring")

else:
    print("📥 Đang đọc file PLY/PCD...")
    pcd = o3d.io.read_point_cloud(file_path)
#pcd = o3d.io.read_point_cloud(file_path)

# Nếu đám mây bị lật ngược (chụp từ dưới lên), lật trục Z
# Ma trận biến đổi để lật theo trục Z: [[1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,1]]
flip_transform = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
pcd.transform(flip_transform)
print("🔄 Đã lật đám mây điểm theo trục Z để sửa hướng.")

# ====== tạo thư mục output ======
filename = os.path.splitext(os.path.basename(file_path))[0]
output_dir = f"captures/SensatUrban/train/Difference/{filename}"

os.makedirs(output_dir, exist_ok=True)

# ====== tính bounding box và khoảng cách hợp lý ======
bbox = pcd.get_axis_aligned_bounding_box()
bbox_extent = bbox.get_extent()  # [dx, dy, dz]
max_dim = max(bbox_extent)
center = bbox.get_center()

# Bán kính camera: 1.5 lần đường kính lớn nhất
R = 1.5 * max_dim
print(f"📏 Kích thước đám mây: {bbox_extent}, bán kính camera: {R:.2f}")

def move_left(vis):
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()

    extrinsic = np.asarray(cam.extrinsic).copy()

    right = extrinsic[:3, 0]
    extrinsic[:3, 3] -= right * 0.5

    cam.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam)
    return False


def move_right(vis):
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()

    extrinsic = np.asarray(cam.extrinsic).copy()

    right = extrinsic[:3, 0]
    extrinsic[:3, 3] += right * 0.5

    cam.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam)
    return False

def move_up(vis):
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()

    extrinsic = np.asarray(cam.extrinsic).copy()

    up = extrinsic[:3, 1]
    extrinsic[:3, 3] += up * 0.5

    cam.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam)
    return False


def move_down(vis):
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()

    extrinsic = np.asarray(cam.extrinsic).copy()

    up = extrinsic[:3, 1]
    extrinsic[:3, 3] -= up * 0.5

    cam.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(cam)
    return False

# ====== tạo camera positions theo yêu cầu ======
def generate_camera_positions(radius, center):
    """
    Tạo các vị trí camera từ trên nhìn xuống:
    - 1 góc trên đỉnh (90 độ)
    - 6 góc tầng cao (elevation 60°), xoay mỗi góc 60°
    - 5 góc tầng rất cao (elevation 75°), xoay mỗi góc 72°
    
    Tổng số: 1 + 6 + 5 = 12 góc, tất cả từ trên nhìn xuống
    """
    positions = []
    
    # 1. Góc nhìn từ trên xuống (elevation 90°)
    positions.append([center[0], center[1], center[2] + radius])
    
    # 2. Các góc tầng cao (elevation 60°), 6 góc xoay đều 60°
    elevation_high = np.radians(15)  # Từ trên nhìn xuống
    for i in range(6):
        azimuth = np.radians(60 * i)  # 0°, 60°, 120°, 180°, 240°, 300°
        x = center[0] + radius * np.cos(elevation_high) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation_high) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation_high)
        positions.append([x, y, z])
    
    # 3. Các góc tầng rất cao (elevation 75°), 5 góc xoay đều 72°
    elevation_very_high = np.radians(30)  # Từ trên nhìn xuống
    for i in range(5):
        azimuth = np.radians(72 * i)  # 0°, 72°, 144°, 216°, 288°
        x = center[0] + radius * np.cos(elevation_very_high) * np.cos(azimuth)
        y = center[1] + radius * np.cos(elevation_very_high) * np.sin(azimuth)
        z = center[2] + radius * np.sin(elevation_very_high)
        positions.append([x, y, z])
    
    return positions

# Sinh vị trí camera
camera_positions = generate_camera_positions(R, center)
print(f"📸 Tổng số góc chụp: {len(camera_positions)}")  # 1 + 6 + 5 = 12 góc

# ====== callback khi nhấn SPACE ======
def capture_views(vis):
    ctr = vis.get_view_control()
    
    print("📸 Bắt đầu chụp ảnh...")
    
    # Chỉ giữ các góc trên và ngang, bỏ góc dưới
    filtered_positions = []
    for cam_pos in camera_positions:
        # Bỏ các góc có z < center_z (nhìn từ dưới lên)
        if cam_pos[2] >= center[2]:  # Chỉ chụp từ ngang trở lên
            filtered_positions.append(cam_pos)
    
    for i, cam_pos in enumerate(filtered_positions):
        # Hướng camera nhìn về center
        front = center - cam_pos
        front = front / np.linalg.norm(front)
        
        # Tính up vector phù hợp
        up = np.array([0., 0., 1.])
        if abs(np.dot(front, up)) > 0.99:
            up = np.array([0., 1., 0.])

        front_norm = front / np.linalg.norm(front)
        if np.abs(np.dot(front_norm, up)) > 0.99:
            up = np.array([0, 1, 0])
        
        ctr.set_lookat(center)
        ctr.set_front(front)
        ctr.set_up(up)
        ctr.set_zoom(0.3)
        
        vis.poll_events()
        vis.update_renderer()
        
        filename = os.path.join(output_dir, f"view_{i+1:02d}.png")
        vis.capture_screen_image(filename)
        print(f"✔ Saved {filename}")
    
    print(f"✅ Done {len(filtered_positions)} ảnh!")
    return False

# ====== tạo visualizer ======
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="Point Cloud Viewer", width=1280, height=720)

vis.add_geometry(pcd)

# ====== TÙY CHỈNH RENDER VỚI ĐIỂM TRÒN ======
render_option = vis.get_render_option()

# THIẾT LẬP ĐIỂM TRÒN - CHO OPEN3D 0.19.0
try:
    # Thử với point_shape (cho Open3D 0.15+)
    render_option.point_shape = 'circle'
    print("✅ Đã thiết lập điểm hình tròn (point_shape='circle')")
except:
    try:
        # Một số phiên bản dùng 'point_shape' dạng số
        render_option.point_shape = o3d.visualization.PointShape.Circle
        print("✅ Đã thiết lập điểm hình tròn (PointShape.Circle)")
    except:
        print("⚠️ Không thể thiết lập điểm tròn, dùng điểm vuông nhưng tăng chất lượng")
        # Fallback: tăng kích thước và bật các tùy chọn khác
        pass

# Các tùy chỉnh khác
render_option.point_size = 3.0  # Điều chỉnh kích thước điểm
render_option.background_color = np.array([0.95, 0.95, 0.95])  # Nền sáng
render_option.light_on = True

# Màu sắc
if pcd.has_colors():
    render_option.point_color_option = o3d.visualization.PointColorOption.Color
else:
    render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate

# Bật anti-aliasing nếu có
try:
    render_option.anti_aliasing = True
except:
    pass

print("👉 Render options:")
print(f"   - Point size: {render_option.point_size}")
print(f"   - Background: {render_option.background_color}")
print(f"   - Light on: {render_option.light_on}")

# ====== Đăng ký phím SPACE ======
vis.register_key_callback(32, capture_views)

print(f"\n👉 Nhấn SPACE để chụp {len(camera_positions)} ảnh từ các góc đã định")
print("👉 Nhấn Q hoặc ESC để thoát")

vis.register_key_callback(265, move_up)
vis.register_key_callback(264, move_down)
vis.register_key_callback(263, move_left)
vis.register_key_callback(262, move_right)
vis.run()
vis.destroy_window()