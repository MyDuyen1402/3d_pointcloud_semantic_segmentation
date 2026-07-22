import open3d as o3d
import numpy as np
import os
import json
from pathlib import Path
import copy

def capture_single_room(points, colors, bbox_min, bbox_max, output_dir, room_name, n_views=12):
    """
    Chụp 1 room/segment với code hoàn toàn giống capture_room_views cũ
    """
    try:
        # Tạo point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # ===== Tính toán thông số camera (GIỐNG HỆT CODE CŨ) =====
        center = np.array([
            (bbox_min[0] + bbox_max[0]) / 2,
            (bbox_min[1] + bbox_max[1]) / 2,
            (bbox_min[2] + bbox_max[2]) / 2
        ])
        
        z_min, z_max = points[:, 2].min(), points[:, 2].max()
        
        size_x = bbox_max[0] - bbox_min[0]
        size_y = bbox_max[1] - bbox_min[1]
        
        min_size = min(size_x, size_y)
        radius = min_size * 0.5
        radius = np.clip(radius, 2.0, 20.0)
        
        print(f"\n   📐 Thông số segment:")
        print(f"      BBox: X[{bbox_min[0]:.2f}, {bbox_max[0]:.2f}], Y[{bbox_min[1]:.2f}, {bbox_max[1]:.2f}]")
        print(f"      Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
        print(f"      Radius: {radius:.2f}")
        
        # ===== Các góc chụp =====
        angles = [i * (2 * np.pi / n_views) for i in range(n_views)]
        camera_poses = []
        view_idx = 0
        
        def capture_all(vis):
            nonlocal view_idx, camera_poses
            
            print(f"\n   📸 Bắt đầu chụp segment...")
            ctr = vis.get_view_control()
            
            for _ in range(5):
                vis.poll_events()
                vis.update_renderer()
            
            for i, angle in enumerate(angles):
                angle_deg = int(np.degrees(angle))
                
                cam_x = center[0] + radius * np.cos(angle)
                cam_y = center[1] + radius * np.sin(angle)
                cam_z = center[2]
                camera_pos = np.array([cam_x, cam_y, cam_z])
                
                direction = center - camera_pos
                direction = direction / np.linalg.norm(direction)
                
                ctr.set_lookat(center.tolist())
                ctr.set_front((-direction).tolist())
                ctr.set_up([0.0, 0.0, 1.0])
                
                zoom_value = max(0.1, min(1.0, radius / 15.0))
                ctr.set_zoom(zoom_value)
                
                for _ in range(5):
                    vis.poll_events()
                    vis.update_renderer()
                
                try:
                    camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
                    
                    pose = {
                        'view_idx': view_idx,
                        'type': 'surround',
                        'angle_deg': angle_deg,
                        'angle_rad': float(angle),
                        'camera_position': camera_pos.tolist(),
                        'look_at': center.tolist(),
                        'up': [0.0, 0.0, 1.0],
                        'intrinsic_matrix': camera_params.intrinsic.intrinsic_matrix.tolist(),
                        'extrinsic_matrix': camera_params.extrinsic.tolist()
                    }
                    camera_poses.append(pose)
                except Exception as e:
                    print(f"      Warning: {e}")
                
                filename = f"view_{view_idx:02d}_angle_{angle_deg:03d}deg.png"
                vis.capture_screen_image(os.path.join(output_dir, filename))
                print(f"      ✅ [{view_idx+1}/{n_views}] {filename}")
                
                view_idx += 1
            
            # Lưu camera poses
            poses_file = os.path.join(output_dir, "camera_poses.json")
            metadata = {
                'room_name': room_name,
                'num_views': n_views,
                'bounding_box_center': center.tolist(),
                'bbox_min': bbox_min.tolist(),
                'bbox_max': bbox_max.tolist(),
                'size_x': float(size_x),
                'size_y': float(size_y),
                'min_size': float(min_size),
                'radius': float(radius),
                'camera_poses': camera_poses
            }
            
            with open(poses_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return False
        
        def set_initial_view(vis):
            ctr = vis.get_view_control()
            ctr.set_lookat(center.tolist())
            ctr.set_front([1.0, 0.0, 0.0])
            ctr.set_up([0.0, 0.0, 1.0])
            zoom_value = max(0.1, min(1.0, radius / 15.0))
            ctr.set_zoom(zoom_value)
            return False
        
        print(f"\n   🎮 Nhấn SPACE để chụp segment này")
        
        o3d.visualization.draw_geometries_with_key_callbacks(
            [pcd],
            {
                ord(' '): capture_all,
                ord('V'): set_initial_view
            },
            window_name=f"Capture - {room_name}",
            width=1280,
            height=720
        )
        
        return len(camera_poses)
        
    except Exception as e:
        print(f"   ❌ Lỗi: {e}")
        return 0


def capture_room_views(room_folder, output_root=None, n_views=12):
    """
    Chụp ảnh từ nhiều góc cho một room point cloud
    """
    
    result = {
        'success': False,
        'output_dir': None,
        'num_images': 0,
        'camera_poses_file': None,
        'error': None
    }
    
    try:
        # ===== Kiểm tra đầu vào =====
        if not os.path.exists(room_folder):
            raise ValueError(f"Room folder không tồn tại: {room_folder}")
        
        # ===== Load data =====
        print(f"\n📂 Đang load dữ liệu từ: {room_folder}")
        all_points = []
        all_colors = []
        
        txt_files = [f for f in os.listdir(room_folder) if f.endswith(".txt")]
        if len(txt_files) == 0:
            raise ValueError(f"Không tìm thấy file .txt nào trong {room_folder}")
        
        print(f"   Tìm thấy {len(txt_files)} file .txt")
        
        for file in txt_files:
            data = np.loadtxt(os.path.join(room_folder, file))
            if data.shape[1] >= 6:
                all_points.append(data[:, :3])
                all_colors.append(data[:, 3:6] / 255.0)
            else:
                print(f"   Warning: {file} có {data.shape[1]} columns, bỏ qua...")
        
        if len(all_points) == 0:
            raise ValueError("Không load được point cloud nào")
        
        points = np.vstack(all_points)
        colors = np.vstack(all_colors)
        
        print(f"   Đã load {len(points):,} points")
        
        # ===== Tính toán bounding box =====
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        
        size_x = bbox_max[0] - bbox_min[0]
        size_y = bbox_max[1] - bbox_min[1]
        size_z = bbox_max[2] - bbox_min[2]
        
        aspect_ratio = max(size_x, size_y) / min(size_x, size_y)
        
        print(f"\n📐 Thông số room:")
        print(f"   Bounding Box:")
        print(f"     X: [{bbox_min[0]:.2f}, {bbox_max[0]:.2f}] -> rộng: {size_x:.2f}")
        print(f"     Y: [{bbox_min[1]:.2f}, {bbox_max[1]:.2f}] -> dài: {size_y:.2f}")
        print(f"     Z: [{bbox_min[2]:.2f}, {bbox_max[2]:.2f}] -> cao: {size_z:.2f}")
        print(f"   Tỉ lệ dài/hẹp: {aspect_ratio:.2f}")
        
        room_name = os.path.basename(room_folder)
        
        # ===== KIỂM TRA VÀ CHIA NHỎ NẾU LÀ HALLWAY =====
        if aspect_ratio > 3:
            # Hallway: chia thành nhiều đoạn
            num_segments = int(np.ceil(aspect_ratio)/2)  # Chia mỗi đoạn ~2 lần chiều hẹp
            num_segments = max(3, min(num_segments, 8))   # Giới hạn từ 3-8 đoạn
            
            print(f"\n⚠️ PHÁT HIỆN HALLWAY! (tỉ lệ {aspect_ratio:.1f}:1)")
            print(f"   → Chia làm {num_segments} đoạn dọc theo chiều dài")
            
            # Xác định chiều dài (trục nào dài hơn)
            if size_x > size_y:
                axis_long = 'x'
                long_min = bbox_min[0]
                long_max = bbox_max[0]
                short_min = bbox_min[1]
                short_max = bbox_max[1]
            else:
                axis_long = 'y'
                long_min = bbox_min[1]
                long_max = bbox_max[1]
                short_min = bbox_min[0]
                short_max = bbox_max[0]
            
            # Chiều dài mỗi đoạn
            segment_length = (long_max - long_min) / num_segments
            
            # Tạo output folder
            if output_root is None:
                base_output_dir = os.path.join(room_folder, "captures")
            else:
                base_output_dir = os.path.join(output_root, room_name)
            os.makedirs(base_output_dir, exist_ok=True)
            
            total_images = 0
            
            # Xử lý từng đoạn
            for seg_idx in range(num_segments):
                seg_start = long_min + seg_idx * segment_length
                seg_end = seg_start + segment_length
                
                # Tạo bounding box cho đoạn này
                if axis_long == 'x':
                    seg_bbox_min = np.array([seg_start, short_min, bbox_min[2]])
                    seg_bbox_max = np.array([seg_end, short_max, bbox_max[2]])
                    # Lọc points
                    mask = (points[:, 0] >= seg_bbox_min[0]) & (points[:, 0] <= seg_bbox_max[0])
                else:
                    seg_bbox_min = np.array([short_min, seg_start, bbox_min[2]])
                    seg_bbox_max = np.array([short_max, seg_end, bbox_max[2]])
                    # Lọc points
                    mask = (points[:, 1] >= seg_bbox_min[1]) & (points[:, 1] <= seg_bbox_max[1])
                
                seg_points = points[mask]
                seg_colors = colors[mask]
                
                if len(seg_points) < 500:
                    print(f"   Segment {seg_idx+1}: chỉ có {len(seg_points)} points - bỏ qua")
                    continue
                
                # ⭐ TÍNH TÂM VÀ RADIUS TỪ SEGMENT ĐƯỢC CHIA (không phải toàn phòng)
                seg_actual_min = seg_points.min(axis=0)
                seg_actual_max = seg_points.max(axis=0)
                
                seg_center = (seg_actual_min + seg_actual_max) / 2
                seg_size_x = seg_actual_max[0] - seg_actual_min[0]
                seg_size_y = seg_actual_max[1] - seg_actual_min[1]
                seg_min_size = min(seg_size_x, seg_size_y)
                seg_radius = seg_min_size * 0.35
                seg_radius = np.clip(seg_radius, 2.0, 20.0)
                
                print(f"\n   📍 Segment {seg_idx+1}/{num_segments}: {len(seg_points):,} points")
                print(f"      Tâm: [{seg_center[0]:.2f}, {seg_center[1]:.2f}, {seg_center[2]:.2f}]")
                print(f"      Radius: {seg_radius:.2f}")
                
                # Output folder cho segment
                seg_output_dir = os.path.join(base_output_dir, f"segment_{seg_idx+1:02d}")
                os.makedirs(seg_output_dir, exist_ok=True)
                
                # Chụp segment này (dùng tâm và radius từ segment)
                num_images = capture_single_room(
                    seg_points, seg_colors, 
                    seg_actual_min, seg_actual_max,  # ⭐ Dùng actual bbox từ segment
                    seg_output_dir, 
                    f"{room_name}_seg{seg_idx+1}", 
                    n_views
                )
                
                total_images += num_images
                print(f"   ✅ Segment {seg_idx+1} hoàn thành: {num_images} ảnh")
            
            result['success'] = True
            result['output_dir'] = base_output_dir
            result['num_images'] = total_images
            
            print(f"\n✅ HOÀN THÀNH HALLWAY!")
            print(f"   Tổng số ảnh: {total_images} từ {num_segments} segments")
            
        else:
            # ===== PHÒNG BÌNH THƯỜNG: GIỮ NGUYÊN CODE CŨ =====
            print(f"\n✅ PHÒNG BÌNH THƯỜNG")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Tính toán thông số camera
            center = np.array([
                (bbox_min[0] + bbox_max[0]) / 2,
                (bbox_min[1] + bbox_max[1]) / 2,
                (bbox_min[2] + bbox_max[2]) / 2
            ])
            
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            
            size_x = bbox_max[0] - bbox_min[0]
            size_y = bbox_max[1] - bbox_min[1]
            
            min_size = min(size_x, size_y)
            radius = min_size * 0.35 
            radius = np.clip(radius, 2.0, 20.0)
            
            print(f"   Center: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
            print(f"   Radius: {radius:.2f}")
            
            # Tạo output directory
            if output_root is None:
                output_dir = os.path.join(room_folder, "captures")
            else:
                output_dir = os.path.join(output_root, room_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Các góc chụp
            angles = [i * (2 * np.pi / n_views) for i in range(n_views)]
            camera_poses = []
            view_idx = 0
            
            def capture_all(vis):
                nonlocal view_idx, camera_poses
                
                print("\n📸 Bắt đầu chụp ảnh...")
                ctr = vis.get_view_control()
                
                for _ in range(5):
                    vis.poll_events()
                    vis.update_renderer()
                
                for i, angle in enumerate(angles):
                    angle_deg = int(np.degrees(angle))
                    
                    cam_x = center[0] + radius * np.cos(angle)
                    cam_y = center[1] + radius * np.sin(angle)
                    cam_z = center[2]
                    camera_pos = np.array([cam_x, cam_y, cam_z])
                    
                    direction = center - camera_pos
                    direction = direction / np.linalg.norm(direction)
                    
                    ctr.set_lookat(center.tolist())
                    ctr.set_front((-direction).tolist())
                    ctr.set_up([0.0, 0.0, 1.0])
                    
                    zoom_value = max(0.1, min(1.0, radius / 15.0))
                    ctr.set_zoom(zoom_value)
                    
                    for _ in range(5):
                        vis.poll_events()
                        vis.update_renderer()
                    
                    try:
                        camera_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
                        
                        pose = {
                            'view_idx': view_idx,
                            'type': 'surround',
                            'angle_deg': angle_deg,
                            'angle_rad': float(angle),
                            'camera_position': camera_pos.tolist(),
                            'look_at': center.tolist(),
                            'up': [0.0, 0.0, 1.0],
                            'intrinsic_matrix': camera_params.intrinsic.intrinsic_matrix.tolist(),
                            'extrinsic_matrix': camera_params.extrinsic.tolist()
                        }
                        camera_poses.append(pose)
                    except Exception as e:
                        print(f"   Warning: {e}")
                    
                    filename = f"view_{view_idx:02d}_angle_{angle_deg:03d}deg.png"
                    vis.capture_screen_image(os.path.join(output_dir, filename))
                    print(f"   ✅ [{view_idx+1}/{n_views}] {filename}")
                    
                    view_idx += 1
                
                poses_file = os.path.join(output_dir, "camera_poses.json")
                metadata = {
                    'room_name': room_name,
                    'num_views': n_views,
                    'bounding_box_center': center.tolist(),
                    'bbox_min': bbox_min.tolist(),
                    'bbox_max': bbox_max.tolist(),
                    'size_x': float(size_x),
                    'size_y': float(size_y),
                    'min_size': float(min_size),
                    'radius': float(radius),
                    'camera_poses': camera_poses
                }
                
                with open(poses_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                result['success'] = True
                result['output_dir'] = output_dir
                result['num_images'] = len(camera_poses)
                result['camera_poses_file'] = poses_file
                
                return False
            
            def set_initial_view(vis):
                ctr = vis.get_view_control()
                ctr.set_lookat(center.tolist())
                ctr.set_front([1.0, 0.0, 0.0])
                ctr.set_up([0.0, 0.0, 1.0])
                zoom_value = max(0.1, min(1.0, radius / 15.0))
                ctr.set_zoom(zoom_value)
                return False
            
            print("\n🎮 Điều khiển:")
            print("   - Nhấn SPACE để bắt đầu chụp ảnh")
            print("   - Nhấn V để reset view")
            print("   - Nhấn ESC để thoát\n")
            
            o3d.visualization.draw_geometries_with_key_callbacks(
                [pcd],
                {
                    ord(' '): capture_all,
                    ord('V'): set_initial_view
                },
                window_name=f"Capture Views - {room_name}",
                width=1280,
                height=720
            )
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        result['error'] = str(e)
        import traceback
        traceback.print_exc()
    
    return result


def capture_multiple_rooms(area_folder, output_root=None, n_views=12):
    """
    Chụp ảnh cho tất cả các room trong một area
    
    Parameters:
    -----------
    area_folder : str
        Đường dẫn đến folder Area chứa các room folders
    output_root : str, optional
        Đường dẫn thư mục gốc để lưu ảnh
    n_views : int, default=12
        Số góc chụp xung quanh cho mỗi room
    
    Returns:
    --------
    list : Kết quả cho từng room
    """
    
    if not os.path.exists(area_folder):
        print(f"❌ Area folder không tồn tại: {area_folder}")
        return []
    
    # Lấy tất cả các room folders
    room_folders = sorted([
        os.path.join(area_folder, f) for f in os.listdir(area_folder)
        if os.path.isdir(os.path.join(area_folder, f))
    ])
    
    print(f"\n{'='*70}")
    print(f"Tìm thấy {len(room_folders)} rooms trong {area_folder}")
    print(f"{'='*70}")
    
    results = []
    
    for i, room_folder in enumerate(room_folders, 1):
        room_name = os.path.basename(room_folder)
        print(f"\n{'#'*70}")
        print(f"[{i}/{len(room_folders)}] Processing room: {room_name}")
        print(f"{'#'*70}")
        
        result = capture_room_views(room_folder, output_root, n_views)
        results.append({
            'room_name': room_name,
            'room_folder': room_folder,
            'result': result
        })
        
        if result['success']:
            print(f"✅ Thành công: {room_name} - {result['num_images']} ảnh")
        else:
            print(f"❌ Thất bại: {room_name} - {result['error']}")
    
    # ===== Tổng kết =====
    print(f"\n{'='*70}")
    print("TỔNG KẾT:")
    print(f"{'='*70}")
    success_count = sum(1 for r in results if r['result']['success'])
    print(f"✅ Thành công: {success_count}/{len(results)} rooms")
    print(f"❌ Thất bại: {len(results) - success_count}/{len(results)} rooms")
    
    # ===== Save report =====
    if output_root:
        os.makedirs(output_root, exist_ok=True)
        report_file = os.path.join(output_root, "capture_report.json")
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n📊 Report: {report_file}")
    
    return results


# ===== SỬ DỤNG =====
if __name__ == "__main__":
    
    # ===== Cách 1: Chụp một room duy nhất =====
    # room_folder = r"D:\Thu\KhoaLuan\data\Stanford3dDataset_v1.2_Aligned_Version\Area_1\office_1"
    # output_root = r"D:\Thu\KhoaLuan\captures\Standford"
    
    # result = capture_room_views(
    #     room_folder=room_folder,
    #     output_root=output_root,
    #     n_views=12
    # )
    
    # if result['success']:
    #     print(f"\n✅ Chụp thành công!")
    #     print(f"   Output: {result['output_dir']}")
    #     print(f"   Số ảnh: {result['num_images']}")
    # else:
    #     print(f"\n❌ Chụp thất bại: {result['error']}")
    
    # ===== Cách 2: Chụp tất cả các room trong Area =====
    area_folder = r"D:\Thu\KhoaLuan\data\Stanford3dDataset_v1.2_Aligned_Version\Area_5"
    output_root = r"D:\Thu\KhoaLuan\captures\Standford\Area_5"
    
    results = capture_multiple_rooms(area_folder, output_root=output_root, n_views=12)