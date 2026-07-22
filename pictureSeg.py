# 1. Cài đặt các thư viện cần thiết
# !pip install transformers torch torchvision pillow numpy matplotlib
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import os
from pathlib import Path
import json

# 2. Khởi tạo thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Đang chạy trên thiết bị: {device}\n")

# 3. Tải mô hình SegFormer
# Có 3 phiên bản (B0 nhẹ nhất, B5 nặng nhất)
# Chọn model phù hợp với GPU/CPU của bạn
print("📦 Tải mô hình SegFormer...")

# Model B3 - Balance giữa tốc độ và chính xác (Recommended)
model_name = "nvidia/segformer-b3-finetuned-ade-512-512"

# Nếu muốn nhanh hơn (cho CPU):
# model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

# Nếu muốn chính xác hơn (cần GPU 8GB+):
# model_name = "nvidia/segformer-b5-finetuned-ade-640-640"

try:
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    print(f"✅ Tải thành công: {model_name}\n")
except Exception as e:
    print(f"❌ Lỗi tải model: {e}")
    print("💡 Thử model nhẹ hơn...")
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
    model.eval()
    print(f"✅ Tải thành công: {model_name}\n")

# ===== CẤU HÌNH FOLDER =====
input_folder = r"D:\Thu\KhoaLuan\captures\Standford"  # <--- ĐỔI THÀNH FOLDER CỦA BẠN
output_folder = r"D:\Thu\KhoaLuan\results\Standford"   # <--- FOLDER LƯU KẾT QUẢ

# Tạo output folder
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, "masks"), exist_ok=True)  # Lưu class ID masks

# Danh sách các định dạng ảnh
supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.jfif', '.tiff')

# Lấy danh sách ảnh
image_files = sorted([f for f in os.listdir(input_folder) 
                      if f.lower().endswith(supported_formats)])

print(f"📁 Tìm thấy {len(image_files)} ảnh trong folder")
print(f"📂 Input: {input_folder}")
print(f"📂 Output: {output_folder}\n")

if len(image_files) == 0:
    print("❌ Không tìm thấy ảnh nào!")
    exit()

# ===== VÒNG LẶP XỬ LÝ TỪNG ẢNH =====
id2label = model.config.id2label
success_count = 0
error_count = 0

# Tạo bảng màu cho tất cả class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(id2label), 3))

# Tạo mapping: class_id -> color (cho visualization)
color_mapping = {int(class_id): colors[class_id].tolist() for class_id in range(len(id2label))}

# Tạo mapping: class_id -> class_name
label_mapping = {int(k): v for k, v in id2label.items()}

for idx, image_file in enumerate(image_files, 1):
    image_path = os.path.join(input_folder, image_file)
    
    print(f"[{idx}/{len(image_files)}] 🔄 {image_file}...", end=" ")
    
    try:
        # ===== LOAD ẢNH =====
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Resize nếu quá lớn (giúp tiết kiệm RAM)
        max_size = 1024
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # ===== DỰ ĐOÁN =====
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # ===== HẬU XỬ LÝ =====
        logits = outputs.logits
        
        # Upsample về kích thước ảnh
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        
        predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # ===== LƯU CLASS ID MASK (cho map.py sử dụng) =====
        mask_filename = f"mask_{Path(image_file).stem}.png"
        mask_path = os.path.join(output_folder, "masks", mask_filename)
        
        # Lưu class ID trực tiếp (0-149)
        mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
        mask_image.save(mask_path)
        
        # ===== VẼ KẾT QUẢ VISUALIZATION =====
        unique_labels = np.unique(predicted_mask)
        
        # Chuyển ảnh sang numpy array
        image_np = np.array(image)
        segmented_image = image_np.copy()
        
        # Tạo figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # Tạo legend patches
        legend_patches = []
        
        # Áp màu cho từng class
        for label_id in unique_labels:
            mask = (predicted_mask == label_id)
            color = colors[label_id]
            segmented_image[mask] = segmented_image[mask] * 0.5 + color * 0.5
            
            # Thêm vào legend
            label_name = id2label.get(label_id, f"Class_{label_id}")
            patch = mpatches.Patch(color=color / 255.0, label=label_name.upper())
            legend_patches.append(patch)
        
        # Hiển thị ảnh đã tô màu
        ax.imshow(segmented_image.astype(np.uint8))
        
        # Vẽ text label lên từng vùng
        for label_id in unique_labels:
            mask = (predicted_mask == label_id)
            y_indices, x_indices = np.where(mask)
            
            # Chỉ vẽ nếu diện tích đủ lớn
            if len(y_indices) > 500:
                center_y = int(np.mean(y_indices))
                center_x = int(np.mean(x_indices))
                label_name = id2label.get(label_id, f"Class_{label_id}")
                
                ax.text(center_x, center_y, label_name.upper(), 
                       color='white', fontsize=10, fontweight='bold',
                       ha='center', va='center',
                       bbox=dict(facecolor='black', alpha=0.7, 
                               edgecolor='white', boxstyle='round,pad=0.4',
                               linewidth=1.5))
        
        # Vẽ legend
        ax.legend(handles=legend_patches, loc='center left', 
                 bbox_to_anchor=(1.02, 0.5),
                 title="OBJECTS DETECTED", title_fontsize='13',
                 fontsize='10', borderaxespad=0.)
        
        # Config
        ax.axis('off')
        ax.set_title(f"Semantic Segmentation - {image_file}", 
                    fontsize=18, fontweight='bold', pad=20)
        
        # Lưu ảnh
        output_filename = f"segmented_{Path(image_file).stem}.png"
        output_path = os.path.join(output_folder, output_filename)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ Lưu: {output_filename} + mask")
        success_count += 1
        
    except Exception as e:
        print(f"❌ Lỗi: {str(e)[:50]}")
        error_count += 1
        continue

# ===== LƯU MAPPINGS =====
print("\n💾 Saving class mappings...")

# Lưu color mapping
color_mapping_path = os.path.join(output_folder, "class_color_mapping.json")
with open(color_mapping_path, 'w') as f:
    json.dump(color_mapping, f, indent=2)
print(f"✅ Saved: class_color_mapping.json")

# Lưu label mapping
label_mapping_path = os.path.join(output_folder, "class_label_mapping.json")
with open(label_mapping_path, 'w') as f:
    json.dump(label_mapping, f, indent=2)
print(f"✅ Saved: class_label_mapping.json")

# Lưu model info
model_info = {
    "model_name": model_name,
    "num_classes": len(id2label),
    "total_images": len(image_files),
    "successful": success_count,
    "failed": error_count,
    "class_names": {int(k): v for k, v in id2label.items()}
}
model_info_path = os.path.join(output_folder, "model_info.json")
with open(model_info_path, 'w') as f:
    json.dump(model_info, f, indent=2)
print(f"✅ Saved: model_info.json")

print(f"\n" + "="*70)
print(f"✅ Hoàn thành!")
print(f"   • Xử lý thành công: {success_count}/{len(image_files)}")
print(f"   • Lỗi: {error_count}/{len(image_files)}")
print(f"📂 Kết quả lưu: {output_folder}")
print(f"📂 Class ID masks: {os.path.join(output_folder, 'masks')}")
print(f"📂 Mappings: class_color_mapping.json, class_label_mapping.json")
print("="*70)