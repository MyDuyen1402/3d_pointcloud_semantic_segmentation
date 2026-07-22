#!/usr/bin/env python3
# ============================================================
# PATH CONFIGURATION HELPER
# ============================================================

import os
import sys
from pathlib import Path

print("\n" + "="*80)
print("CHECKING SYSTEM PATHS AND CONFIGURATION")
print("="*80)

# Current working directory
cwd = os.getcwd()
print(f"\nCurrent working directory:")
print(f"  {cwd}")

# Home directory
home = os.path.expanduser("~")
print(f"\nHome directory:")
print(f"  {home}")

# Script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"\nScript directory:")
print(f"  {script_dir}")

# Check for data directories
print(f"\n{'='*80}")
print("SEARCHING FOR DATA DIRECTORIES")
print(f"{'='*80}")

search_paths = [
    "/compute_home/slurmdang12/datasets/mapped_pcd",
    "/home/slurmdang12/datasets/mapped_pcd",
    os.path.expanduser("~/datasets/mapped_pcd"),
    os.path.join(cwd, "datasets/mapped_pcd"),
    os.path.join(cwd, "../datasets/mapped_pcd"),
    os.path.join(script_dir, "datasets/mapped_pcd"),
]

print("\nSearching for mapped_pcd (training data):")
data_root_found = None
for path in search_paths:
    expanded_path = os.path.expanduser(path)
    exists = os.path.exists(expanded_path)
    status = "✓ FOUND" if exists else "✗ not found"
    print(f"  {status}: {expanded_path}")
    if exists and not data_root_found:
        data_root_found = expanded_path

if data_root_found:
    print(f"\n✓ Using: {data_root_found}")
    
    # List areas in mapped_pcd
    areas = [d for d in os.listdir(data_root_found) if os.path.isdir(os.path.join(data_root_found, d)) and d.startswith("Area")]
    print(f"\nAreas in {os.path.basename(data_root_found)}:")
    for area in sorted(areas):
        area_path = os.path.join(data_root_found, area)
        rooms = [d for d in os.listdir(area_path) if os.path.isdir(os.path.join(area_path, d))]
        print(f"  - {area}: {len(rooms)} rooms")
else:
    print(f"\n✗ NOT FOUND - Please check your data location")

# Check for model files
print(f"\n{'='*80}")
print("SEARCHING FOR MODEL FILES")
print(f"{'='*80}")

model_search_paths = [
    "/compute_home/slurmdang12/datasets/pointcnn_refinement/pointcnn_refinement_final.pth",
    "/home/slurmdang12/datasets/pointcnn_refinement/pointcnn_refinement_final.pth",
    os.path.expanduser("~/datasets/pointcnn_refinement/pointcnn_refinement_final.pth"),
    os.path.join(cwd, "pointcnn_refinement_final.pth"),
    os.path.join(script_dir, "pointcnn_refinement_final.pth"),
]

print("\nSearching for pointcnn_refinement_final.pth:")
model_found = None
for path in model_search_paths:
    expanded_path = os.path.expanduser(path)
    exists = os.path.exists(expanded_path)
    status = "✓ FOUND" if exists else "✗ not found"
    if exists:
        size_mb = os.path.getsize(expanded_path) / (1024**2)
        print(f"  {status}: {expanded_path} ({size_mb:.2f} MB)")
        if not model_found:
            model_found = expanded_path
    else:
        print(f"  {status}: {expanded_path}")

if model_found:
    print(f"\n✓ Using: {model_found}")
else:
    print(f"\n✗ NOT FOUND - Please check your model location")

# Summary and recommendations
print(f"\n{'='*80}")
print("CONFIGURATION SUMMARY")
print(f"{'='*80}")

if data_root_found and model_found:
    print("\n✓ All required files found!")
    print(f"\nYou can now run:")
    print(f"  python test_final_model.py")
else:
    print("\n⚠️  Some required files are missing!")
    
    if not data_root_found:
        print("\nTo fix mapped_pcd location:")
        print("  1. Check your data storage location")
        print("  2. Update ROOT_DIR in test_final_model.py")
        print("  3. Or symlink to one of the expected locations")
    
    if not model_found:
        print("\nTo fix model file location:")
        print("  1. Check your trained model location")
        print("  2. Update FINAL_MODEL_PATH in test_final_model.py")
        print("  3. Or copy model to current directory as: pointcnn_refinement_final.pth")

print("\n" + "="*80 + "\n")
