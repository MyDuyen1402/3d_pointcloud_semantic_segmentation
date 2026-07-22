# ============================================================
# DEBUG SCRIPT - ANALYZE AREA 5 STRUCTURE
# ============================================================

import os
from pathlib import Path
from collections import defaultdict

ROOT_DIR = "E:\\MyDuyen\\KLTN\\mapped_pcd"
AREA = "Area_5"

print("\n" + "="*80)
print(f"ANALYZING DIRECTORY STRUCTURE: {AREA}")
print("="*80)

area_path = Path(ROOT_DIR) / AREA

if not area_path.exists():
    print(f"\n❌ ERROR: {area_path} does not exist!")
    print(f"\nAvailable areas in {ROOT_DIR}:")
    root = Path(ROOT_DIR)
    if root.exists():
        for item in sorted(root.iterdir()):
            if item.is_dir():
                print(f"  - {item.name}")
    exit(1)

print(f"\nPath: {area_path}")
print(f"Exists: {area_path.exists()}")

# List all subdirectories (rooms)
rooms = [item for item in area_path.iterdir() if item.is_dir()]
print(f"\n{'='*80}")
print(f"ROOMS IN {AREA}: {len(rooms)}")
print(f"{'='*80}\n")

# Track file statistics
room_stats = defaultdict(lambda: {"points": 0, "labels": 0, "conf": 0, "other": 0})

for room_path in sorted(rooms):
    room_name = room_path.name
    
    # Count files by type
    all_files = list(room_path.glob("*"))
    
    points_files = list(room_path.glob("*_points.npy"))
    labels_files = list(room_path.glob("*_labels.npy"))
    conf_files = list(room_path.glob("*_confidences.npy"))
    other_files = [f for f in all_files if f.suffix not in ['.npy', '.txt']]
    
    room_stats[room_name]["points"] = len(points_files)
    room_stats[room_name]["labels"] = len(labels_files)
    room_stats[room_name]["conf"] = len(conf_files)
    room_stats[room_name]["other"] = len(other_files)
    
    # Check if complete
    is_complete = (len(points_files) > 0 and len(labels_files) > 0 and len(conf_files) > 0)
    status = "✓ COMPLETE" if is_complete else "✗ INCOMPLETE"
    
    print(f"{room_name:<40} {status}")
    print(f"  _points.npy:      {len(points_files)}")
    print(f"  _labels.npy:      {len(labels_files)}")
    print(f"  _confidences.npy: {len(conf_files)}")
    
    if len(points_files) > 0:
        points_file = points_files[0]
        size_mb = points_file.stat().st_size / (1024**2)
        print(f"  Total size:       {size_mb:.2f} MB")
    
    if not is_complete:
        print(f"  ⚠️  Missing files!")
        if len(points_files) == 0:
            print(f"      - *_points.npy")
        if len(labels_files) == 0:
            print(f"      - *_labels.npy")
        if len(conf_files) == 0:
            print(f"      - *_confidences.npy")
    
    print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)

complete_rooms = sum(1 for stats in room_stats.values() 
                     if stats["points"] > 0 and stats["labels"] > 0 and stats["conf"] > 0)
total_rooms = len(room_stats)

print(f"\nTotal rooms: {total_rooms}")
print(f"Complete rooms (all 3 files): {complete_rooms}")
print(f"Incomplete rooms: {total_rooms - complete_rooms}")

if complete_rooms == 0:
    print("\n⚠️  WARNING: NO COMPLETE ROOMS FOUND!")
    print("\nPossible issues:")
    print("  1. Files are in different location than expected")
    print("  2. File naming convention is different")
    print("  3. Some files are missing from all rooms")
    print("\nAction: Check file structure and naming convention")
else:
    print(f"\n✓ Found {complete_rooms} valid room(s) for testing")

print("\n" + "="*80)
