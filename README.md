# 3D Semantic Segmentation with Multi-View Foundation Models

This repository implements a 4-stage pipeline for semantic segmentation of 3D point clouds by combining 2D foundation models, multi-view projection, and 3D refinement. The goal is to transform raw 3D scenes into accurate semantic labels without relying solely on expensive 3D annotations.

## Overview

The project follows a research-oriented workflow:

1. Multi-view rendering from a 3D point cloud
2. 2D semantic segmentation using foundation models such as Grounding DINO and SAM 2
3. Back-projection of 2D predictions into 3D space
4. 3D refinement using a PointCNN-based network to correct noisy labels

This pipeline is particularly suitable for indoor scenes such as the S3DIS dataset, where dense 3D geometry and multi-view observations can be combined to improve segmentation quality.

## Key Concepts

### Stage 1: Multi-view Rendering
A 3D scene is observed from multiple virtual camera viewpoints. This produces a set of 2D images that capture different parts of the scene from different perspectives.

### Stage 2: 2D Semantic Segmentation
The rendered images are processed with 2D foundation models to generate pixel-level semantic masks and confidence information.

### Stage 3: 2D-to-3D Back-projection
The 2D predictions are mapped back to the original 3D point cloud using camera intrinsics and extrinsics. This step produces preliminary labels, which may contain noise or inconsistencies.

### Stage 4: 3D Refinement
A 3D deep learning model learns local geometric structure and corrects noisy labels based on spatial consistency and context.

## Features

- Zero-shot 2D segmentation using foundation models
- Multi-view label aggregation for 3D point clouds
- Back-projection from 2D image space to 3D geometry
- PointCNN-based refinement for noisy pseudo-labels
- Support for large indoor 3D scenes and S3DIS-style data

## Project Structure

```text
3d_semantic_segmentation-PointCNN/
├── DINOSeg.py                 # Stage 2: 2D segmentation with Grounding DINO + SAM 2
├── DINOMapping.py             # Stage 3: 2D-to-3D back-projection and label fusion
├── PointCNN_new.py            # Stage 4: PointCNN-based 3D refinement
├── refine_by_geometry.py      # Geometry-based refinement utilities
├── pointcnn_model.py          # PointCNN model definition
├── mapping.py                 # Mapping helpers
├── EvaluateDino.py            # Evaluation utilities
├── EvaluateResults.py         # Result evaluation utilities
└── ...
```

## Requirements

The project requires:

- Python 3.8 or newer
- PyTorch
- OpenCV
- Open3D
- NumPy
- scikit-learn
- Pillow
- Matplotlib
- tqdm

For 2D segmentation, the following external components are also used:

- Grounding DINO
- SAM 2

## Installation

Install the Python dependencies:

```bash
pip install torch torchvision opencv-python pillow matplotlib numpy tqdm open3d scikit-learn
```

Then configure the Grounding DINO and SAM 2 paths in [DINOSeg.py](DINOSeg.py) according to your local environment.

## Data Preparation

The pipeline expects:

- A set of rendered RGB images from multiple viewpoints
- Camera pose information in JSON format
- A 3D point cloud representing the scene
- Optional ground-truth labels for evaluation

A typical folder layout is:

```text
YOUR_PROJECT_NAME/
├── INPUTS/
│   ├── images/
│   │   ├── view_0.png
│   │   ├── view_1.png
│   │   └── camera_poses.json
│   └── pointcloud/
│       └── scene.ply or scene.txt
└── OUTPUTS/
    └── stage4_refined/
```

## Quick Start

### 1. Prepare the input data
Place your rendered images, camera poses, and point cloud into the expected input folders.

### 2. Run 2D segmentation
Execute:

```bash
python DINOSeg.py
```

This step generates semantic masks and confidence maps for the multi-view images.

### 3. Run 2D-to-3D mapping
Execute:

```bash
python DINOMapping.py
```

This step projects 2D segmentation results back to the original 3D point cloud and creates noisy pseudo-labels.

### 4. Run 3D refinement
Execute:

```bash
python PointCNN_new.py
```

This step trains or runs the refinement model and produces the final segmented point cloud.

## Outputs

The pipeline can produce:

- 2D semantic masks for each viewpoint
- Back-projected 3D pseudo-labels
- Confidence values for each projected point
- Refined 3D segmentation results
- Evaluation metrics when ground truth is available

## Notes

- If the 2D views and camera poses are already available, Stage 1 can be skipped.
- The quality of the final 3D segmentation depends strongly on the quality of the initial 2D predictions and camera calibration.
- Some scripts require manual configuration of local paths and model checkpoints.

## License

This project is intended for research and educational purposes.

## Acknowledgement

This work is based on the idea of combining multi-view 2D segmentation with 3D geometric refinement to improve semantic segmentation of point clouds.
