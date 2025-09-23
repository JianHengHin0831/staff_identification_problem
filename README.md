# Staff Identification Problem

This project presents a comprehensive, multi-stage pipeline for identifying and tracking staff members in a top-down video stream. The solution is designed to be robust against common real-world challenges such as small object detection, occlusion, and significant appearance changes, while strictly adhering to a zero data leakage policy for model training.

## Features

- **Multi-Stage Processing:** A modular pipeline separates the problem into distinct, manageable stages: Tracking, Clustering, Identification, and Refinement.
- **Robust Tracking:** Utilizes SAHI (Slicing Aided Hyper Inference) with a YOLOv8L detector for superior small object recall, ensuring even distant or partially occluded individuals are tracked.
- **Intelligent Clustering:** A custom spatio-temporal clustering algorithm merges fragmented tracks based on time, space, and appearance similarity (using ResNet50 embeddings).
- **Zero Data Leakage Identification:** A lightweight YOLOv8n detector is fine-tuned exclusively on a synthetically generated dataset to recognize staff name tags, ensuring the evaluation on the test video is fair and unbiased.
- **Track Completion:** A final non-AI backtracking and interpolation step fills in missing frames for identified staff members, creating a more continuous trajectory.
- **Automated Workflow:** The entire process is orchestrated through a series of numbered Python scripts, ensuring clarity and reproducibility.

## Project Structure

```
.
├── external_person_images/ # Directory for external images used in data synthesis
├── final_output/           # Contains the final report, annotated video, and JSON data
├── helper/                 # Helper modules and utility functions
├── output/                 # Intermediate output files from each stage
├── testing/                # Contains intermediate video from each stage
│
├── 01_run_tracker.py       # Stage 1: Initial detection and tracking
├── 02_run_cluster.py       # Stage 2: Spatio-temporal clustering of tracks
├── 03_manual_merge.py      # (Optional) Human-in-the-loop track merging
├── 04_generate_finetune_dataset.py # Stage 3a: Synthetic data generation
├── 05_finetune_detector.py # Stage 3b: Fine-tuning the staff tag detector
├── 06_staff_identification.py # Stage 3c: Identifying staff using the fine-tuned model
├── 07_staff_tracking.py    # Stage 4: Trajectory refinement and completion
├── 08_final_report.py      # Stage 5: Generation of final report and video
│
├── config.py               # Configuration file for paths and parameters
├── AI Evaluation doc.pdf   # Documentation for this model
├── requirements.txt        # Project dependencies
└── yolov8l.pt              # Pre-trained YOLOv8 model weights
```

## Getting Started

### Prerequisites

- Python 3.9+
- PyTorch
- OpenCV
- An NVIDIA GPU is recommended for faster processing.

### Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/JianHengHin0831/staff_identification_problem.git
    cd staff_identification_problem
    ```

2.  Install PyTorch (choose one depending on your environment):

    CPU only:

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

    CUDA 12.1: （For other CUDA versions, check PyTorch official installation guide）

    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

3.  Install the remaining dependencies:
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```

### Execution Workflow

The pipeline is designed to be run sequentially. Execute the scripts in numerical order:

```bash
# Stage 1: Generate initial tracks
python 01_run_tracker.py

# Stage 2: Cluster fragmented tracks
python 02_run_cluster.py
python 03_manual_merge.py (if needs)

# Stage 3: Generate synthetic data, fine-tune, and identify staff
python 04_generate_finetune_dataset.py
python 05_finetune_detector.py
python 06_staff_identification.py

# Stage 4: Refine and complete staff tracks
python 07_staff_tracking.py

# Stage 5: Generate the final report and annotated video
python 08_final_report.py
```

After running the complete workflow, the final outputs, including the annotated video and text report, will be available in the `final_output/` directory.

## Key Technologies

- **Object Detection:** YOLOv8L, SAHI
- **Object Tracking:** DeepSort / ByteTrack principles
- **Feature Extraction:** ResNet50 (for clustering)
- **Fine-tuning:** PyTorch, YOLOv8n
- **Data Synthesis & Augmentation:** OpenCV, NumPy
