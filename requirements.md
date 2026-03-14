# Requirements for Multi-View Lab Video Analysis

## 1. Python Version
- Python 3.10 or higher

---

## 2. Core Libraries
These are required for basic video processing and handling metadata:

- `opencv-python` – Video loading, frame extraction, and manipulation
- `numpy` – Array computations for frames and tasks
- `pandas` – Metadata management and tabular data
- `moviepy` – Video clip creation and editing
- `json` – Reading/writing metadata (built-in Python)
- `pathlib` – File paths handling (built-in Python)

---

## 3. Optional / Advanced Libraries
These are for automatic task detection, hand tracking, and visualization:

- `torch` – Deep learning framework (for task segmentation or activity classification)
- `torchvision` – Models and datasets for video/image analysis
- `mediapipe` – Hand tracking / pose estimation for first-person view
- `scikit-learn` – Clustering and data analysis
- `matplotlib` / `seaborn` – Plotting timelines or task visualizations
- `detectron2` – Advanced object detection in video (optional)

---

## 4. System Dependencies
Some Python libraries require external programs:

- **FFmpeg** – Required by `moviepy` for video reading and writing
  - Install via:
    - macOS: `brew install ffmpeg`
    - Linux (Ubuntu/Debian): `sudo apt install ffmpeg`
    - Windows: download from [https://ffmpeg.org](https://ffmpeg.org)

---

## 5. Notes
- All Python libraries can be installed via pip:

```bash
pip install opencv-python numpy pandas moviepy torch torchvision mediapipe scikit-learn matplotlib seaborn