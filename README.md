# Real-Time Object Detection with YOLOv3 🔍📷

This project uses YOLOv3 and OpenCV to perform real-time object detection from a webcam feed.

## 👩‍💻 Technologies Used
- Python
- OpenCV
- YOLOv3
- Numpy

## 🗂 Folder Structure
real-time-object-detection/
│
├── src/
│ └── object_detection.py
│
├── data/
│ ├── yolov3.cfg
│ ├── coco.names
│ └── yolov3.weights

## 🚀 How to Run

```bash
# Step 1: Create virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate

# Step 2: Install dependencies
pip install opencv-python numpy

# Step 3: Run the detection script
cd src
python object_detection.py