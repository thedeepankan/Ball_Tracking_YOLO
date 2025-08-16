# Ball Tracking with YOLO v12 🏀

A real-time ball tracking application using **YOLOv12** with **CUDA acceleration**.

![Demo](assets/demo.gif)

---

## 🚀 Features
- ⚡ Real-time ball and player detection  
- ✨ Glowing ball trail effects  
- 🖥️ CUDA GPU acceleration  
- 🔢 Player counting  

---

## 📋 Requirements
- NVIDIA GPU with **CUDA 12.1+**  
- Python **3.8+**  
- **8GB+ RAM**  

---

## ⚙️ Quick Setup

### 1. Verify CUDA Installation
```bash
nvidia-smi
```

### Step 2: Create Virtual Environment
```bash
python -m venv ball_tracking_env
# Windows
ball_tracking_env\Scripts\activate
# Linux/Mac
source ball_tracking_env/bin/activate
```
### Step 3: Install PyTorch with CUDA

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```
### Step 4: Install Requirements.txt
```bash
pip install -r requirements.txt

```

# Usage

## Default Video

```bash
python main.py
```


## 🎮 Controls
- Press **q** to quit
- Detection runs automatically

---

## 🛠️ Troubleshooting

**CUDA not available:**

## Uninstall existing PyTorch packages:
```bash
pip uninstall torch torchvision torchaudio
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```


# Project Structure

Ball_Tracking_YOLO/
├── assets/
│   ├── demo.gif
│   └── test.mp4
├── main.py
├── requirements.txt
└── README.md