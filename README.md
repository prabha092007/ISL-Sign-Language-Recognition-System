# 🤟 ISL Sign Language Recognition System

Real-time Indian Sign Language (ISL) recognition using MediaPipe, TensorFlow and OpenCV.

## 📌 Features
- Detects **A-Z alphabets** (26 signs)
- Detects **0-9 numbers** (10 signs)
- Detects **10 words** (hello, thanks, yes, no, help, please, sorry, good, bad, more)
- Works in **real-time** using webcam
- Supports **1 or 2 hands**
- **99%+ accuracy** on test data
- Fully **offline** — no internet needed

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.10 | Programming language |
| MediaPipe | Hand landmark detection |
| TensorFlow/Keras | Neural network model |
| OpenCV | Webcam and display |
| scikit-learn | Data preprocessing |

## 📁 Project Structure
isl-sign-language-recognition/

│

├── data_collection.py   # Stage 1 - Collect hand landmark data

├── train_model.py       # Stage 2 - Train neural network

├── inference.py         # Stage 3 - Live real-time recognition

└── requirements.txt     # Python dependencies

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/prabha092007/isl-sign-language-recognition.git
cd isl-sign-language-recognition

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 How to Run

### Stage 1 — Collect Data
```bash
python data_collection.py
```
- Press keys to record each sign
- 400 samples per class

### Stage 2 — Train Model
```bash
python train_model.py
```
- Trains neural network automatically
- Best model saved to `model/` folder

### Stage 3 — Live Recognition
```bash
python inference.py
```

## 🎮 Controls
| Key | Action |
|-----|--------|
| SPACE | Clear current word |
| ENTER | Confirm word to sentence |
| BACKSPACE | Delete last character |
| w | Add recognised word |
| q | Quit |

## 📊 Model Performance
- Test Accuracy: **99%+**
- Classes: **46 total** (26 letters + 10 numbers + 10 words)
- Training samples: **18,400+**

## 👤 Author
**Prabha** — [@prabha092007](https://github.com/prabha092007)


