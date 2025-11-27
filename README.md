# MultiModal Emotion Detection System

A comprehensive emotion detection system that analyzes emotions from both **text** and **facial expressions** using multiple machine learning and deep learning models.

## 📊 Project Overview

This project implements **14 different models** (7 for text, 7 for facial) to detect and classify emotions:

### Text Emotion Models (6 emotions)
- **Deep Learning**: CNN-1D, BiLSTM
- **Machine Learning**: SVM, XGBoost, Random Forest, Naive Bayes, KNN

### Facial Emotion Models (7 emotions)
- **Deep Learning**: Deep CNN, MLP
- **Machine Learning**: SVM, XGBoost, Random Forest, Naive Bayes, KNN

---

## 🎯 Results

### Text Emotion Detection
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| **CNN-1D** | **91.15%** | 34.7s |
| BiLSTM | 91.10% | 48.9s |
| SVM | 89.55% | 44.3s |
| XGBoost | 85.50% | 30.7s |
| Naive Bayes | 81.80% | 0.2s |
| KNN | 74.50% | 39.4s |
| Random Forest | 49.60% | 47.8s |

### Facial Emotion Detection
| Model | Accuracy | Training Time |
|-------|----------|---------------|
| **Deep CNN** | **66.34%** | 8.6 min |
| MLP | 42.59% | 6.8 min |
| XGBoost | 39.36% | - |
| SVM | 35.72% | - |
| Random Forest | 34.68% | - |
| KNN | 19.27% | - |
| Naive Bayes | 20.09% | - |

---

## 📁 Project Structure

```
ML_EVAL/
├── main.ipynb                          # Main training notebook (all 14 models)
├── webcam.py                           # Real-time facial emotion detection
├── README.md                           # This file
├── VIVA_UPDATE.txt                     # Comprehensive viva preparation guide
├── UPDATES.txt                         # Implementation guide for extensions
│
├── results/
│   ├── models/
│   │   ├── text/                       # Trained text models (.pth, .pkl)
│   │   └── facial/                     # Trained facial models (.pth, .pkl)
│   │
│   ├── metrics/
│   │   ├── all_models_results.csv      # Combined results
│   │   ├── text_all_results.csv
│   │   └── facial_all_results.csv
│   │
│   ├── graphs/
│   │   ├── training_curves/            # Loss/accuracy plots
│   │   └── master_comparison.png       # Final comparison chart
│   │
│   └── frontend/                       # React web application
│       ├── src/
│       ├── public/
│       └── package.json
│
└── backend/                            # FastAPI backend
    ├── main.py
    ├── models.py
    └── requirements.txt
```

---

## 🚀 Quick Start

### 1. Training Models (Jupyter Notebook)
```bash
# Open main.ipynb and run all cells
jupyter notebook main.ipynb
```

### 2. Webcam Real-time Detection
```bash
python webcam.py
```

### 3. Web Application

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python main.py
# Runs on http://localhost:8000
```

**Frontend:**
```bash
cd results/frontend
npm install
npm run dev
# Runs on http://localhost:3000
```

---

## 📚 Datasets

### Text Dataset
- **Source**: Emotion Detection Text Corpus
- **Total Samples**: 416,809
- **Emotions**: 6 classes (sadness, joy, love, anger, fear, surprise)
- **Split**: 90% train, 10% test

### Facial Dataset
- **Source**: FER-2013 (Facial Expression Recognition)
- **Total Samples**: 35,887 images
- **Train**: 28,709 | **Test**: 7,178
- **Image Size**: 48×48 grayscale
- **Emotions**: 7 classes (angry, disgust, fear, happy, neutral, sad, surprise)

---

## 🛠️ Tech Stack

### Training & Models
- **PyTorch** 2.5.1 (Deep Learning)
- **scikit-learn** (Machine Learning)
- **XGBoost** (Gradient Boosting)
- **NLTK** (Text Processing)

### Web Application
- **Backend**: FastAPI, Uvicorn
- **Frontend**: React 18, Vite, Tailwind CSS
- **Real-time**: OpenCV (webcam)

### Environment
- **GPU**: NVIDIA RTX 4090 (24GB)
- **Platform**: Vast.ai
- **Python**: 3.12

---

## 📖 Features

### 1. Text Analysis
- Input any text (tweets, messages, reviews)
- Get predictions from 7 different models
- Confidence scores for each prediction
- Clean comparison table

### 2. Facial Analysis
- **Upload Image**: Analyze emotions from photos
- **Live Camera**: Real-time facial emotion detection
- Predictions from 7 facial models
- Visual confidence bars

### 3. Webcam Application
- Real-time emotion detection from webcam
- Face detection using Haar Cascades
- Live probability bars for all emotions
- Color-coded emotion display
- Screenshot capability

---

## 🎓 Key Learnings

### Why Text Performs Better (91% vs 66%)?
1. **Dataset Quality**: Text corpus is cleaner, facial dataset has ~30% label noise
2. **Data Quantity**: 416K text samples vs 35K facial images
3. **Feature Richness**: 10,000 word vocabulary vs 2,304 pixels
4. **Task Difficulty**: Text has clear indicators, facial expressions are subtle

### Why Deep Learning > Machine Learning?
- **Text**: DL (91%) vs ML (89%) - small gap, both work
- **Facial**: DL (66%) vs ML (39%) - HUGE gap, DL essential for images
- **Reason**: CNNs learn spatial features automatically, HOG features insufficient

---

## 🔧 Installation

```bash
# Clone repository
git clone <repo-url>
cd ML_EVAL

# Install Python dependencies
pip install torch torchvision scikit-learn xgboost opencv-python nltk pandas numpy matplotlib seaborn

# Install frontend dependencies
cd results/frontend
npm install

# Install backend dependencies
cd ../../backend
pip install -r requirements.txt
```

---

## 📊 Model Architectures

### CNN-1D (Text)
```
Embedding(10000, 128)
→ Conv1D(256, kernel=3)
→ MaxPool1D(2)
→ Conv1D(128, kernel=3)
→ GlobalMaxPool
→ Dense(128)
→ Dense(6)
```

### Deep CNN (Facial)
```
Conv2D(128) → Pool → Dropout
→ Conv2D(256) → Pool → Dropout
→ Conv2D(512) → Pool → Dropout
→ Conv2D(512) → Pool → Dropout
→ Dense(512) → Dense(256) → Dense(7)
```

Total Parameters: 9.8M

---

## 👥 Authors

**Dhruv Goyal**
**Jeevant Verma**
Thapar Institute of Engineering & Technology, Patiala

---

## 📝 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

- FER-2013 dataset creators
- PyTorch and scikit-learn communities
- Thapar Institute faculty
- Vast.ai for GPU resources

---

## 📞 Contact

For questions or feedback:
- GitHub: [DhruvGoyal404]
- Email: [dhruv621999goyal@gmail.com]

---

**Last Updated**: November 2024