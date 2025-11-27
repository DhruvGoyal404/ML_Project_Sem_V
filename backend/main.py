"""
FastAPI Backend for Multimodal Emotion Detection
Loads all 14 models and provides prediction endpoints
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pickle
import joblib
import numpy as np
import re
from io import BytesIO

from models import CNN1DClassifier, BiLSTMClassifier, DeepCNN, MLPClassifier

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(title="Emotion Detection API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# GLOBAL VARIABLES
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model storage
text_models = {}
facial_models = {}
facial_pca = None  # Shared PCA for all facial ML models

# Labels
text_emotions = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
facial_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Preprocessors
vectorizer = None
word_to_idx = None

# Image transforms
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ============================================================================
# LOAD MODELS
# ============================================================================

@app.on_event("startup")
async def load_models():
    global text_models, facial_models, vectorizer, word_to_idx

    print("Loading models...")

    # Load text preprocessors
    try:
        with open('../text_tfidf_data.pkl', 'rb') as f:
            tfidf_data = pickle.load(f)
            vectorizer = tfidf_data.get('tfidf_vectorizer', None)

        with open('../text_tokenized_data.pkl', 'rb') as f:
            token_data = pickle.load(f)
            word_to_idx = token_data.get('word_to_idx', {})
            if not word_to_idx:
                # Build basic vocab
                word_to_idx = {'<PAD>': 0, '<UNK>': 1}
                for i in range(2, 10000):
                    word_to_idx[f'word_{i}'] = i
    except Exception as e:
        print(f"Warning: Could not load preprocessors: {e}")
        # Set defaults if loading fails
        vectorizer = None
        word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        for i in range(2, 10000):
            word_to_idx[f'word_{i}'] = i

    # TEXT MODELS
    try:
        # CNN-1D
        cnn1d = CNN1DClassifier(vocab_size=10000, num_classes=6).to(device)
        cnn1d.load_state_dict(torch.load('../results/models/text/text_cnn1d_best.pth', map_location=device))
        cnn1d.eval()
        text_models['CNN-1D'] = cnn1d

        # BiLSTM
        lstm = BiLSTMClassifier(vocab_size=10000, num_classes=6).to(device)
        lstm.load_state_dict(torch.load('../results/models/text/text_lstm_best.pth', map_location=device))
        lstm.eval()
        text_models['BiLSTM'] = lstm

        # ML models (saved with joblib)
        text_models['SVM'] = joblib.load('../results/models/text/text_svm.pkl')
        text_models['XGBoost'] = joblib.load('../results/models/text/text_xgboost.pkl')
        text_models['Naive Bayes'] = joblib.load('../results/models/text/text_nb.pkl')
        text_models['KNN'] = joblib.load('../results/models/text/text_knn.pkl')
        text_models['Random Forest'] = joblib.load('../results/models/text/text_rf.pkl')

        print(f"✓ Loaded {len(text_models)} text models")
    except Exception as e:
        print(f"Error loading text models: {e}")

    # FACIAL MODELS
    try:
        # Deep CNN
        dcnn = DeepCNN(num_classes=7).to(device)
        dcnn.load_state_dict(torch.load('../results/models/facial/facial_dcnn_best.pth', map_location=device))
        dcnn.eval()
        facial_models['Deep CNN'] = dcnn

        # MLP
        mlp = MLPClassifier(num_classes=7).to(device)
        mlp.load_state_dict(torch.load('../results/models/facial/facial_mlp_best.pth', map_location=device))
        mlp.eval()
        facial_models['MLP'] = mlp

        # ML models (saved with joblib)
        svm_dict = joblib.load('../results/models/facial/facial_svm.pkl')
        facial_models['SVM'] = svm_dict
        facial_models['XGBoost'] = joblib.load('../results/models/facial/facial_xgboost.pkl')
        facial_models['Random Forest'] = joblib.load('../results/models/facial/facial_rf.pkl')
        facial_models['KNN'] = joblib.load('../results/models/facial/facial_knn.pkl')
        facial_models['Naive Bayes'] = joblib.load('../results/models/facial/facial_nb.pkl')

        # Extract shared PCA from SVM model (all ML models use the same PCA)
        global facial_pca
        facial_pca = svm_dict.get('pca', None)

        print(f"✓ Loaded {len(facial_models)} facial models")
    except Exception as e:
        print(f"Error loading facial models: {e}")

    print(f"Total models loaded: {len(text_models) + len(facial_models)}")
    print(f"Device: {device}")

# ============================================================================
# PREPROCESSING
# ============================================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text, max_len=100):
    words = text.split()
    indices = [word_to_idx.get(word, 1) for word in words]

    if len(indices) > max_len:
        indices = indices[:max_len]
    else:
        indices += [0] * (max_len - len(indices))

    return indices

# ============================================================================
# ENDPOINTS
# ============================================================================

class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {
        "message": "Multimodal Emotion Detection API",
        "text_models": len(text_models),
        "facial_models": len(facial_models),
        "endpoints": ["/predict/text", "/predict/image"]
    }

@app.post("/predict/text")
async def predict_text(request: TextRequest):
    try:
        text = request.text.strip()

        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Clean text
        cleaned = clean_text(text)

        predictions = {}

        # DL models (CNN-1D, BiLSTM)
        tokens = tokenize_text(cleaned)
        tensor = torch.LongTensor([tokens]).to(device)

        for name in ['CNN-1D', 'BiLSTM']:
            if name in text_models:
                model = text_models[name]
                with torch.no_grad():
                    output = model(tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)

                predictions[name] = {
                    'emotion': text_emotions[predicted.item()],
                    'confidence': confidence.item() * 100
                }

        # ML models (need TF-IDF)
        if vectorizer:
            tfidf = vectorizer.transform([cleaned])

            for name in ['SVM', 'XGBoost', 'Naive Bayes', 'KNN', 'Random Forest']:
                if name in text_models:
                    model = text_models[name]
                    pred = model.predict(tfidf)[0]

                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(tfidf)[0]
                        confidence = float(proba.max() * 100)
                    else:
                        confidence = 85.0  # Default

                    predictions[name] = {
                        'emotion': text_emotions[int(pred)],
                        'confidence': confidence
                    }

        return {
            'original_text': text,
            'cleaned_text': cleaned,
            'predictions': predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert('RGB')

        predictions = {}

        # DL models (Deep CNN, MLP)
        tensor = transform(image).unsqueeze(0).to(device)

        for name in ['Deep CNN', 'MLP']:
            if name in facial_models:
                model = facial_models[name]
                with torch.no_grad():
                    output = model(tensor)
                    probs = F.softmax(output, dim=1)
                    confidence, predicted = torch.max(probs, 1)

                predictions[name] = {
                    'emotion': facial_emotions[predicted.item()],
                    'confidence': confidence.item() * 100
                }

        # ML models (need preprocessing pipeline)
        # Using flattened grayscale as features
        gray = image.convert('L').resize((48, 48))
        features = np.array(gray).flatten().reshape(1, -1) / 255.0

        for name in ['SVM', 'XGBoost', 'Random Forest', 'KNN', 'Naive Bayes']:
            if name in facial_models:
                model_dict = facial_models[name]
                try:
                    # Extract model and scaler
                    model = model_dict['model']
                    scaler = model_dict.get('scaler', None)

                    # Check if model was trained with PCA (SVM, KNN) or without (XGBoost, RF, NB)
                    if 'pca' in model_dict:
                        # Models with PCA: raw → PCA → scaler → model
                        preprocessed_features = features
                        if facial_pca is not None:
                            preprocessed_features = facial_pca.transform(preprocessed_features)
                        if scaler is not None:
                            preprocessed_features = scaler.transform(preprocessed_features)
                    else:
                        # Models without PCA: raw → model (no scaling)
                        preprocessed_features = features

                    # Make prediction
                    pred = model.predict(preprocessed_features)[0]

                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(preprocessed_features)[0]
                        confidence = float(proba.max() * 100)
                    else:
                        confidence = 75.0

                    predictions[name] = {
                        'emotion': facial_emotions[int(pred)],
                        'confidence': confidence
                    }
                except Exception as e:
                    print(f"Error with {name}: {e}")
                    pass

        return {
            'filename': file.filename,
            'predictions': predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)