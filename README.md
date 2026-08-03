# Seizure Prediction — SBAM + MSTM Real-World Simulation Suite

An interactive showcase web application demonstrating the clinical and real-world deployment of a deep learning framework designed to predict and classify epileptic seizures.

## Project Motivation & Architecture
Epileptic seizures are marked by sudden, runaway synchronization of large neural networks. Forewarning patients and clinical teams requires detecting subtle, pre-onset (pre-ictal) patterns. This project implements a dual-branch neural network designed to capture these signatures:
1. **SBAM (Synchronization-Based Attention Mechanism)**: Permutes input features to treat them as tokens, using Multi-Head Attention to compute spatial synchronization weights across brain regions.
2. **MSTM (Multi-Scale Temporal Modeling)**: Employs 1D dilated causal convolutions in parallel at three scales (kernels 3, 5, and 7; dilations 1, 2, and 4) to capture signal shifts across fast, medium, and slow temporal frequencies.
3. **Fusion Classifier**: Extracts penultimate SBM and MSTM features, feeds them through temporal + spatial attention layers, and classifies them using a Dense softmax classification head.

---

## ⚠️ Important Label Leakage Caveat
The original training dataset (`train.csv`) from the HMS Harmful Brain Activity Classification dataset contains expert annotator vote percentages (`seizure_vote`, `lpd_vote`, etc.). Because these percentages encode the target class almost directly, models trained on them achieve an unrealistic **98.8% accuracy**.

To present this transparently for portfolio review, this application supports two modes:
1. **As Trained**: Replicates the original notebook pipeline using all 14 features (98%+ accuracy).
2. **Leakage-corrected**: Trains the identical network architecture with all vote columns removed, establishing an honest baseline (~42% accuracy) on structural metadata.

---

## Workspace Structure
```
backend/
  app/main.py           FastAPI server serving live predictions, Grad-CAM 1D weights, and direct feature gradients.
  train_model.py        Modular training pipeline script producing saved weights for both As-Trained and Corrected modes.
  saved_models/         Saved model weights, scaling parameters, encoders, and JSON metrics for both modes.
  requirements.txt      Python package dependencies (TensorFlow, FastAPI, scikit-learn, etc.).
  Dockerfile            Lightweight deployment configuration.
frontend/
  index.html            narrative portfolio UI presenting ICU bedside monitoring, clinical decision support, and wearable alert simulators.
  script.js             Real-time EEG canvas rendering, SBAM sync animations, interactive sliders, and live API fetch logic.
train.csv               Dataset of consensus metadata.
Seizure_prediction_using_MSTM_&_SBAM.ipynb  Original Jupyter Notebook.
```

---

## Running Locally

### 1. Backend API
Create a virtual environment (Python 3.11 or 3.12 is recommended to prevent path conflicts) and install dependencies:
```bash
# From workspace root
py -3.12 -m venv venv
.\venv\Scripts\activate
pip install -r backend/requirements.txt
```
To start the FastAPI server:
```bash
cd backend
..\venv\Scripts\python -m uvicorn app.main:app --port 8000
```
API docs will be available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Interface
You can serve the `frontend/` directory with any static server:
```bash
cd frontend
python -m http.server 5173
```
Open `http://localhost:5173` in your browser.

---

## Cloud Deployment

- **Backend**: Deploy the `/backend` folder to **Render**, **Railway**, or **Fly.io** using the supplied `Dockerfile`.
- **Frontend**: Deploy the static `/frontend` folder to **Vercel**, **Netlify**, or **GitHub Pages**. (Update `API_BASE` in `frontend/script.js` to point to your deployed backend URL).
