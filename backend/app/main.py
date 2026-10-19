"""
FastAPI backend for the seizure prediction real-world showcase.
Serves both As Trained and Leakage-corrected model branches and fusion models.
"""
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "saved_models"

# Detect if static frontend assets are co-located (e.g. for unified deployment or local run)
frontend_candidates = [BASE_DIR.parent / "frontend", BASE_DIR / "frontend"]
frontend_dir = next((p for p in frontend_candidates if p.exists() and (p / "index.html").exists()), None)

app = FastAPI(title="Seizure Prediction API (SBAM + MSTM Fusion)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Load model artifacts for both modes at startup
# ----------------------------------------------------------------------
pipelines = {}
available_modes = ["as_trained", "leakage_corrected"]

for mode in available_modes:
    mode_dir = MODELS_DIR / mode
    if mode_dir.exists():
        try:
            sync_model = tf.keras.models.load_model(mode_dir / "sync_model.keras")
            tcn_mstm_model = tf.keras.models.load_model(mode_dir / "tcn_mstm_model.keras")
            fusion_model = tf.keras.models.load_model(mode_dir / "fusion_model.keras")
            
            with open(mode_dir / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open(mode_dir / "label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            with open(mode_dir / "feature_columns.json") as f:
                feature_columns = json.load(f)
            with open(mode_dir / "metrics.json") as f:
                metrics = json.load(f)
                
            sbm_feat_extractor = tf.keras.models.Model(
                inputs=sync_model.input, outputs=sync_model.get_layer("sbm_dense").output
            )
            mstm_feat_extractor = tf.keras.models.Model(
                inputs=tcn_mstm_model.input, outputs=tcn_mstm_model.get_layer("mstm_dense").output
            )
            
            pipelines[mode] = {
                "sync_model": sync_model,
                "tcn_mstm_model": tcn_mstm_model,
                "fusion_model": fusion_model,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_columns": feature_columns,
                "metrics": metrics,
                "sbm_feat_extractor": sbm_feat_extractor,
                "mstm_feat_extractor": mstm_feat_extractor
            }
            print(f"Successfully loaded {mode} pipeline.")
        except Exception as e:
            print(f"Error loading {mode} pipeline: {e}")

# Load sample records
SAMPLE_RECORDS = []
samples_path = MODELS_DIR / "sample_records.json"
if samples_path.exists():
    with open(samples_path) as f:
        SAMPLE_RECORDS = json.load(f)

class PredictRequest(BaseModel):
    mode: str = "as_trained"
    features: Dict[str, float]

def compute_gradcam_1d(fusion_model, sbm_features, mstm_features, class_index: int) -> List[float]:
    """
    Computes 1D Grad-CAM over the conv1d_sbm layer in the Fusion model.
    """
    # Create gradient model
    grad_model = tf.keras.models.Model(
        inputs=fusion_model.inputs,
        outputs=[fusion_model.get_layer("conv1d_sbm").output, fusion_model.output]
    )
    
    sbm_tensor = tf.convert_to_tensor(sbm_features, dtype=tf.float32)
    mstm_tensor = tf.convert_to_tensor(mstm_features, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([sbm_tensor, mstm_tensor])
        loss = predictions[:, class_index]
        
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0)
    conv_outputs = conv_outputs[0].numpy()
    
    for i in range(conv_outputs.shape[-1]):
        conv_outputs[:, i] *= pooled_grads[i].numpy()
        
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap)
    if max_val != 0:
        heatmap /= max_val
    return heatmap.tolist()

def compute_feature_gradients(sync_model, tcn_mstm_model, fusion_model, x_scaled, class_index: int) -> List[float]:
    """
    Computes direct input gradients of the target class prediction w.r.t the scaled input features.
    This provides a direct attribution of input feature importance.
    """
    x_tensor = tf.convert_to_tensor(x_scaled, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        # Recreate forward pass through the entire pipeline
        sbm_dense = sync_model.get_layer("sbm_dense").output
        sbm_fe_model = tf.keras.models.Model(inputs=sync_model.input, outputs=sbm_dense)
        
        mstm_dense = tcn_mstm_model.get_layer("mstm_dense").output
        mstm_fe_model = tf.keras.models.Model(inputs=tcn_mstm_model.input, outputs=mstm_dense)
        
        sbm_feats = sbm_fe_model(x_tensor)
        mstm_feats = mstm_fe_model(x_tensor)
        
        probs = fusion_model([sbm_feats, mstm_feats])
        loss = probs[:, class_index]
        
    grads = tape.gradient(loss, x_tensor)
    importance = tf.abs(grads)[0, :, 0].numpy()
    max_val = np.max(importance)
    if max_val != 0:
        importance /= max_val
    return importance.tolist()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Seizure Prediction API is healthy",
        "available_modes": list(pipelines.keys())
    }

if not frontend_dir:
    @app.get("/")
    def root():
        return {"status": "ok", "message": "Seizure Prediction API — see /docs for endpoints"}

@app.get("/samples")
def get_samples():
    """Real held-out rows so the frontend can offer 'try a real example'."""
    return SAMPLE_RECORDS

@app.get("/model-info")
def model_info(mode: str = "as_trained"):
    if mode not in pipelines:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Available: {list(pipelines.keys())}")
    
    pipe = pipelines[mode]
    return {
        "architecture": {
            "sbam": "Synchronization-Based Attention: MultiHeadAttention across feature/channel axis + Conv1D stack",
            "mstm": "Multi-Scale Temporal Modeling: parallel dilated Conv1D at 3 receptive-field scales",
            "fusion": "Concatenates SBAM + MSTM penultimate features, temporal + spatial attention, Dense classification head",
        },
        **pipe["metrics"]
    }

@app.post("/predict")
def predict(req: PredictRequest):
    mode = req.mode
    if mode not in pipelines:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Available: {list(pipelines.keys())}")
        
    pipe = pipelines[mode]
    scaler = pipe["scaler"]
    label_encoder = pipe["label_encoder"]
    feature_columns = pipe["feature_columns"]
    sbm_feat_extractor = pipe["sbm_feat_extractor"]
    mstm_feat_extractor = pipe["mstm_feat_extractor"]
    fusion_model = pipe["fusion_model"]
    sync_model = pipe["sync_model"]
    tcn_mstm_model = pipe["tcn_mstm_model"]
    
    # Extract values in correct feature column order
    missing = [c for c in feature_columns if c not in req.features]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
        
    row = np.array([[req.features[c] for c in feature_columns]], dtype="float32")
    scaled = scaler.transform(row)
    x_scaled = np.expand_dims(scaled, axis=-1)  # (1, F, 1)
    
    # Feature extraction
    sbm_features = sbm_feat_extractor.predict(x_scaled, verbose=0)
    mstm_features = mstm_feat_extractor.predict(x_scaled, verbose=0)
    
    # Fusion prediction
    probs = fusion_model.predict([sbm_features, mstm_features], verbose=0)[0]
    
    classes = label_encoder.classes_.tolist()
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    class_idx = int(np.argmax(probs))
    predicted_class = classes[class_idx]
    
    # Compute explainability weights
    gradcam_weights = compute_gradcam_1d(fusion_model, sbm_features, mstm_features, class_idx)
    feature_importance = compute_feature_gradients(sync_model, tcn_mstm_model, fusion_model, x_scaled, class_idx)
    
    # Map feature importance back to column names
    feature_importance_dict = {col: float(imp) for col, imp in zip(feature_columns, feature_importance)}
    
    return {
        "predicted_class": predicted_class,
        "probabilities": prob_dict,
        "gradcam_weights": gradcam_weights,
        "feature_importance": feature_importance_dict
    }

# ----------------------------------------------------------------------
# Mount static frontend if co-located
# ----------------------------------------------------------------------
if frontend_dir:
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
