"""
FastAPI backend for the seizure prediction real-world showcase.
Serves both As Trained and Leakage-corrected model branches and fusion models.
Optimized for high-throughput, low-latency inference on CPU.
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
                
            # Pre-compile feature extractors once at startup
            sbm_feat_extractor = tf.keras.models.Model(
                inputs=sync_model.input, outputs=sync_model.get_layer("sbm_dense").output
            )
            mstm_feat_extractor = tf.keras.models.Model(
                inputs=tcn_mstm_model.input, outputs=tcn_mstm_model.get_layer("mstm_dense").output
            )
            
            # Pre-compile Grad-CAM model once at startup to avoid rebuilding graph on each request
            grad_model = tf.keras.models.Model(
                inputs=fusion_model.inputs,
                outputs=[fusion_model.get_layer("conv1d_sbm").output, fusion_model.output]
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
                "mstm_feat_extractor": mstm_feat_extractor,
                "grad_model": grad_model
            }
            
            # Warm up execution graphs to eliminate first-request latency
            try:
                dummy_x = tf.zeros((1, len(feature_columns), 1), dtype=tf.float32)
                dummy_sbm = sbm_feat_extractor(dummy_x, training=False)
                dummy_mstm = mstm_feat_extractor(dummy_x, training=False)
                _ = fusion_model([dummy_sbm, dummy_mstm], training=False)
                
                with tf.GradientTape() as tape:
                    co, pr = grad_model([dummy_sbm, dummy_mstm], training=False)
                    loss = pr[:, 0]
                _ = tape.gradient(loss, co)
                
                with tf.GradientTape() as tape:
                    tape.watch(dummy_x)
                    s = sbm_feat_extractor(dummy_x, training=False)
                    m = mstm_feat_extractor(dummy_x, training=False)
                    p = fusion_model([s, m], training=False)
                    loss = p[:, 0]
                _ = tape.gradient(loss, dummy_x)
            except Exception as warm_err:
                pass
                
            print(f"Successfully loaded and warmed up {mode} pipeline.")
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

def compute_gradcam_1d(grad_model, sbm_tensor, mstm_tensor, class_index: int) -> List[float]:
    """
    Computes 1D Grad-CAM over the conv1d_sbm layer in the Fusion model.
    Optimized: Reuses pre-compiled grad_model and uses vectorized channel broadcasting.
    """
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([sbm_tensor, mstm_tensor], training=False)
        loss = predictions[:, class_index]
        
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0) # Shape: (channels,)
    
    # Vectorized channel-wise weighting (avoids Python for-loop)
    conv_outputs_np = conv_outputs[0].numpy() * pooled_grads.numpy()
    
    heatmap = np.mean(conv_outputs_np, axis=-1)
    heatmap = np.maximum(heatmap, 0.0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap.tolist()

def compute_feature_gradients(sbm_feat_extractor, mstm_feat_extractor, fusion_model, x_tensor, class_index: int) -> List[float]:
    """
    Computes direct input gradients of the target class prediction w.r.t the scaled input features.
    Optimized: Reuses pre-compiled feature extractors without rebuilding Keras Model graphs on each call.
    """
    with tf.GradientTape() as tape:
        tape.watch(x_tensor)
        sbm_feats = sbm_feat_extractor(x_tensor, training=False)
        mstm_feats = mstm_feat_extractor(x_tensor, training=False)
        probs = fusion_model([sbm_feats, mstm_feats], training=False)
        loss = probs[:, class_index]
        
    grads = tape.gradient(loss, x_tensor)
    importance = tf.abs(grads)[0, :, 0].numpy()
    max_val = np.max(importance)
    if max_val > 0:
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
    grad_model = pipe["grad_model"]
    
    # Fast feature extraction in expected column order
    features_dict = req.features
    try:
        row = np.array([[features_dict[c] for c in feature_columns]], dtype=np.float32)
    except KeyError:
        missing = [c for c in feature_columns if c not in features_dict]
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
        
    scaled = scaler.transform(row)
    x_scaled = np.expand_dims(scaled, axis=-1)  # (1, F, 1)
    x_tensor = tf.convert_to_tensor(x_scaled, dtype=tf.float32)
    
    # Direct fast tensor inference (avoids slow model.predict() batching overhead)
    sbm_features = sbm_feat_extractor(x_tensor, training=False)
    mstm_features = mstm_feat_extractor(x_tensor, training=False)
    probs_tensor = fusion_model([sbm_features, mstm_features], training=False)[0]
    probs = probs_tensor.numpy()
    
    classes = label_encoder.classes_.tolist()
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    class_idx = int(np.argmax(probs))
    predicted_class = classes[class_idx]
    
    # Fast explainability weights (reusing pre-compiled models and vectorized operations)
    gradcam_weights = compute_gradcam_1d(grad_model, sbm_features, mstm_features, class_idx)
    feature_importance = compute_feature_gradients(sbm_feat_extractor, mstm_feat_extractor, fusion_model, x_tensor, class_idx)
    
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
