"""
FastAPI backend for the seizure prediction real-world showcase.
Serves both As Trained and Leakage-corrected model branches and fusion models.
Memory-Optimized for Cloud Deployment (Render 512MB RAM free tier).
"""
import json
import os
import pickle
import gc
from pathlib import Path
from typing import Dict, List, Optional
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
import tensorflow as tf
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]
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

@app.on_event("startup")
def on_startup():
    port = os.environ.get("PORT", "10000")
    print(f"INFO: Seizure Prediction API initialized successfully, listening on port {port}")

# ----------------------------------------------------------------------
# Lazy-Loaded Model Pipelines (Keeps Startup RAM < 60MB on 512MB Render tier)
# ----------------------------------------------------------------------
pipelines = {}
available_modes = ["as_trained", "leakage_corrected"]


def get_pipeline(mode: str) -> Dict:
    """
    Loads Keras model artifacts on-demand rather than at startup.
    Ensures container never exceeds Render's 512MB memory limit.
    """
    if mode not in available_modes:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Available: {available_modes}"
        )

    if mode in pipelines:
        return pipelines[mode]

    # Memory guard: If switching modes, clear prior TensorFlow session to prevent RAM bloat
    if len(pipelines) > 0:
        pipelines.clear()
        tf.keras.backend.clear_session()
        gc.collect()

    mode_dir = MODELS_DIR / mode
    if not mode_dir.exists():
        raise HTTPException(status_code=404, detail=f"Model directory for '{mode}' not found.")

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
        return pipelines[mode]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline '{mode}': {str(e)}")


# Load sample records
SAMPLE_RECORDS = []
samples_path = MODELS_DIR / "sample_records.json"
if samples_path.exists():
    try:
        with open(samples_path) as f:
            SAMPLE_RECORDS = json.load(f)
    except Exception:
        SAMPLE_RECORDS = []


class PredictRequest(BaseModel):
    mode: str = "as_trained"
    features: Dict[str, float]


def compute_gradcam_1d(grad_model, sbm_tensor, mstm_tensor, class_index: int) -> List[float]:
    """Computes 1D Grad-CAM over the conv1d_sbm layer in the Fusion model."""
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([sbm_tensor, mstm_tensor], training=False)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=0)

    conv_outputs_np = conv_outputs[0].numpy() * pooled_grads.numpy()
    heatmap = np.mean(conv_outputs_np, axis=-1)
    heatmap = np.maximum(heatmap, 0.0)
    max_val = np.max(heatmap)
    if max_val > 0:
        heatmap /= max_val
    return heatmap.tolist()


def compute_feature_gradients(sbm_feat_extractor, mstm_feat_extractor, fusion_model, x_tensor, class_index: int) -> List[float]:
    """Computes direct input gradients of target class w.r.t scaled input features."""
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
        "available_modes": available_modes
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
    if mode not in available_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Available: {available_modes}")

    # Read metrics directly from disk (takes 0MB RAM, instant response)
    metrics_path = MODELS_DIR / mode / "metrics.json"
    metrics_data = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)

    return {
        "architecture": {
            "sbam": "Synchronization-Based Attention: MultiHeadAttention across feature/channel axis + Conv1D stack",
            "mstm": "Multi-Scale Temporal Modeling: parallel dilated Conv1D at 3 receptive-field scales",
            "fusion": "Concatenates SBAM + MSTM penultimate features, temporal + spatial attention, Dense classification head",
        },
        **metrics_data
    }


@app.post("/predict")
def predict(req: PredictRequest):
    mode = req.mode
    pipe = get_pipeline(mode)

    scaler = pipe["scaler"]
    label_encoder = pipe["label_encoder"]
    feature_columns = pipe["feature_columns"]
    sbm_feat_extractor = pipe["sbm_feat_extractor"]
    mstm_feat_extractor = pipe["mstm_feat_extractor"]
    fusion_model = pipe["fusion_model"]
    grad_model = pipe["grad_model"]

    features_dict = req.features
    try:
        row = np.array([[features_dict[c] for c in feature_columns]], dtype=np.float32)
    except KeyError:
        missing = [c for c in feature_columns if c not in features_dict]
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    scaled = scaler.transform(row)
    x_scaled = np.expand_dims(scaled, axis=-1)  # (1, F, 1)
    x_tensor = tf.convert_to_tensor(x_scaled, dtype=tf.float32)

    # Direct fast tensor inference
    sbm_features = sbm_feat_extractor(x_tensor, training=False)
    mstm_features = mstm_feat_extractor(x_tensor, training=False)
    probs_tensor = fusion_model([sbm_features, mstm_features], training=False)[0]
    probs = probs_tensor.numpy()

    classes = label_encoder.classes_.tolist()
    prob_dict = {cls: float(p) for cls, p in zip(classes, probs)}
    class_idx = int(np.argmax(probs))
    predicted_class = classes[class_idx]

    gradcam_weights = compute_gradcam_1d(grad_model, sbm_features, mstm_features, class_idx)
    feature_importance = compute_feature_gradients(sbm_feat_extractor, mstm_feat_extractor, fusion_model, x_tensor, class_idx)

    feature_importance_dict = {col: float(imp) for col, imp in zip(feature_columns, feature_importance)}

    return {
        "predicted_class": predicted_class,
        "probabilities": prob_dict,
        "gradcam_weights": gradcam_weights,
        "feature_importance": feature_importance_dict
    }


# Mount static frontend if co-located
if frontend_dir:
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
