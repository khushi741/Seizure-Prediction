"""
Trains and saves the Synchronization-Based Attention Mechanism (SBAM) +
Multi-Scale Temporal Model (MSTM) fusion pipeline in two modes:
  1. As Trained: Faithful replication utilizing all 14 columns (including leaking vote columns).
  2. Leakage-corrected: Train the same architecture with the vote_* columns removed.

Saves artifacts for both modes under saved_models/as_trained and saved_models/leakage_corrected:
  - fusion_model.keras   (final classifier)
  - sync_model.keras     (SBAM branch)
  - tcn_mstm_model.keras (MSTM branch)
  - scaler.pkl           (StandardScaler fit on training features)
  - label_encoder.pkl    (class name <-> index mapping)
  - feature_columns.json (ordered list of input feature names used)
  - metrics.json         (accuracy, confusion matrix, precision, recall, F1)
"""
import json
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "train.csv")
BASE_OUT_DIR = os.path.join(SCRIPT_DIR, "saved_models")

FEATURE_COLUMNS_AS_TRAINED = [
    "eeg_id", "eeg_sub_id", "eeg_label_offset_seconds",
    "spectrogram_id", "spectrogram_sub_id", "spectrogram_label_offset_seconds",
    "patient_id", "seizure_vote", "lpd_vote", "gpd_vote",
    "lrda_vote", "grda_vote", "other_vote", "fold",
]

FEATURE_COLUMNS_CORRECTED = [
    "eeg_id", "eeg_sub_id", "eeg_label_offset_seconds",
    "spectrogram_id", "spectrogram_sub_id", "spectrogram_label_offset_seconds",
    "patient_id", "fold",
]

LABEL_COLUMN = "expert_consensus"

def create_tcn_mstm_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="MSTM_Input")
    conv1 = layers.Conv1D(64, kernel_size=3, dilation_rate=1, activation="relu", padding="causal")(inputs)
    conv2 = layers.Conv1D(64, kernel_size=5, dilation_rate=2, activation="relu", padding="causal")(inputs)
    conv3 = layers.Conv1D(64, kernel_size=7, dilation_rate=4, activation="relu", padding="causal")(inputs)
    concatenated = layers.concatenate([conv1, conv2, conv3], name="mstm_multiscale_concat")
    pooled = layers.GlobalAveragePooling1D()(concatenated)
    dense = layers.Dense(128, activation="relu", name="mstm_dense")(pooled)
    dropout = layers.Dropout(0.5)(dense)
    outputs = layers.Dense(num_classes, activation="softmax")(dropout)
    model = models.Model(inputs=inputs, outputs=outputs, name="MSTM_Model")
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def create_sync_spatial_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape, name="SBAM_Input")  # (features, 1)
    x = layers.Permute((2, 1))(inputs)
    x = layers.Dense(64)(x)
    attn_out = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="SBAM_MultiHeadAttention")(x, x)
    x = layers.Add()([x, attn_out])
    x = layers.LayerNormalization()(x)
    x = layers.Permute((2, 1))(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same", name="conv1d_sbm")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu", name="sbm_dense")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="SBAM_Model")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def build_fusion_model(input_shape_sbm, input_shape_mstm, num_classes):
    input_sbm = layers.Input(shape=input_shape_sbm, name="SBM_Feature_Input")
    input_mstm = layers.Input(shape=input_shape_mstm, name="MSTM_Feature_Input")

    # SBAM Branch inside Fusion
    x1 = layers.Reshape((input_shape_sbm[0], 1))(input_sbm)
    x1 = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same", name="conv1d_sbm")(x1)
    x1 = layers.BatchNormalization()(x1)
    x1_temporal = layers.Attention(name="Fusion_Temporal_Attn")([x1, x1])
    x1_spatial = layers.Permute((2, 1))(x1_temporal)
    x1_spatial = layers.MultiHeadAttention(num_heads=4, key_dim=32, name="Fusion_Spatial_MultiHead_Attn")(x1_spatial, x1_spatial)
    x1_spatial = layers.Permute((2, 1))(x1_spatial)
    x1 = layers.Dropout(0.3)(x1_spatial)
    x1 = layers.Flatten()(x1)

    # MSTM Branch inside Fusion
    x2 = layers.Reshape((input_shape_mstm[0], 1))(input_mstm)
    b1 = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu", name="mstcn_branch_3")(x2)
    b2 = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu", name="mstcn_branch_5")(x2)
    b3 = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu", name="mstcn_branch_7")(x2)
    x2 = layers.Concatenate(name="mstcn_concat")([b1, b2, b3])
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.3)(x2)
    x2 = layers.Flatten()(x2)

    merged = layers.Concatenate(name="Fusion_Layer")([x1, x2])
    merged = layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001), name="Dense_Fusion")(merged)
    merged = layers.Dropout(0.3)(merged)
    output = layers.Dense(num_classes, activation="softmax", name="Classification_Output")(merged)

    model = models.Model(inputs=[input_sbm, input_mstm], outputs=output, name="Fusion_Model")
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_pipeline(mode, feature_columns):
    print(f"\n==================================================")
    print(f"TRAINING PIPELINE: {mode.upper()}")
    print(f"==================================================")
    
    out_dir = f"{BASE_OUT_DIR}/{mode}"
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(DATA_PATH)
    
    X_raw = df[feature_columns].values.astype("float32")
    y_raw = df[LABEL_COLUMN].values
    
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    y_cat = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_cat, test_size=0.2, random_state=42, stratify=y_int
    )
    
    X_train = np.expand_dims(X_train, axis=-1)  # (N, F, 1)
    X_test = np.expand_dims(X_test, axis=-1)
    input_shape = X_train.shape[1:]
    
    print(f"Train features: {X_train.shape}, Test features: {X_test.shape}")
    
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    
    # 1. Train MSTM Model
    print("Training MSTM branch...")
    tcn_mstm_model = create_tcn_mstm_model(input_shape, num_classes)
    tcn_mstm_model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=8, batch_size=32, callbacks=[early_stop], verbose=1
    )
    y_pred_mstm = tcn_mstm_model.predict(X_test).argmax(axis=1)
    mstm_acc = float(accuracy_score(y_test.argmax(axis=1), y_pred_mstm))
    print(f"MSTM Branch Test Accuracy: {mstm_acc:.4f}")
    
    # 2. Train SBAM Model
    print("Training SBAM branch...")
    sync_model = create_sync_spatial_model(input_shape, num_classes)
    sync_model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        epochs=8, batch_size=32, callbacks=[early_stop], verbose=1
    )
    y_pred_sync = sync_model.predict(X_test).argmax(axis=1)
    sbam_acc = float(accuracy_score(y_test.argmax(axis=1), y_pred_sync))
    print(f"SBAM Branch Test Accuracy: {sbam_acc:.4f}")
    
    # 3. Train Fusion Model
    print("Extracting features for Fusion training...")
    sbm_feature_extractor = models.Model(inputs=sync_model.input, outputs=sync_model.get_layer("sbm_dense").output)
    mstm_feature_extractor = models.Model(inputs=tcn_mstm_model.input, outputs=tcn_mstm_model.get_layer("mstm_dense").output)
    
    X_train_sbm = sbm_feature_extractor.predict(X_train)
    X_test_sbm = sbm_feature_extractor.predict(X_test)
    X_train_mstm = mstm_feature_extractor.predict(X_train)
    X_test_mstm = mstm_feature_extractor.predict(X_test)
    
    fusion_model = build_fusion_model(X_train_sbm.shape[1:], X_train_mstm.shape[1:], num_classes)
    reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, verbose=1)
    
    print("Training Fusion model...")
    fusion_model.fit(
        [X_train_sbm, X_train_mstm], y_train,
        validation_data=([X_test_sbm, X_test_mstm], y_test),
        epochs=12, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=1
    )
    
    y_pred_fusion_raw = fusion_model.predict([X_test_sbm, X_test_mstm])
    y_pred_fusion = y_pred_fusion_raw.argmax(axis=1)
    y_true = y_test.argmax(axis=1)
    fusion_acc = float(accuracy_score(y_true, y_pred_fusion))
    print(f"Fusion Model Test Accuracy: {fusion_acc:.4f}")
    
    # Calculate detailed metrics
    classes = label_encoder.classes_.tolist()
    report = classification_report(y_true, y_pred_fusion, target_names=classes, output_dict=True)
    conf_mat = confusion_matrix(y_true, y_pred_fusion).tolist()
    
    metrics = {
        "mstm_accuracy": mstm_acc,
        "sbam_accuracy": sbam_acc,
        "fusion_accuracy": fusion_acc,
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": conf_mat,
        "num_train": int(len(X_train)),
        "num_test": int(len(X_test)),
    }
    
    # Save artifacts
    sync_model.save(f"{out_dir}/sync_model.keras")
    tcn_mstm_model.save(f"{out_dir}/tcn_mstm_model.keras")
    fusion_model.save(f"{out_dir}/fusion_model.keras")
    
    with open(f"{out_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{out_dir}/label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(f"{out_dir}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(f"{out_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"Saved all artifacts to {out_dir}")
    return metrics

if __name__ == "__main__":
    # Train both modes
    as_trained_metrics = train_pipeline("as_trained", FEATURE_COLUMNS_AS_TRAINED)
    corrected_metrics = train_pipeline("leakage_corrected", FEATURE_COLUMNS_CORRECTED)
    
    # Save sample records for frontend (using raw columns)
    df = pd.read_csv(DATA_PATH)
    sample_idx = np.random.RandomState(42).choice(len(df), size=30, replace=False)
    # Save with all possible columns
    all_cols = list(set(FEATURE_COLUMNS_AS_TRAINED + [LABEL_COLUMN]))
    df.iloc[sample_idx][all_cols].to_json(f"{BASE_OUT_DIR}/sample_records.json", orient="records")
    print("\nSaved sample records for the frontend to saved_models/sample_records.json")
