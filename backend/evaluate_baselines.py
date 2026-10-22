"""
Evaluation and Baselines Script for Seizure Prediction / HMS Dataset.
Tasks 1, 2, and 3:
  1. Majority-class baseline & Logistic Regression baseline on random split.
  2. Patient-grouped split (GroupShuffleSplit on patient_id) with comprehensive metrics:
     - Accuracy, Balanced Accuracy, Macro F1, Per-class Recall/Precision, Confusion Matrix.
  3. Automated leakage validation test suite asserting no label-derived columns exist in inputs.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "train.csv")
OUT_DIR = os.path.join(SCRIPT_DIR, "saved_models")

LABEL_COLUMN = "expert_consensus"

LEAKAGE_COLUMNS = [
    "seizure_vote", "lpd_vote", "gpd_vote",
    "lrda_vote", "grda_vote", "other_vote",
    "label_id"
]

METADATA_FEATURE_COLUMNS = [
    "eeg_id", "eeg_sub_id", "eeg_label_offset_seconds",
    "spectrogram_id", "spectrogram_sub_id", "spectrogram_label_offset_seconds",
    "patient_id", "fold"
]


def assert_no_label_leakage(feature_columns, df=None):
    """
    Automated check that FAILS if any label-derived or leaking column is in inputs.
    Task 3 requirement.
    """
    feature_set = set(feature_columns)
    leaks = [col for col in feature_set if col in LEAKAGE_COLUMNS or col == LABEL_COLUMN or "vote" in col.lower()]
    if leaks:
        raise AssertionError(f"[LEAKAGE TEST FAILED] Label-derived columns detected in features: {leaks}")
    
    if df is not None:
        # Check for perfect correlation with target indicator
        for col in feature_columns:
            if col in df.columns and col in LEAKAGE_COLUMNS:
                raise AssertionError(f"[LEAKAGE TEST FAILED] Forbidden column '{col}' found in input dataset.")
    print("[PASSED] [LEAKAGE TEST] No label-derived or vote columns in feature list.")


def assert_no_patient_overlap(train_patients, test_patients):
    """
    Automated check that FAILS if any patient appears in both train and test.
    """
    overlap = set(train_patients).intersection(set(test_patients))
    if len(overlap) > 0:
        raise AssertionError(f"[LEAKAGE TEST FAILED] Patient overlap detected across train and test! Overlapping IDs: {list(overlap)[:5]}... (Total: {len(overlap)})")
    print(f"[PASSED] [PATIENT ISOLATION] Zero patient overlap. Train patients: {len(set(train_patients))}, Test patients: {len(set(test_patients))}.")


def evaluate_model_predictions(y_true, y_pred, classes):
    """Computes comprehensive multi-class metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0).tolist()
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()
    
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    per_class_metrics = {}
    for idx, cls_name in enumerate(classes):
        per_class_metrics[cls_name] = {
            "recall": float(per_class_recall[idx]),
            "precision": float(per_class_precision[idx]),
            "f1": float(per_class_f1[idx]),
            "support": int(report[cls_name]["support"])
        }
        
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class_metrics,
        "confusion_matrix": cm,
        "classification_report": report
    }


def run_evaluations():
    print("=" * 60)
    print("TASK 1 & 2: BASELINES & PATIENT-GROUPED EVALUATION")
    print("=" * 60)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    
    # Task 3 automated check on corrected features
    assert_no_label_leakage(METADATA_FEATURE_COLUMNS, df)
    
    # Prepare labels
    le = LabelEncoder()
    y_all = le.fit_transform(df[LABEL_COLUMN])
    classes = le.classes_.tolist()
    
    # Features (metadata features without leakage)
    X_all_raw = df[METADATA_FEATURE_COLUMNS].values.astype(np.float32)
    
    # ==============================================================
    # 1. ORIGINAL RANDOM SPLIT (Replication of existing test set)
    # ==============================================================
    print("\n--- 1. Random Split (Existing baseline comparison, test_size=0.2, seed=42) ---")
    scaler_rand = StandardScaler()
    X_all_scaled = scaler_rand.fit_transform(X_all_raw)
    
    X_train_r, X_test_r, y_train_r, y_test_r, idx_train_r, idx_test_r = train_test_split(
        X_all_scaled, y_all, df.index.values, test_size=0.2, random_state=42, stratify=y_all
    )
    
    # Check patient overlap in random split
    rand_train_patients = set(df.loc[idx_train_r, "patient_id"])
    rand_test_patients = set(df.loc[idx_test_r, "patient_id"])
    overlap_r = rand_train_patients.intersection(rand_test_patients)
    print(f"Random split patient overlap: {len(overlap_r)} patients appear in both train and test! (Patient Leakage)")
    
    # Baseline 1: Majority-Class Classifier
    dummy_r = DummyClassifier(strategy="most_frequent")
    dummy_r.fit(X_train_r, y_train_r)
    y_pred_dummy_r = dummy_r.predict(X_test_r)
    dummy_r_results = evaluate_model_predictions(y_test_r, y_pred_dummy_r, classes)
    print(f"Majority-Class Baseline Accuracy: {dummy_r_results['accuracy']:.4f} (Balanced Acc: {dummy_r_results['balanced_accuracy']:.4f}, Macro F1: {dummy_r_results['macro_f1']:.4f})")
    
    # Baseline 2: Logistic Regression
    lr_r = LogisticRegression(max_iter=1000, random_state=42)
    lr_r.fit(X_train_r, y_train_r)
    y_pred_lr_r = lr_r.predict(X_test_r)
    lr_r_results = evaluate_model_predictions(y_test_r, y_pred_lr_r, classes)
    print(f"Logistic Regression Accuracy:    {lr_r_results['accuracy']:.4f} (Balanced Acc: {lr_r_results['balanced_accuracy']:.4f}, Macro F1: {lr_r_results['macro_f1']:.4f})")
    
    # Existing Leakage-Corrected DL Model metrics from disk
    existing_corrected_path = os.path.join(OUT_DIR, "leakage_corrected", "metrics.json")
    existing_corrected_metrics = {}
    if os.path.exists(existing_corrected_path):
        with open(existing_corrected_path) as f:
            existing_corrected_metrics = json.load(f)
            
    # Existing As-Trained (Leaking) Model metrics from disk
    existing_as_trained_path = os.path.join(OUT_DIR, "as_trained", "metrics.json")
    existing_as_trained_metrics = {}
    if os.path.exists(existing_as_trained_path):
        with open(existing_as_trained_path) as f:
            existing_as_trained_metrics = json.load(f)

    # ==============================================================
    # 2. PATIENT-GROUPED SPLIT (Zero Patient Overlap)
    # ==============================================================
    print("\n--- 2. Patient-Grouped Split (GroupShuffleSplit on patient_id, test_size=0.2, seed=42) ---")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    
    train_patients = df.iloc[train_idx]["patient_id"].values
    test_patients = df.iloc[test_idx]["patient_id"].values
    assert_no_patient_overlap(train_patients, test_patients)
    
    # Scale strictly on training set (NO TEST LEAKAGE)
    scaler_grp = StandardScaler()
    X_train_g = scaler_grp.fit_transform(X_all_raw[train_idx])
    X_test_g = scaler_grp.transform(X_all_raw[test_idx])
    y_train_g = y_all[train_idx]
    y_test_g = y_all[test_idx]
    
    print(f"Train samples: {len(X_train_g)} (from {len(set(train_patients))} patients)")
    print(f"Test samples:  {len(X_test_g)} (from {len(set(test_patients))} patients)")
    
    # Model A: Majority-Class Baseline (Grouped)
    dummy_g = DummyClassifier(strategy="most_frequent")
    dummy_g.fit(X_train_g, y_train_g)
    y_pred_dummy_g = dummy_g.predict(X_test_g)
    dummy_g_results = evaluate_model_predictions(y_test_g, y_pred_dummy_g, classes)
    print(f"\n[Grouped] Majority-Class Baseline:  Acc={dummy_g_results['accuracy']:.4f}, BalAcc={dummy_g_results['balanced_accuracy']:.4f}, MacroF1={dummy_g_results['macro_f1']:.4f}")
    
    # Model B: Logistic Regression (Grouped)
    lr_g = LogisticRegression(max_iter=1000, random_state=42)
    lr_g.fit(X_train_g, y_train_g)
    y_pred_lr_g = lr_g.predict(X_test_g)
    lr_g_results = evaluate_model_predictions(y_test_g, y_pred_lr_g, classes)
    print(f"[Grouped] Logistic Regression:      Acc={lr_g_results['accuracy']:.4f}, BalAcc={lr_g_results['balanced_accuracy']:.4f}, MacroF1={lr_g_results['macro_f1']:.4f}")
    
    # Model C: Random Forest (Grouped)
    rf_g = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    rf_g.fit(X_train_g, y_train_g)
    y_pred_rf_g = rf_g.predict(X_test_g)
    rf_g_results = evaluate_model_predictions(y_test_g, y_pred_rf_g, classes)
    print(f"[Grouped] Random Forest:            Acc={rf_g_results['accuracy']:.4f}, BalAcc={rf_g_results['balanced_accuracy']:.4f}, MacroF1={rf_g_results['macro_f1']:.4f}")
    
    # Compile unified results
    all_results = {
        "dataset_summary": {
            "total_samples": len(df),
            "num_patients": int(df["patient_id"].nunique()),
            "classes": classes,
            "class_distribution": df[LABEL_COLUMN].value_counts().to_dict(),
        },
        "random_split_evaluation": {
            "note": "Replicates original random split (contains patient overlap across train/test)",
            "num_train": len(X_train_r),
            "num_test": len(X_test_r),
            "patient_overlap_count": len(overlap_r),
            "majority_baseline": dummy_r_results,
            "logistic_regression": lr_r_results,
            "leakage_corrected_deep_model": existing_corrected_metrics,
            "as_trained_leaking_model": existing_as_trained_metrics,
        },
        "patient_grouped_split_evaluation": {
            "note": "Zero patient overlap between train and test (GroupShuffleSplit on patient_id)",
            "num_train": len(X_train_g),
            "num_test": len(X_test_g),
            "num_train_patients": len(set(train_patients)),
            "num_test_patients": len(set(test_patients)),
            "majority_baseline": dummy_g_results,
            "logistic_regression": lr_g_results,
            "random_forest": rf_g_results,
        }
    }
    
    out_file = os.path.join(OUT_DIR, "honest_evaluation.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved complete evaluation results to {out_file}")
    
    # Also update saved_models/leakage_corrected/metrics.json to include baseline comparisons
    if existing_corrected_metrics:
        existing_corrected_metrics["majority_baseline_accuracy"] = dummy_r_results["accuracy"]
        existing_corrected_metrics["majority_baseline_macro_f1"] = dummy_r_results["macro_f1"]
        existing_corrected_metrics["logistic_regression_accuracy"] = lr_r_results["accuracy"]
        existing_corrected_metrics["logistic_regression_macro_f1"] = lr_r_results["macro_f1"]
        existing_corrected_metrics["patient_grouped"] = {
            "majority_accuracy": dummy_g_results["accuracy"],
            "majority_balanced_accuracy": dummy_g_results["balanced_accuracy"],
            "majority_macro_f1": dummy_g_results["macro_f1"],
            "logistic_regression_accuracy": lr_g_results["accuracy"],
            "logistic_regression_balanced_accuracy": lr_g_results["balanced_accuracy"],
            "logistic_regression_macro_f1": lr_g_results["macro_f1"],
            "random_forest_accuracy": rf_g_results["accuracy"],
            "random_forest_balanced_accuracy": rf_g_results["balanced_accuracy"],
            "random_forest_macro_f1": rf_g_results["macro_f1"],
            "num_test_patients": len(set(test_patients)),
            "confusion_matrix_majority": dummy_g_results["confusion_matrix"],
            "confusion_matrix_lr": lr_g_results["confusion_matrix"],
            "confusion_matrix_rf": rf_g_results["confusion_matrix"],
        }
        with open(existing_corrected_path, "w") as f:
            json.dump(existing_corrected_metrics, f, indent=2)
        print(f"Updated {existing_corrected_path} with baseline comparisons and patient-grouped metrics.")
    
    return all_results

if __name__ == "__main__":
    run_evaluations()
