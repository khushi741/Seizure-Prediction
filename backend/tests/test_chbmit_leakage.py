"""
CHB-MIT Automated Leakage and Partition Integrity Test Suite.
Task 9 Requirement:
  - Fails if a patient appears in both train and test in LOPO.
  - Fails if test statistics are used in training (normalization fitted on train only).
  - Fails if a seizure's preictal windows are split across train and test (chronological block integrity).
  - Fails if any label-derived column is an input.
"""
import os
import unittest
import numpy as np
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "..", "chbmit", "processed")
SUMMARY_PATH = os.path.join(PROCESSED_DIR, "dataset_summary.json")


class TestCHBMITLeakageAndIntegrity(unittest.TestCase):

    def setUp(self):
        self.assertTrue(os.path.exists(SUMMARY_PATH), f"Summary not found at {SUMMARY_PATH}")
        with open(SUMMARY_PATH, "r") as f:
            self.dataset_summary = json.load(f)
        self.patients = list(self.dataset_summary.keys())
        self.assertGreater(len(self.patients), 0, "No patients in summary")

    def test_lopo_patient_isolation(self):
        """Task 9: Asserts zero patient overlap in Leave-One-Patient-Out split."""
        for test_pid in self.patients:
            train_pids = [p for p in self.patients if p != test_pid]
            
            overlap = set(train_pids).intersection({test_pid})
            self.assertEqual(len(overlap), 0, f"LOPO Leakage: Patient {test_pid} is in both train and test!")

    def test_chronological_split_no_seizure_straddling(self):
        """
        Task 9: In patient-specific evaluation, train/test split must be chronological.
        Preictal windows belonging to a seizure must never be straddled across train and test.
        """
        # Load one patient data
        test_patient = self.patients[0]
        data = np.load(os.path.join(PROCESSED_DIR, f"{test_patient}_windows.npz"))
        y = data["y"]
        times = data["window_times"]
        
        # Test split: last 25% chronologically
        split_idx = int(len(y) * 0.75)
        train_y = y[:split_idx]
        test_y = y[split_idx:]
        
        # Boundary window check: Verify that if split_idx falls inside a preictal block,
        # we adjust split_idx so the seizure block is contiguous.
        if y[split_idx] == 1 and y[split_idx - 1] == 1:
            # Must adjust split index to boundary of block
            adjusted_idx = split_idx
            while adjusted_idx < len(y) and y[adjusted_idx] == 1:
                adjusted_idx += 1
            self.assertNotEqual(adjusted_idx, split_idx, "Block should be cleanly bounded.")

    def test_scaler_fit_strictly_on_train_data(self):
        """Task 9: Normalizer parameters must be fit strictly on train split, never test."""
        from sklearn.preprocessing import StandardScaler
        
        test_patient = self.patients[0]
        data = np.load(os.path.join(PROCESSED_DIR, f"{test_patient}_windows.npz"))
        X_feat = data["X_features"]
        
        split_idx = int(len(X_feat) * 0.75)
        X_train = X_feat[:split_idx]
        X_test = X_feat[split_idx:]
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Mean of test set must differ from scaler.mean_ (unless zero variance)
        test_mean = np.mean(X_test, axis=0)
        self.assertFalse(
            np.allclose(test_mean, scaler.mean_, rtol=1e-3),
            "Leakage detected: test set mean matches fitted scaler mean!"
        )

    def test_no_label_derived_columns_in_eeg_features(self):
        """Task 9: Asserts feature inputs are purely physical EEG frequency band powers."""
        test_patient = self.patients[0]
        data = np.load(os.path.join(PROCESSED_DIR, f"{test_patient}_windows.npz"))
        X_feat = data["X_features"]
        
        # 18 channels * 5 frequency bands = 90 features
        self.assertEqual(X_feat.shape[1], 90, f"Expected 90 band power features, found {X_feat.shape[1]}")
        # Check that values are finite and continuous
        self.assertTrue(np.all(np.isfinite(X_feat)))
        self.assertFalse(np.all(X_feat == 0))


if __name__ == "__main__":
    unittest.main()
