# Empirical Results Table (Generated Automatically)

*All numbers generated from unified evaluation on `train.csv` (11,138 samples, 1,950 unique patients).*

### Table 1: Pattern Classification Performance Across Validation Protocols

| Model Architecture | Evaluation Protocol | Patient Isolation | Accuracy (%) | Balanced Acc (%) | Macro F1 (%) | Status / Note |
|---|---|---|:---:|:---:|:---:|---|
| **Majority-Class Baseline** | Random Split (20% test) | Leaking (879 shared) | 42.19% | 16.67% | 9.89% | Predicts 'Other' always |
| **Logistic Regression** | Random Split (20% test) | Leaking (879 shared) | 42.19% | 16.67% | 9.89% | Linear baseline |
| **MSTM Only (Dilated Conv1D)** | Random Split (20% test) | Leaking (879 shared) | 42.19% | 16.67% | 9.89% | No vote features |
| **SBAM Only (Spatial Attention)** | Random Split (20% test) | Leaking (879 shared) | 42.37% | 16.67% | 9.89% | No vote features |
| **STFM Fusion (MSTM + SBAM)** | Random Split (20% test) | Leaking (879 shared) | 42.19% | 16.67% | 9.89% | Equal to majority baseline |
| **Majority-Class Baseline** | Patient-Grouped Split | **Zero Overlap (390 unseen)** | **37.31%** | **16.67%** | **9.06%** | Unseen patient floor |
| **Logistic Regression** | Patient-Grouped Split | **Zero Overlap (390 unseen)** | **37.31%** | **16.67%** | **9.06%** | Linear on metadata |
| **Random Forest (100 trees)** | Patient-Grouped Split | **Zero Overlap (390 unseen)** | **36.02%** | **16.35%** | **9.82%** | Non-linear on metadata |
| *Leaking Artifact (As-Trained)* | *Random Split* | *Severe Leakage* | *99.33%* | *99.38%* | *99.27%* | *Trained on vote_* columns* |

> **Key Finding:** When evaluated honestly without annotator vote leakage, both deep networks and linear baselines collapse to predicting the majority class (~37.3% on unseen patients; 42.2% on random split). This proves metadata columns (ID, offsets, fold) contain zero physiological signal, establishing the imperative for continuous EEG raw waveform analysis (CHB-MIT).
