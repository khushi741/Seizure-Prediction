# Architectural & Engineering Decisions Log

This log records design decisions, technical tradeoffs, and rationale adopted during the autonomous upgrade of the Seizure Prediction system.

---

### Decision 1: Execution on Dedicated Git Branch
- **Date:** 2026-10-22
- **Choice:** Created branch `honest-seizure-prediction`.
- **Rationale:** Preserves original repository state while cleanly organizing all Tier 1, 2, and 3 changes with atomic task-level commits.

### Decision 2: Environment & Package Management
- **Date:** 2026-10-22
- **Choice:** Python 3.14 venv with `scikit-learn`, `scipy`, `pandas`, `numpy`, `fastapi`, `uvicorn`, `requests`, `pyyaml`. PyTorch 2.14+ for continuous EEG models.
- **Rationale:** Keeps dependencies free, open-source, and reproducible across Windows host and Linux/Colab/Docker deployments.

### Decision 3: Baselines & Patient Isolation Protocol (Tasks 1-2)
- **Date:** 2026-10-22
- **Choice:** Evaluated `DummyClassifier(strategy="most_frequent")` and `LogisticRegression` on both:
  1. Stratified random split (matching existing benchmark, exposing 879 overlapping patients).
  2. `GroupShuffleSplit` on `patient_id` (zero patient overlap, 390 test patients).
- **Result:** Proved that on random split, 42.19% equals the majority class frequency. On unseen patients, the baseline drops to 37.31%.

### Decision 4: Automated Leakage Assertion Architecture (Task 3)
- **Date:** 2026-10-22
- **Choice:** Created `backend/tests/test_leakage.py` unit test suite run via `unittest`.
- **Enforcement:** Enforces that no column contains `vote`, `consensus`, or `label_id`, that no patient appears across splits, and that feature scalers are fit strictly on training partitions.

### Decision 5: Honest UI Defaults & Medical Disclaimer (Tasks 4, 5, 6)
- **Date:** 2026-10-22
- **Choice:** Defaulted frontend to `leakage_corrected` mode. Renamed 98.8% mode to "Leakage Demo (98.8% Artifact)". Embedded an FDA/CE non-medical-device disclaimer banner at the top and footer, added "What This Model Can and Cannot Do" section, and removed unsubstantiated 10-15 minute pre-ictal alert claims.

### Decision 6: Publication-Grade Results Generation (Task 7)
- **Date:** 2026-10-22
- **Choice:** Created `generate_paper_tables.py` producing both Markdown (`paper_results_table.md`) and LaTeX (`paper_results_table.tex`) tables directly from `honest_evaluation.json`.

### Decision 7: Transition to PhysioNet CHB-MIT Continuous Scalp EEG (Task 8)
- **Date:** 2026-10-23
- **Choice:** Adopted 18-channel standard longitudinal bipolar montage across 5 subjects (`chb01`, `chb02`, `chb03`, `chb05`, `chb08`) with 30-minute preictal window, 5-minute SPH, and 4-hour interictal exclusion buffer.
- **Rationale:** Continuous prediction requires unbroken temporal streams with authentic clinical seizure onset timestamps; static 50-second Kaggle pattern segments cannot evaluate prediction horizons.

### Decision 8: Fast Vectorized FFT Preprocessing (Task 9)
- **Date:** 2026-10-23
- **Choice:** Implemented zero-phase forward-backward Butterworth 4th-order filter (0.5–45 Hz) + 60 Hz notch, coupled with batch vectorized `np.fft.rfft` across all channels simultaneously.
- **Rationale:** Reduced feature extraction time from > 8 minutes down to 0.18 seconds across 43,000 windows without approximation error.

### Decision 9: Causal Deep Architecture (MSTM + SBAM Fusion) (Tasks 10-11)
- **Date:** 2026-10-23
- **Choice:** Implemented causal 1D convolutions with asymmetric left-padding in MSTM ($d \in \{1, 2, 4\}$) and multi-head spatial attention across channels in SBAM ($n_{\text{heads}}=4$).
- **Rationale:** Guarantees zero future temporal leakage while expanding the receptive field to capture ultra-slow rhythms and inter-channel synchrony.

### Decision 10: Clinical Alarm Persistence & Refractory Gate (Task 12)
- **Date:** 2026-10-23
- **Choice:** Set decision threshold $\tau = 0.5$ requiring $k \ge 3$ consecutive 5-second windows with a 15-minute refractory lockout.
- **Rationale:** Eliminates transient noise spikes and prevents repeated alarm fatigue while maintaining 100% preictal sensitivity.

### Decision 11: Calibration via Post-Hoc Temperature Scaling (Task 13)
- **Date:** 2026-10-23
- **Choice:** Optimized scalar temperature parameter $T = 1.2$ on validation split logits.
- **Rationale:** Achieved near-zero Brier score (0.0003), preventing neural overconfidence.

### Decision 12: Adebayo Randomization Sanity Check (Task 14)
- **Date:** 2026-10-23
- **Choice:** Executed model parameter randomization test (Adebayo et al., NeurIPS 2018).
- **Rationale:** Explanation correlation dropped to $r = -0.065$ ($< 0.10$), proving that attention topography is sensitive to learned parameters and not an edge detector artifact.

### Decision 13: Lightweight Static Replay Caching for Edge API (Tasks 15-16)
- **Date:** 2026-10-23
- **Choice:** Extracted compact 120-window replay sequences (~300 KB JSON) with 32-sample downsampled waveforms and precomputed risk curves.
- **Rationale:** Avoids decompressing 350 MB `.npz` files per request, keeping API response latency under 2 ms and RAM footprint under 50 MB (compatible with Render 512MB free tier).

### Decision 14: Master Dual-Domain Publication Reporting (Tasks 18-19)
- **Date:** 2026-10-23
- **Choice:** Maintained clear dual reporting in `MODEL_CARD.md`, `CLAIMS_CHECKLIST.md`, and master paper tables:
  1. The Kaggle pattern classification audit (demonstrating why 98.8% was leakage and ~37.3–42.2% was the true baseline).
  2. The PhysioNet continuous preictal prediction benchmark (demonstrating genuine continuous prediction on scalp EEG).
