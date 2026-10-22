# Architectural & Engineering Decisions Log

This log records design decisions, technical tradeoffs, and rationale adopted during the autonomous upgrade of the Seizure Prediction system.

---

### Decision 1: Execution on Dedicated Git Branch
- **Date:** 2026-10-22
- **Choice:** Created branch `honest-seizure-prediction`.
- **Rationale:** Preserves original repository state while cleanly organizing all Tier 1, 2, and 3 changes with atomic task-level commits.

### Decision 2: Environment & Package Management
- **Date:** 2026-10-22
- **Choice:** Python 3.14 venv with `scikit-learn`, `scipy`, `pandas`, `numpy`, `fastapi`, `uvicorn`, `requests`, `pyyaml`. PyTorch 2.14+ and MNE for continuous EEG models.
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
