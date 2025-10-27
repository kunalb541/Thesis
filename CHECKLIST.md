# 📋 THESIS BENCHMARKING CHECKLIST

## Phase 0: Understanding (30 minutes)
- [ ] Read PARAMETERS_EXPLAINED.md (understand why u0 matters!)
- [ ] Read COMPLETE_SUMMARY.md (understand the benchmark)
- [ ] Review FILE_GUIDE.txt (understand the structure)

**Key takeaway**: Small u0 → crosses caustics → easy to detect! Large u0 → misses caustics → hard!

## Phase 1: Setup (Today)
- [ ] On laptop: Initialize git repository
- [ ] On laptop: Create GitHub repository
- [ ] On laptop: Push code to GitHub
- [ ] On cluster: Clone repository
- [ ] On cluster: Create conda environment
- [ ] On cluster: Install all dependencies
- [ ] On cluster: Make scripts executable (`chmod +x slurm/*.sh`)

**Verification**: Run `python -c "import tensorflow as tf; print(tf.__version__)"`

## Phase 2: Quick Test (1 hour)
- [ ] Request interactive GPU session
- [ ] Activate conda environment
- [ ] Run `python code/test_quick.py`
- [ ] Verify all 4 tests pass:
  - [ ] Data loading ✓
  - [ ] GPU setup ✓
  - [ ] Model building ✓
  - [ ] Small training run ✓

**If all tests pass**: You're ready for full training!

## Phase 3: Baseline Experiment (1-2 days)

### Generate Data (if needed)
- [ ] Check if events_1M.npz exists
- [ ] If not, generate: `python code/simulate.py --n_pspl 500000 --n_binary 500000 --output data/raw/events_1M.npz`

### Train Baseline Model
- [ ] Submit job: `sbatch slurm/slurm_train_baseline.sh`
- [ ] Monitor: `squeue -u hd_vm305`
- [ ] Watch logs: `tail -f logs/train_baseline_*.out`
- [ ] Wait 6-12 hours for completion

### Evaluate Baseline
- [ ] Run evaluation: `python code/evaluate.py --model models/baseline.keras --data data/raw/events_1M.npz --output_dir results/baseline`
- [ ] Check results:
  - [ ] metrics.json (ROC AUC should be ~0.98)
  - [ ] confusion_matrix.png (check classification errors)
  - [ ] roc_curve.png (should look good)
  - [ ] early_detection.png (shows feasibility)

**Expected**: 95-96% accuracy, ROC AUC ~0.98

## Phase 4: Cadence Experiments (3-5 days)

### Dense Cadence (5% missing)
- [ ] Generate data: `python code/simulate.py --n_pspl 100000 --n_binary 100000 --cadence 0.05 --output data/raw/events_cadence_05.npz`
- [ ] Train: `python code/train.py --data data/raw/events_cadence_05.npz --output models/cadence_05.keras --experiment_name cadence_05`
- [ ] Evaluate: `python code/evaluate.py --model models/cadence_05.keras --data data/raw/events_cadence_05.npz --output_dir results/cadence_05`
- [ ] Record ROC AUC: _______

### Sparse Cadence (30% missing)
- [ ] Generate data: `--cadence 0.30`
- [ ] Train model
- [ ] Evaluate
- [ ] Record ROC AUC: _______

### Compare Results
- [ ] Plot: Accuracy vs % coverage
- [ ] Thesis insight: "Dense sampling improves to X%, sparse degrades to Y%"

## Phase 5: Photometric Error Experiments (3-5 days)

### Low Error (0.05 mag)
- [ ] Generate data: `python code/simulate.py --n_pspl 100000 --n_binary 100000 --error 0.05 --output data/raw/events_error_low.npz`
- [ ] Train model
- [ ] Evaluate
- [ ] Record ROC AUC: _______

### High Error (0.20 mag)
- [ ] Generate data: `--error 0.20`
- [ ] Train model
- [ ] Evaluate
- [ ] Record ROC AUC: _______

### Compare Results
- [ ] Plot: Accuracy vs photometric error
- [ ] Thesis insight: "Performance robust to X mag, degrades beyond Y"

## Phase 6: Binary Difficulty Experiments (3-5 days)

### Easy Binaries (small u0, clear caustics)
- [ ] Generate data: `python code/simulate.py --n_pspl 100000 --n_binary 100000 --binary_difficulty easy --output data/raw/events_binary_easy.npz`
- [ ] Train model
- [ ] Evaluate
- [ ] Record ROC AUC: _______ (expect ~0.99!)

### Hard Binaries (large u0, PSPL-like)
- [ ] Generate data: `--binary_difficulty hard`
- [ ] Train model
- [ ] Evaluate
- [ ] Record ROC AUC: _______ (expect ~0.85)

### Compare Results
- [ ] Plot: Easy vs Standard vs Hard performance
- [ ] Thesis insight: "Caustic-crossing events 99% detectable, but 15-20% are intrinsically PSPL-like"

## Phase 7: Results Summary

### Download All Results
- [ ] On laptop: `scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./thesis_results`

### Create Summary Table
| Experiment | ROC AUC | Accuracy | Notes |
|-----------|---------|----------|-------|
| Baseline | _____ | _____ | |
| Dense cadence | _____ | _____ | |
| Sparse cadence | _____ | _____ | |
| Low error | _____ | _____ | |
| High error | _____ | _____ | |
| Easy binaries | _____ | _____ | |
| Hard binaries | _____ | _____ | |

### Key Findings
- [ ] Upper bound: _____ % (best case)
- [ ] Lower bound: _____ % (worst case)
- [ ] Realistic (LSST): _____ %
- [ ] Physical limit: ~____% of binaries are intrinsically hard

## Phase 8: Thesis Figures

### Figure 1: Baseline Performance
- [ ] Confusion matrix
- [ ] ROC curve
- [ ] Sample light curves (PSPL and Binary)

### Figure 2: Cadence Analysis
- [ ] Accuracy vs % coverage
- [ ] Early detection curves

### Figure 3: Photometric Sensitivity
- [ ] Accuracy vs photometric error

### Figure 4: Binary Detectability
- [ ] Easy vs Standard vs Hard comparison
- [ ] Physical interpretation (u0 distribution)

### Figure 5: Combined Heatmap
- [ ] 2D: Cadence vs Error
- [ ] Shows optimal survey parameters

## Phase 9: Thesis Writing

### Results Section
- [ ] Baseline: "We achieve 95-96% accuracy on standard binaries"
- [ ] Cadence: "Performance ranges from X% (5% missing) to Y% (30% missing)"
- [ ] Error: "Robust to 0.15 mag, degrades with poorer photometry"
- [ ] Binary types: "99% for caustic-crossing, 85% for high-u0 events"
- [ ] Early detection: "Feasible at 500 points with 85% accuracy"

### Discussion Section
- [ ] LSST implications: "Will achieve ~95% with typical cadence/photometry"
- [ ] Roman implications: "Could achieve ~97-98% with better parameters"
- [ ] Physical limits: "Not all binaries detectable - intrinsic astrophysical limit"
- [ ] Survey optimization: "Quantitative guidance for design"
- [ ] Future work: Multi-band, more complex topologies

### Conclusions
- [ ] Established performance bounds: 82-99%
- [ ] Identified physical limits: ~15-20% intrinsically hard
- [ ] Demonstrated early classification feasibility
- [ ] Provided survey optimization guidance

## Final Checks
- [ ] All experiments completed
- [ ] All figures generated
- [ ] Results table complete
- [ ] Key findings documented
- [ ] Thesis sections drafted

## Bonus: Optional Experiments
- [ ] Early 300 points (very early detection)
- [ ] Multiple cadence values (10%, 15%, 25%)
- [ ] Intermediate error values (0.08, 0.12, 0.15 mag)
- [ ] Combined stress tests
- [ ] LSST-specific parameters
- [ ] Roman-specific parameters

═══════════════════════════════════════════════════════════════════════════════

## Timeline Estimates

**Minimum (core experiments)**: 2-3 weeks
- Week 1: Baseline + cadence experiments
- Week 2: Error + binary difficulty experiments  
- Week 3: Analysis + writing

**Recommended (complete benchmark)**: 3-4 weeks
- Week 1: Baseline + quick test of all variations
- Week 2-3: Full experimental suite
- Week 4: Analysis, figures, writing

**With buffer (include writing)**: 5-6 weeks
- Weeks 1-3: All experiments
- Week 4: Analysis and figures
- Weeks 5-6: Thesis writing

═══════════════════════════════════════════════════════════════════════════════

## Success = Checked All Core Items! ✓

You will have:
✅ Quantified upper bound (easy binaries): 98-99%
✅ Quantified lower bound (hard binaries): 82-88%
✅ Established realistic performance (LSST-like): 94-96%
✅ Identified physical limits (~15-20% intrinsically hard)
✅ Demonstrated early detection feasibility
✅ Provided survey optimization guidance

**This is a complete benchmark!** 🎯
