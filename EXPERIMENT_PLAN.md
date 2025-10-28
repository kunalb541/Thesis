# Systematic Experiment Plan (Clean Start)

**Start Date:** October 28, 2025  
**Status:** Fresh start after fixing aggregation bug

---

## ✅ Prerequisites (COMPLETE)

- [x] Environment setup
- [x] Code fixed (mean aggregation)
- [x] Baseline dataset generated (1M events)
- [x] Cadence datasets generated (05, 40)
- [x] Error datasets generated (low)

---

## 📋 Experiment Queue

### **Priority 1: Core Experiments** (Week 1)

#### 1.1 Baseline
- **Dataset:** `events_baseline_1M.npz` (wide u₀ range)
- **Status:** ⏳ Ready to train
- **Expected:** 70-75% accuracy
- **Purpose:** Establish realistic population performance
- **Command:**
```bash
python train.py --data ../data/raw/events_baseline_1M.npz \
    --output ../models/baseline.pt --epochs 50 --batch_size 128 \
    --experiment_name baseline
```

#### 1.2 Distinct
- **Dataset:** `events_distinct_1M.npz` (u₀ < 0.15)
- **Status:** ⏳ Need to generate
- **Expected:** 85-95% accuracy
- **Purpose:** Best-case performance with guaranteed caustics
- **Command:**
```bash
# Generate
python simulate.py --n_pspl 500000 --n_binary 500000 \
    --output ../data/raw/events_distinct_1M.npz --binary_params distinct

# Train
python train.py --data ../data/raw/events_distinct_1M.npz \
    --output ../models/distinct.pt --epochs 50 --batch_size 128 \
    --experiment_name distinct
```

#### 1.3 Cadence Dense
- **Dataset:** `events_cadence_05.npz` (5% missing)
- **Status:** ✅ Dataset ready
- **Expected:** 73-78% accuracy
- **Purpose:** Show importance of dense observations
- **Command:**
```bash
python train.py --data ../data/raw/events_cadence_05.npz \
    --output ../models/cadence_05.pt --epochs 50 --batch_size 128 \
    --experiment_name cadence_05
```

---

### **Priority 2: Cadence Studies** (Week 2)

#### 2.1 Cadence Sparse
- **Dataset:** `events_cadence_40.npz` (40% missing)
- **Status:** ✅ Dataset ready
- **Expected:** 60-65% accuracy
- **Purpose:** Worst-case observation scenario

#### 2.2 Cadence Medium
- **Dataset:** Need to generate (30% missing)
- **Status:** ⏳ Generate next
- **Expected:** 65-70% accuracy

---

### **Priority 3: Photometric Quality** (Week 2-3)

#### 3.1 Low Error
- **Dataset:** `events_error_low.npz` (0.05 mag)
- **Status:** ✅ Dataset ready
- **Expected:** 75-80% accuracy
- **Purpose:** Space-based quality (Roman)

#### 3.2 High Error
- **Dataset:** Need to generate (0.20 mag)
- **Status:** ⏳ Generate next
- **Expected:** 65-70% accuracy
- **Purpose:** Poor ground conditions

---

### **Priority 4: Binary Types** (Week 3)

#### 4.1 Planetary
- **Dataset:** Need to generate (q < 0.01)
- **Status:** ⏳ Generate next
- **Expected:** 70-80% accuracy
- **Purpose:** Planet-hosting systems

#### 4.2 Stellar
- **Dataset:** Need to generate (q > 0.3)
- **Status:** ⏳ Generate next
- **Expected:** 70-75% accuracy
- **Purpose:** Binary stars

---

## 📊 Results Tracking

### Completed Experiments

| Experiment | Date | Accuracy | ROC AUC | Notes |
|------------|------|----------|---------|-------|
| - | - | - | - | - |

### In Progress

| Experiment | Started | Expected | Status |
|------------|---------|----------|--------|
| - | - | - | - |

---

## 🎯 Success Criteria

**Minimum for thesis:**
- ✅ Baseline: >70% accuracy
- ✅ Distinct: >85% accuracy  
- ✅ At least 2 cadence experiments
- ✅ u₀ distribution analysis

**Nice to have:**
- ⏳ All 8-10 experiments complete
- ⏳ Comprehensive comparison plots
- ⏳ Early detection analysis for all

---

## 📝 Analysis TODO

- [ ] u₀ distribution analysis
- [ ] Misclassification analysis by u₀
- [ ] Cadence comparison plot
- [ ] Error comparison plot
- [ ] Early detection curves
- [ ] Real-time benchmarking for all models

---

## 📅 Timeline

**Week 1 (Current):**
- Train: Baseline, Distinct, Cadence_05
- Analyze: Initial results, u₀ distributions
- Write: Chapter 3 (Methodology)

**Week 2:**
- Train: Remaining cadence, error experiments
- Analyze: Cadence/error comparisons
- Write: Chapter 4.1-4.2 (Results)

**Week 3:**
- Train: Binary type experiments
- Analyze: Complete comparisons
- Write: Chapter 4.3-4.4 (Results cont.)

**Week 4:**
- Final analysis and figures
- Write: Chapter 5 (Discussion), Chapter 6 (Conclusion)
- Thesis polish

---

**Last Updated:** October 28, 2025
