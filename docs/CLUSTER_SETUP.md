# bwUniCluster 3.0 Setup Guide

Complete setup instructions for running your thesis project on the cluster.

---

## 🖥️ Cluster Specifications

**Hardware**:
- **GPUs**: 4× AMD MI300 (128GB each)
- **CPUs**: 24 cores per GPU
- **Memory**: 128GB per GPU
- **Software**: ROCm 6.0, PyTorch 2.1+

**Your allocation**:
- Partition: `gpu_mi300`
- Max time: 72 hours per job
- Max GPUs: 4 simultaneous

---

## 🚀 Initial Setup (One-Time)

### Step 1: Clone Repository
```bash
