# Quick Reference Card

## Key Commands

### On Your Laptop
```bash
# Setup repository
./setup_repo.sh

# Commit and push
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/thesis-microlens.git
git push -u origin main
```

### On bwUniCluster 3.0

**Initial Setup (once)**
```bash
git clone https://github.com/USERNAME/thesis-microlens.git
cd thesis-microlens
conda create -n microlens python=3.10 -y
conda activate microlens
pip install tensorflow[and-cuda] numpy scipy pandas matplotlib seaborn scikit-learn tqdm VBMicrolensing
chmod +x slurm/*.sh
```

**Quick Test**
```bash
salloc --partition=gpu_mi300 --gres=gpu:4 --cpus-per-gpu=24 --mem-per-gpu=128200mb --time=4:00:00
conda activate microlens
cd ~/thesis-microlens/code
python test_quick.py
exit
```

**Run Baseline Training**
```bash
sbatch slurm/slurm_train_baseline.sh
squeue -u hd_vm305  # check status
tail -f logs/train_baseline_*.out  # monitor progress
```

**Evaluate Model**
```bash
cd ~/thesis-microlens/code
python evaluate.py \
    --model ../models/baseline_model.keras \
    --data ../data/raw/events_1M.npz \
    --output_dir ../results/baseline
```

**Download Results**
```bash
# On your laptop
scp -r hd_vm305@uc3.scc.kit.edu:~/thesis-microlens/results ./
```

## File Locations

**On Cluster:**
- Project: `/u/hd_vm305/thesis-microlens/`
- Data: `/u/hd_vm305/thesis-microlens/data/raw/events_1M.npz`
- Models: `/u/hd_vm305/thesis-microlens/models/`
- Results: `/u/hd_vm305/thesis-microlens/results/`

## Important Parameters

**GPU Settings:**
- Partition: `gpu_mi300`
- GPUs: 4x AMD MI300
- Memory per GPU: 128200mb
- CPUs per GPU: 24

**Training:**
- Batch size: 128 (optimized for 4 GPUs)
- Epochs: 50
- Mixed precision: Enabled
- TimeDistributed: Preserved for real-time classification

**Binary Parameters (UNCHANGED):**
- s: [0.1, 2.5] (separation)
- q: [0.1, 1.0] (mass ratio)
- rho: [0.01, 0.1] (source size)
- These ensure distinct caustic crossings

## Troubleshooting

**No GPUs detected:**
```bash
module load devel/cuda/12.1
export ROCR_VISIBLE_DEVICES=0,1,2,3
```

**Out of memory:**
- Reduce batch_size in slurm script to 64 or 32

**Training too slow:**
- Verify 4 GPUs are being used: `rocm-smi`
- Check logs for GPU utilization

**Cancel jobs:**
```bash
scancel -u hd_vm305  # cancel all your jobs
```

## Expected Results

- **Training time**: 6-12 hours (1M samples, 50 epochs, 4 GPUs)
- **Accuracy**: >95% (binary vs PSPL)
- **ROC AUC**: >0.98
- **Early detection**: ~85% accuracy at 500 points

## Contact

For cluster issues: bwUniCluster support
For code issues: Check logs in `logs/` directory
