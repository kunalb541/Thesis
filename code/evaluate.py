#!/usr/bin/env python3
"""
Distributed Model Evaluation for Microlensing Classification
============================================================

Complete evaluation pipeline with DDP support for multi-GPU evaluation.

Features:
- Full DDP support with proper metric gathering
- Classification metrics (accuracy, precision, recall, F1)
- ROC curves and confusion matrix
- Calibration analysis
- Impact parameter (u₀) dependency
- Early detection performance
- Real-time classification evolution

Author: Kunal Bhatia
Version: 14.4 (Bug fixes - dtype, parameter indexing, memory management, validation, SAMPLING FIX)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import pickle
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import os
import warnings
warnings.filterwarnings("ignore")

import sys
from transformer import MicrolensingTransformer

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Suppress NCCL warnings
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_TIMEOUT'] = '1800'
os.environ['NCCL_DEBUG'] = 'WARN'


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_distributed():
    """Initialize distributed training"""
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if world_size > 1:
        if rank == 0:
            print(f"[Rank {rank}] Initializing process group...")
        dist.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        torch.cuda.set_device(local_rank)
    
    return rank, local_rank, world_size


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(message, rank=0):
    """Print only from rank 0"""
    if rank == 0:
        print(message)


class StableNormalizer:
    """Robust normalizer matching training (for pickle compatibility)"""
    
    def __init__(self, pad_value=-1.0):
        self.pad_value = pad_value
        self.mean = 0.0
        self.std = 1.0
    
    def fit(self, X):
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            valid_values = X[valid_mask]
            self.mean = np.median(valid_values)
            self.std = np.median(np.abs(valid_values - self.mean))
            
            if self.std < 1e-8:
                self.std = 1.0
            
            self.mean = np.clip(self.mean, -100, 100)
            self.std = np.clip(self.std, 0.01, 100)
        
        return self
    
    def transform(self, X):
        X_norm = X.copy()
        valid_mask = (X != self.pad_value) & np.isfinite(X)
        
        if valid_mask.any():
            X_norm[valid_mask] = (X[valid_mask] - self.mean) / self.std
            X_norm[valid_mask] = np.clip(X_norm[valid_mask], -10, 10)
        
        return np.nan_to_num(X_norm, nan=0.0, posinf=10.0, neginf=-10.0)
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)


class EvaluationDataset(Dataset):
    """Simple dataset for evaluation with proper dtype handling"""
    
    def __init__(self, X, y):
        # FIX: Ensure float32 dtype
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ComprehensiveEvaluator:
    """Complete evaluation with DDP support"""
    
    def __init__(self, model_path, normalizer_path, data_path, output_dir, 
                 rank=0, local_rank=0, world_size=1, batch_size=128, n_samples=None):
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print_rank0("="*70, rank)
        print_rank0("MODEL EVALUATION (DDP-READY)", rank)
        print_rank0("="*70, rank)
        print_rank0(f"GPUs: {world_size}", rank)
        print_rank0(f"Device: {self.device}", rank)
        print_rank0(f"Output directory: {self.output_dir}", rank)
        if n_samples:
            print_rank0(f"Sample limit: {n_samples} events", rank)
        
        print_rank0("\nLoading model...", rank)
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print_rank0("Model loaded", rank)
        
        print_rank0("\nLoading normalizer...", rank)
        self.normalizer = self._load_normalizer(normalizer_path)
        print_rank0("Normalizer loaded", rank)
        
        print_rank0("\nLoading data...", rank)
        self.X, self.y, self.params, self.timestamps, self.n_classes = self._load_data(data_path)
        
        print_rank0("\nNormalizing...", rank)
        # FIX: Ensure float32 dtype after normalization
        self.X_norm = self.normalizer.transform(self.X).astype(np.float32)
        
        print_rank0("Getting predictions...", rank)
        self.predictions, self.confidences, self.probs = self._get_predictions()
        
        # Synchronize before computing metrics
        if world_size > 1:
            dist.barrier()
        
        if rank == 0:
            print("\nComputing metrics...")
            self.metrics = self._compute_metrics()
            self._print_summary()
    
    def _load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        config_path = Path(model_path).parent / 'config.json'
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            
            d_model = config.get('d_model', 128)
            nhead = config.get('nhead', 4)
            num_layers = config.get('num_layers', 4)
            dropout = config.get('dropout', 0.1)
            
            if self.rank == 0:
                print(f"   Config: d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
        else:
            if self.rank == 0:
                print("   Warning: config.json not found, using defaults")
            d_model = 128
            nhead = 4
            num_layers = 4
            dropout = 0.1
        
        model = MicrolensingTransformer(
            n_points=1500,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            pad_value=-1.0
        )
        
        state_dict = checkpoint['model_state_dict']
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict)
        
        if self.rank == 0:
            print(f"   Parameters: {count_parameters(model):,}")
        
        return model
    
    def _load_normalizer(self, normalizer_path):
        normalizer_path = Path(normalizer_path)
        
        if not normalizer_path.exists():
            if self.rank == 0:
                print(f"   Warning: Normalizer not found, creating default")
            return StableNormalizer(pad_value=-1.0)
        
        with open(normalizer_path, 'rb') as f:
            normalizer = pickle.load(f)
        
        if self.rank == 0:
            print(f"   Mean={normalizer.mean:.3f}, Std={normalizer.std:.3f}")
        return normalizer
    
    def _load_data(self, data_path):
        data = np.load(data_path)
        X = data['X']
        y = data['y']
        
        if X.ndim == 3:
            X = X.squeeze(1)
        
        # FIX: Ensure float32 from the start
        X = X.astype(np.float32)
        y = y.astype(np.int64)
        
        if 'timestamps' in data:
            timestamps = data['timestamps'].astype(np.float32)
        else:
            timestamps = np.linspace(-100, 100, X.shape[1], dtype=np.float32)
        
        # Detect number of classes
        if 'n_classes' in data:
            n_classes = int(data['n_classes'])
            if self.rank == 0:
                print(f"   Dataset: {n_classes} classes")
        else:
            n_classes = len(np.unique(y))
            if self.rank == 0:
                print(f"   Dataset: {n_classes} classes (inferred)")
        
        params = None
        if 'params_binary_json' in data:
            params_binary = json.loads(str(data['params_binary_json']))
            params_dict = {'binary': params_binary}
            
            if 'params_pspl_json' in data:
                params_pspl = json.loads(str(data['params_pspl_json']))
                params_dict['pspl'] = params_pspl
            
            if 'params_flat_json' in data and n_classes == 3:
                params_flat = json.loads(str(data['params_flat_json']))
                params_dict['flat'] = params_flat
            
            params = params_dict
            
            # FIX: Validate parameter alignment BEFORE sampling
            if self.rank == 0:
                print("\n   Validating parameter alignment (before sampling):")
                for key in params.keys():
                    if key in ['binary', 'pspl', 'flat']:
                        if n_classes == 3:
                            class_id = {'flat': 0, 'pspl': 1, 'binary': 2}.get(key)
                        else:
                            class_id = {'pspl': 0, 'binary': 1}.get(key)
                        
                        if class_id is not None:
                            expected_count = (data['y'] == class_id).sum()
                            actual_count = len(params[key])
                            status = "✓" if expected_count == actual_count else "✗ MISMATCH"
                            print(f"     {status} {key}: expected {expected_count}, got {actual_count}")
                            
                            if expected_count != actual_count:
                                print(f"     WARNING: Parameter count mismatch detected!")
        
        # FIX: Proper sampling that maintains parameter alignment AND gets exact count
        if self.n_samples is not None and self.n_samples < len(X):
            if self.rank == 0:
                print(f"\n   Sampling {self.n_samples} events...")
            
            # Set seed for reproducibility across ranks
            np.random.seed(42)
            
            # CRITICAL FIX: Distribute remainder to ensure exactly n_samples total
            base_per_class = self.n_samples // n_classes
            extra = self.n_samples % n_classes  # Handle remainder
            
            indices_per_class = []
            for c in range(n_classes):
                class_mask = y == c
                # First 'extra' classes get one additional sample
                n_class = base_per_class + (1 if c < extra else 0)
                n_class = min(n_class, class_mask.sum())
                class_indices = np.random.choice(np.where(class_mask)[0], n_class, replace=False)
                indices_per_class.append(class_indices)
            
            all_indices = np.concatenate(indices_per_class)
            # Ensure we have exactly n_samples (trim if needed due to class size limits)
            if len(all_indices) > self.n_samples:
                all_indices = all_indices[:self.n_samples]
            np.random.shuffle(all_indices)
            
            # Store original data for parameter mapping
            original_y = data['y'].copy()
            
            # Sample X and y
            X = X[all_indices]
            y = y[all_indices]
            
            # FIX: Properly subset parameters based on sampled indices
            if params is not None:
                new_params = {}
                
                # Build class-to-param-index mapping for original data
                class_param_indices = {}
                for key in params.keys():
                    if key in ['binary', 'pspl', 'flat']:
                        # Get the class ID
                        if n_classes == 3:
                            class_id = {'flat': 0, 'pspl': 1, 'binary': 2}.get(key)
                        else:
                            class_id = {'pspl': 0, 'binary': 1}.get(key)
                        
                        if class_id is not None:
                            # Get ALL indices for this class in original dataset
                            original_class_indices = np.where(original_y == class_id)[0]
                            # Create mapping: global_index → position_in_class_params
                            class_param_indices[key] = {
                                global_idx: param_idx 
                                for param_idx, global_idx in enumerate(original_class_indices)
                            }
                
                # Now map sampled indices to parameters
                for key in params.keys():
                    if key in class_param_indices:
                        param_indices = []
                        for sample_idx in all_indices:
                            if sample_idx in class_param_indices[key]:
                                param_idx = class_param_indices[key][sample_idx]
                                param_indices.append(param_idx)
                        
                        # Select the corresponding parameters
                        if param_indices:
                            new_params[key] = [params[key][i] for i in param_indices]
                        else:
                            new_params[key] = []
                
                params = new_params
        
        if self.rank == 0:
            print(f"\n   Final dataset statistics:")
            print(f"   Events: {len(X)}")
            if n_classes == 3:
                print(f"   Flat:   {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
                print(f"   PSPL:   {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
                print(f"   Binary: {(y == 2).sum()} ({(y == 2).mean()*100:.1f}%)")
            else:
                print(f"   PSPL:   {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
                print(f"   Binary: {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
            
            if params is not None:
                print(f"\n   Parameter data available (u0 analysis enabled)")
                # FIX: Validate parameter alignment AFTER sampling
                print("   Validating parameter alignment (after sampling):")
                for key in params.keys():
                    if key in ['binary', 'pspl', 'flat']:
                        if n_classes == 3:
                            class_id = {'flat': 0, 'pspl': 1, 'binary': 2}.get(key)
                        else:
                            class_id = {'pspl': 0, 'binary': 1}.get(key)
                        
                        if class_id is not None:
                            n_events = (y == class_id).sum()
                            n_params = len(params[key])
                            status = "✓" if n_events == n_params else "✗ MISMATCH"
                            print(f"     {status} {key}: {n_events} events, {n_params} params")
                            
                            if n_events != n_params:
                                print(f"     ERROR: Parameter alignment broken after sampling!")
                                print(f"     This will cause incorrect u0 analysis.")
            else:
                print("   No parameter data (u0 analysis disabled)")
        
        return X, y, params, timestamps, n_classes
    
    def _get_predictions(self):
        """Get predictions using distributed data loading"""
        # FIX: Dataset now handles dtype conversion internally
        dataset = EvaluationDataset(self.X_norm, self.y)
        
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
            )
        else:
            sampler = None
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )
        
        local_predictions = []
        local_confidences = []
        local_probs = []
        
        with torch.no_grad():
            if self.rank == 0:
                pbar = tqdm(loader, desc="   Evaluating")
            else:
                pbar = loader
            
            for batch_idx, (x_batch, y_batch) in enumerate(pbar):
                x_batch = x_batch.to(self.device, non_blocking=True)
                
                output = self.model(x_batch, return_all=False)
                logits = output['logits'] if 'logits' in output else output['binary']
                probs = F.softmax(logits, dim=1).cpu().numpy()
                
                preds = probs.argmax(axis=1)
                confs = probs.max(axis=1)
                
                local_predictions.extend(preds)
                local_confidences.extend(confs)
                local_probs.append(probs)
        
        local_predictions = np.array(local_predictions)
        local_confidences = np.array(local_confidences)
        local_probs = np.vstack(local_probs)
        
        # FIX: Gather predictions only to rank 0 to save memory
        if self.world_size > 1:
            # Gather sizes from all ranks
            local_size = torch.tensor([len(local_predictions)], device=self.device)
            all_sizes = [torch.zeros(1, dtype=torch.long, device=self.device) 
                        for _ in range(self.world_size)]
            dist.all_gather(all_sizes, local_size)
            all_sizes = [int(s.item()) for s in all_sizes]
            
            # Gather to rank 0 only
            if self.rank == 0:
                # Prepare receiving buffers on rank 0
                gathered_preds = [torch.zeros(s, dtype=torch.long, device=self.device) 
                                 for s in all_sizes]
                gathered_confs = [torch.zeros(s, dtype=torch.float32, device=self.device) 
                                 for s in all_sizes]
                gathered_probs = [torch.zeros(s, self.n_classes, dtype=torch.float32, device=self.device) 
                                 for s in all_sizes]
            else:
                gathered_preds = None
                gathered_confs = None
                gathered_probs = None
            
            # Gather predictions
            local_preds_tensor = torch.from_numpy(local_predictions).long().to(self.device)
            dist.gather(local_preds_tensor, gathered_preds, dst=0)
            
            # Gather confidences
            local_confs_tensor = torch.from_numpy(local_confidences).float().to(self.device)
            dist.gather(local_confs_tensor, gathered_confs, dst=0)
            
            # Gather probabilities
            local_probs_tensor = torch.from_numpy(local_probs).float().to(self.device)
            dist.gather(local_probs_tensor, gathered_probs, dst=0)
            
            # FIX: Free memory on non-zero ranks immediately
            if self.rank != 0:
                del local_preds_tensor, local_confs_tensor, local_probs_tensor
                del local_predictions, local_confidences, local_probs
                torch.cuda.empty_cache()
                # Return dummy values for non-rank-0 processes
                return np.array([]), np.array([]), np.array([[]])
            
            # Combine on rank 0
            predictions = torch.cat(gathered_preds).cpu().numpy()
            confidences = torch.cat(gathered_confs).cpu().numpy()
            probs = torch.cat(gathered_probs).cpu().numpy()
            
            # FIX: Free gathered tensors
            del gathered_preds, gathered_confs, gathered_probs
            del local_preds_tensor, local_confs_tensor, local_probs_tensor
            torch.cuda.empty_cache()
        else:
            predictions = local_predictions
            confidences = local_confidences
            probs = local_probs
        
        return predictions, confidences, probs
    
    def _compute_metrics(self):
        """Compute metrics (only on rank 0)"""
        # Basic metrics
        accuracy = accuracy_score(self.y, self.predictions)
        
        # Per-class metrics
        if self.n_classes == 3:
            target_names = ['Flat', 'PSPL', 'Binary']
        else:
            target_names = ['PSPL', 'Binary']
        
        report = classification_report(
            self.y, self.predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(self.y, self.predictions)
        
        metrics = {
            'accuracy': accuracy,
            'n_classes': self.n_classes,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # Add per-class metrics
        for i, name in enumerate(target_names):
            metrics[f'{name.lower()}_precision'] = report[name]['precision']
            metrics[f'{name.lower()}_recall'] = report[name]['recall']
            metrics[f'{name.lower()}_f1'] = report[name]['f1-score']
        
        return metrics
    
    def _print_summary(self):
        """Print summary (only on rank 0)"""
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS ({self.n_classes} classes)")
        print(f"{'='*70}")
        print(f"Overall Accuracy: {self.metrics['accuracy']*100:.2f}%")
        print(f"\nPer-Class Performance:")
        
        if self.n_classes == 3:
            classes = ['flat', 'pspl', 'binary']
            names = ['Flat', 'PSPL', 'Binary']
        else:
            classes = ['pspl', 'binary']
            names = ['PSPL', 'Binary']
        
        for cls, name in zip(classes, names):
            prec = self.metrics[f'{cls}_precision']
            rec = self.metrics[f'{cls}_recall']
            f1 = self.metrics[f'{cls}_f1']
            print(f"  {name:8s}: Prec={prec*100:5.1f}%, Rec={rec*100:5.1f}%, F1={f1*100:5.1f}%")
        
        print(f"{'='*70}\n")
    
    def plot_roc_curve(self):
        """Plot ROC curves (only on rank 0)"""
        if self.rank != 0:
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        # One-vs-rest ROC curves
        for i, (name, color) in enumerate(zip(class_names, colors)):
            y_true_binary = (self.y == i).astype(int)
            y_score = self.probs[:, i]
            
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                auc = roc_auc_score(y_true_binary, y_score)
                
                ax.plot(fpr, tpr, linewidth=3, color=color,
                       label=f'{name} (AUC = {auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(f'ROC Curves ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        output_path = self.output_dir / 'roc_curve.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix (only on rank 0)"""
        if self.rank != 0:
            return
        
        cm = np.array(self.metrics['confusion_matrix'])
        
        if self.n_classes == 3:
            labels = ['Flat', 'PSPL', 'Binary']
        else:
            labels = ['PSPL', 'Binary']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        
        ax.set_xticks(range(self.n_classes))
        ax.set_yticks(range(self.n_classes))
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_yticklabels(labels, fontsize=12)
        ax.set_xlabel('Predicted label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True label', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                text = ax.text(j, i, cm[i, j], ha="center", va="center",
                             color="white" if cm[i, j] > cm.max()/2 else "black",
                             fontsize=16, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        
        output_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_confidence_distribution(self):
        """Plot confidence distribution (only on rank 0)"""
        if self.rank != 0:
            return
        
        correct = self.predictions == self.y
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bins = np.linspace(0.3 if self.n_classes == 3 else 0.5, 1.0, 50)
        ax.hist(self.confidences[correct], bins=bins, alpha=0.7, color='green',
               label=f'Correct (n={correct.sum()})', edgecolor='black')
        ax.hist(self.confidences[~correct], bins=bins, alpha=0.7, color='red',
               label=f'Incorrect (n={(~correct).sum()})', edgecolor='black')
        
        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'Confidence Distribution ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = self.output_dir / 'confidence_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_calibration_curve(self):
        """Plot calibration curve (only on rank 0)"""
        if self.rank != 0:
            return
        
        correct = self.predictions == self.y
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Left: Calibration bins
        ax = axes[0]
        conf_min = 0.3 if self.n_classes == 3 else 0.5
        conf_bins = np.linspace(conf_min, 1.0, 11)
        accuracies, bin_centers, counts = [], [], []
        
        for i in range(len(conf_bins)-1):
            mask = (self.confidences >= conf_bins[i]) & (self.confidences < conf_bins[i+1])
            if mask.sum() > 0:
                accuracies.append(correct[mask].mean())
                bin_centers.append((conf_bins[i] + conf_bins[i+1]) / 2)
                counts.append(mask.sum())
        
        if len(bin_centers) > 0:
            bars = ax.bar(bin_centers, accuracies, width=0.06, alpha=0.7, edgecolor='black', linewidth=1.5)
            for bar, cnt in zip(bars, counts):
                bar.set_facecolor(plt.cm.Blues(0.3 + 0.7 * cnt / max(counts)))
            
            for bc, acc, cnt in zip(bin_centers, accuracies, counts):
                ax.text(bc, acc + 0.02, f'n={cnt}', ha='center', fontsize=7)
        
        ax.plot([conf_min, 1.0], [conf_min, 1.0], 'r--', linewidth=2, alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
        ax.set_title('Model Calibration', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.3 if self.n_classes == 3 else 0.4, 1.05])
        
        # Right: Scatter plot
        ax = axes[1]
        n_plot = min(5000, len(correct))
        idx = np.random.choice(len(correct), n_plot, replace=False)
        jitter = np.random.normal(0, 0.01, size=n_plot)
        
        ax.scatter(self.confidences[idx][correct[idx]] + jitter[correct[idx]],
                  correct[idx][correct[idx]],
                  alpha=0.3, s=5, color='green', label='Correct')
        ax.scatter(self.confidences[idx][~correct[idx]] + jitter[~correct[idx]],
                  correct[idx][~correct[idx]],
                  alpha=0.3, s=5, color='red', label='Incorrect')
        
        ax.set_xlabel('Confidence Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correctness')
        ax.set_title('Confidence vs Correctness', fontweight='bold')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Wrong', 'Correct'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'calibration.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_example_grid(self, n_per_class=3):
        """Plot example grid (only on rank 0)"""
        if self.rank != 0:
            return
        
        print(f"  Generating example plots...")
        
        correct = self.predictions == self.y
        
        examples = []
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        # Collect examples for each true class
        for true_class, class_name, color in zip(range(self.n_classes), class_names, colors):
            true_mask = self.y == true_class
            correct_mask = true_mask & correct
            incorrect_mask = true_mask & ~correct
            
            # Get correct predictions
            if correct_mask.sum() > 0:
                indices = np.where(correct_mask)[0]
                conf_sorted = indices[np.argsort(-self.confidences[indices])]
                selected = conf_sorted[:n_per_class]
                for idx in selected:
                    examples.append((idx, f'{class_name} (Correct)', 'green'))
            
            # Get incorrect predictions
            if incorrect_mask.sum() > 0:
                indices = np.where(incorrect_mask)[0]
                conf_sorted = indices[np.argsort(-self.confidences[indices])]
                selected = conf_sorted[:1]
                for idx in selected:
                    pred_name = class_names[self.predictions[idx]]
                    examples.append((idx, f'{class_name}→{pred_name}', 'red'))
        
        # Create figure
        n_examples = len(examples)
        n_cols = 4
        n_rows = (n_examples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (idx, label, color) in enumerate(examples):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            flux = self.X[idx]
            valid_mask = flux != -1.0
            times = self.timestamps[valid_mask]
            fluxes = flux[valid_mask]
            
            baseline = 20.0
            magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
            
            true_name = class_names[self.y[idx]]
            pred_name = class_names[self.predictions[idx]]
            
            ax.scatter(times, magnitudes, c=color, s=8, alpha=0.7, edgecolors='black', linewidth=0.3)
            ax.invert_yaxis()
            
            ax.set_title(f'{label}\nTrue: {true_name}, Pred: {pred_name}\nConf: {self.confidences[idx]:.2f}',
                        fontsize=9, color=color, fontweight='bold')
            ax.set_xlabel('Time (days)', fontsize=8)
            ax.set_ylabel('Magnitude', fontsize=8)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=7)
        
        # Hide empty subplots
        for i in range(len(examples), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Example Light Curves ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        output_path = self.output_dir / f'example_grid_{self.n_classes}class.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_real_time_evolution(self, event_idx=None, event_type='binary'):
        """Plot real-time classification evolution (only on rank 0)"""
        if self.rank != 0:
            return
        
        if event_idx is None:
            # Select a good example
            if self.n_classes == 3:
                target_class = {'flat': 0, 'pspl': 1, 'binary': 2}.get(event_type, 2)
            else:
                target_class = {'pspl': 0, 'binary': 1}.get(event_type, 1)
            
            good_examples = np.where(
                (self.y == target_class) & 
                (self.predictions == target_class) & 
                (self.confidences > 0.7)
            )[0]
            
            if len(good_examples) == 0:
                good_examples = np.where(self.y == target_class)[0]
            
            if len(good_examples) == 0:
                return
            
            event_idx = np.random.choice(good_examples)
        
        light_curve = self.X[event_idx]
        light_curve_norm = self.X_norm[event_idx]
        true_label = self.y[event_idx]
        
        # Test multiple fractions
        fractions = np.linspace(0.1, 1.0, 10)
        
        if self.n_classes == 3:
            flat_probs, pspl_probs, binary_probs = [], [], []
        else:
            pspl_probs, binary_probs = [], []
        
        confidences = []
        
        with torch.no_grad():
            for frac in fractions:
                n_points = int(1500 * frac)
                partial_curve = np.full(1500, -1.0, dtype=np.float32)
                partial_curve[:n_points] = light_curve_norm[:n_points]
                
                x = torch.from_numpy(partial_curve).float().unsqueeze(0).to(self.device)
                output = self.model(x, return_all=False)
                logits = output['logits'] if 'logits' in output else output['binary']
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                if self.n_classes == 3:
                    flat_probs.append(probs[0])
                    pspl_probs.append(probs[1])
                    binary_probs.append(probs[2])
                else:
                    pspl_probs.append(probs[0])
                    binary_probs.append(probs[1])
                
                confidences.append(probs.max())
        
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1, 1], hspace=0.3)
        
        # Top: Light curve
        ax1 = fig.add_subplot(gs[0])
        valid_mask = light_curve != -1.0
        times = self.timestamps[valid_mask]
        fluxes = light_curve[valid_mask]
        baseline = 20.0
        magnitudes = baseline - 2.5 * np.log10(np.maximum(fluxes, 1e-10))
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        color = colors[true_label]
        ax1.scatter(times, magnitudes, c=color, s=15, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax1.invert_yaxis()
        
        true_str = class_names[true_label]
        pred_str = class_names[self.predictions[event_idx]]
        ax1.set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        ax1.set_title(f'Light Curve - True: {true_str}, Predicted: {pred_str} (Conf: {self.confidences[event_idx]:.2f})',
                     fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Middle: Class probabilities
        ax2 = fig.add_subplot(gs[1])
        completeness = [f*100 for f in fractions]
        
        if self.n_classes == 3:
            ax2.plot(completeness, flat_probs, 'o-', linewidth=3, markersize=8, 
                    color='gray', label='Flat (No Event)', alpha=0.8)
            ax2.plot(completeness, pspl_probs, 's-', linewidth=3, markersize=8, 
                    color='darkred', label='PSPL', alpha=0.8)
            ax2.plot(completeness, binary_probs, '^-', linewidth=3, markersize=8, 
                    color='darkblue', label='Binary', alpha=0.8)
        else:
            ax2.plot(completeness, pspl_probs, 's-', linewidth=3, markersize=8, 
                    color='darkred', label='PSPL', alpha=0.8)
            ax2.plot(completeness, binary_probs, '^-', linewidth=3, markersize=8, 
                    color='darkblue', label='Binary', alpha=0.8)
        
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, label='50% Threshold')
        ax2.axhline(y=0.8, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='High Confidence')
        
        ax2.fill_between(completeness, 0.8, 1.0, alpha=0.15, color='green')
        ax2.fill_between(completeness, 0.5, 0.8, alpha=0.15, color='yellow')
        ax2.fill_between(completeness, 0.0, 0.5, alpha=0.15, color='lightblue')
        
        ax2.set_ylabel('Class Probability', fontsize=12, fontweight='bold')
        ax2.legend(loc='center left', fontsize=9, bbox_to_anchor=(1, 0.5))
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.05, 1.05])
        
        # Bottom: Overall confidence
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(completeness, confidences, 'd-', linewidth=3, markersize=8, color='purple', label='Overall Confidence')
        ax3.axhline(y=0.8, color='orange', linestyle='--', linewidth=2, label='80% Threshold')
        ax3.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90% Threshold')
        ax3.set_xlabel('Observation Completeness (%)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Prediction Confidence', fontsize=12, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3)
        
        if self.n_classes == 3:
            ax3.set_ylim([0.3, 1.05])
        else:
            ax3.set_ylim([0.4, 1.05])
        
        plt.suptitle(f'Real-Time Classification Evolution - {true_str} Event ({self.n_classes}-Class)', 
                    fontsize=14, fontweight='bold')
        
        event_type_str = event_type.lower()
        output_path = self.output_dir / f'real_time_evolution_{event_type_str}_event_{event_idx}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def plot_early_detection(self):
        """Compute performance vs. observation completeness (only on rank 0)"""
        if self.rank != 0:
            return
        
        print("  Computing early detection performance...")
        
        fractions = [0.1, 0.167, 0.25, 0.5, 0.67, 0.833, 1.0]
        overall_accs = []
        per_class_recalls = [[] for _ in range(self.n_classes)]
        
        for frac in tqdm(fractions, desc="    Testing fractions"):
            predictions = []
            
            with torch.no_grad():
                for i in range(0, len(self.X_norm), self.batch_size):
                    batch_end = min(i + self.batch_size, len(self.X_norm))
                    batch_size_actual = batch_end - i
                    n_points = int(1500 * frac)
                    
                    partial_curves_np = np.full((batch_size_actual, 1500), -1.0, dtype=np.float32)
                    partial_curves_np[:, :n_points] = self.X_norm[i:batch_end, :n_points]
                    
                    x_batch = torch.from_numpy(partial_curves_np).float().to(self.device)
                    
                    output = self.model(x_batch, return_all=False)
                    logits = output['logits'] if 'logits' in output else output['binary']
                    probs = F.softmax(logits, dim=1).cpu().numpy()
                    
                    predictions.extend(probs.argmax(axis=1))
            
            predictions = np.array(predictions)
            overall_accs.append(accuracy_score(self.y, predictions))
            
            # Per-class recall
            for c in range(self.n_classes):
                class_mask = self.y == c
                if class_mask.sum() > 0:
                    class_recall = (predictions[class_mask] == c).mean()
                    per_class_recalls[c].append(class_recall)
                else:
                    per_class_recalls[c].append(0.0)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 7))
        completeness = [f*100 for f in fractions]
        
        ax.plot(completeness, [a*100 for a in overall_accs], 'o-', linewidth=3, markersize=10,
               color='purple', label='Overall Accuracy')
        
        if self.n_classes == 3:
            class_names = ['Flat', 'PSPL', 'Binary']
            colors = ['gray', 'darkred', 'darkblue']
        else:
            class_names = ['PSPL', 'Binary']
            colors = ['darkred', 'darkblue']
        
        for c, (name, color) in enumerate(zip(class_names, colors)):
            ax.plot(completeness, [r*100 for r in per_class_recalls[c]], 's-', 
                   linewidth=2.5, markersize=8, color=color, label=f'{name} Recall', alpha=0.7)
        
        # Add thresholds
        ax.axhline(y=33.3 if self.n_classes == 3 else 50, color='red', linestyle='--', 
                  linewidth=1, alpha=0.5, label='Random' if self.n_classes == 3 else 'Random')
        ax.axhline(y=70, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Target (70%)')
        ax.axvline(x=50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='50% observed')
        
        # Add annotations
        idx_10 = fractions.index(0.1)
        idx_50 = fractions.index(0.5)
        
        ax.annotate(f'{overall_accs[idx_10]*100:.1f}%',
                   xy=(10, overall_accs[idx_10]*100),
                   xytext=(15, overall_accs[idx_10]*100 + 5),
                   fontsize=9, fontweight='bold', color='red',
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        ax.annotate(f'{overall_accs[idx_50]*100:.1f}%',
                   xy=(50, overall_accs[idx_50]*100),
                   xytext=(55, overall_accs[idx_50]*100 - 5),
                   fontsize=9, fontweight='bold', color='green',
                   arrowprops=dict(arrowstyle='->', color='green'))
        
        ax.set_xlabel('Observation Completeness (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Performance (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Early Detection Performance ({self.n_classes}-Class)', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'early_detection.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def analyze_u0_dependency(self, n_bins=10, threshold=0.3):
        """Analyze u0 dependency (Binary class only, only on rank 0)"""
        if self.rank != 0:
            return None
        
        if self.params is None or 'binary' not in self.params:
            print("\nSkipping u0 analysis (no binary parameter data)")
            return None
        
        print(f"\n{'='*70}")
        print("u0 DEPENDENCY ANALYSIS (Binary Class)")
        print(f"{'='*70}")
        
        binary_params = self.params['binary']
        
        if self.n_classes == 3:
            binary_mask = self.y == 2
            binary_class_id = 2
        else:
            binary_mask = self.y == 1
            binary_class_id = 1
        
        # FIX: Validate parameter count matches binary event count
        n_binary_events = binary_mask.sum()
        n_binary_params = len(binary_params)
        
        if n_binary_events != n_binary_params:
            print(f"ERROR: Binary event count ({n_binary_events}) != param count ({n_binary_params})")
            print("This indicates a parameter indexing issue.")
            print("u0 analysis cannot proceed - parameter indices are misaligned with events.")
            return None
        
        u0_values = np.array([p['u_0'] for p in binary_params])
        
        # Validate u0 values are reasonable
        if u0_values.min() < 0 or u0_values.max() > 10:
            print(f"WARNING: Unusual u0 range detected: [{u0_values.min():.3f}, {u0_values.max():.3f}]")
            print("Expected u0 ∈ [0, ~2]. Check parameter extraction.")
        
        u0_bins = np.linspace(u0_values.min(), u0_values.max(), n_bins + 1)
        u0_centers = (u0_bins[:-1] + u0_bins[1:]) / 2
        
        accuracies = []
        counts = []
        
        # Get predictions for binary events only
        binary_predictions = self.predictions[binary_mask]
        binary_true = self.y[binary_mask]
        
        for i in range(n_bins):
            u0_low, u0_high = u0_bins[i], u0_bins[i+1]
            in_bin = (u0_values >= u0_low) & (u0_values < u0_high)
            
            if in_bin.sum() > 0:
                bin_true = binary_true[in_bin]
                bin_pred = binary_predictions[in_bin]
                acc = accuracy_score(bin_true, bin_pred)
                accuracies.append(acc)
                counts.append(int(in_bin.sum()))
            else:
                accuracies.append(np.nan)
                counts.append(0)
        
        threshold_idx = np.argmin(np.abs(u0_centers - threshold))
        acc_at_threshold = accuracies[threshold_idx] if not np.isnan(accuracies[threshold_idx]) else None
        
        n_below = int((u0_values < threshold).sum())
        n_above = int((u0_values >= threshold).sum())
        
        print(f"Physical Detection Threshold: u₀ = {threshold}")
        print(f"Accuracy at threshold: {acc_at_threshold*100:.1f}%" if acc_at_threshold else "N/A")
        print(f"\nBinary Event Distribution:")
        print(f"  Below threshold (u₀ < {threshold}): {n_below} ({n_below/len(u0_values)*100:.1f}%)")
        print(f"  Above threshold (u₀ ≥ {threshold}): {n_above} ({n_above/len(u0_values)*100:.1f}%)")
        
        if n_below < 10:
            print(f"  WARNING: Very few events below threshold ({n_below}). Results may be unreliable.")
        
        return {
            'u0_bins': [float(x) for x in u0_bins],
            'u0_centers': [float(x) for x in u0_centers],
            'accuracies': [float(a) if not np.isnan(a) else None for a in accuracies],
            'counts': counts,
            'all_u0': [float(x) for x in u0_values],
            'threshold': float(threshold),
            'accuracy_at_threshold': float(acc_at_threshold) if acc_at_threshold else None,
            'events_below_threshold': n_below,
            'events_above_threshold': n_above
        }
    
    def plot_u0_dependency(self, u0_results, threshold=0.3):
        """Plot u0 dependency (only on rank 0)"""
        if self.rank != 0 or u0_results is None:
            return
        
        u0_centers = u0_results['u0_centers']
        accuracies = [a*100 if a is not None else None for a in u0_results['accuracies']]
        counts = u0_results['counts']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        valid_indices = [i for i, a in enumerate(accuracies) if a is not None]
        valid_u0 = [u0_centers[i] for i in valid_indices]
        valid_acc = [accuracies[i] for i in valid_indices]
        
        ax1.plot(valid_u0, valid_acc, 'o-', linewidth=2.5, markersize=10, color='#2E86AB')
        ax1.axvline(x=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Physical Limit (u₀ = {threshold})')
        ax1.axhline(y=70, color='gray', linestyle=':', alpha=0.5, label='Target (70%)')
        
        ax1.set_ylabel('Binary Classification Accuracy (%)', fontsize=13)
        ax1.set_title('Binary Class Accuracy vs. Impact Parameter', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 105])
        
        for u, a, c in zip(valid_u0, valid_acc, [counts[i] for i in valid_indices]):
            ax1.annotate(f'{a:.1f}%\n(n={c})', 
                        xy=(u, a), xytext=(0, 10), textcoords='offset points',
                        ha='center', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax2.bar(u0_centers, counts, width=(u0_centers[1]-u0_centers[0])*0.8, 
               color='#A23B72', alpha=0.7, edgecolor='black')
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Impact Parameter u₀', fontsize=13)
        ax2.set_ylabel('Number of Binary Events', fontsize=13)
        ax2.set_title('Distribution of Impact Parameters (Binary Class Only)', fontsize=12)
        ax2.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'u0_dependency.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_path.name}")
        plt.close()
    
    def generate_all_plots(self, include_u0=True, include_early=False, n_evolution_per_type=3, 
                          u0_threshold=0.3, u0_bins=10):
        """Generate all visualizations (only on rank 0)"""
        if self.rank != 0:
            return
        
        print(f"\n{'='*70}")
        print(f"GENERATING VISUALIZATIONS ({self.n_classes} classes)")
        print(f"{'='*70}\n")
        
        print("1. ROC Curves...")
        self.plot_roc_curve()
        
        print("\n2. Confusion Matrix...")
        self.plot_confusion_matrix()
        
        print("\n3. Confidence Distribution...")
        self.plot_confidence_distribution()
        
        print("\n4. Calibration Curve...")
        self.plot_calibration_curve()
        
        print("\n5. Example Grid...")
        self.plot_example_grid(n_per_class=3)
        
        print(f"\n6. Real-Time Evolution ({n_evolution_per_type} examples per class)...")
        
        if self.n_classes == 3:
            event_types = ['flat', 'pspl', 'binary']
        else:
            event_types = ['pspl', 'binary']
        
        for event_type in event_types:
            print(f"   Generating {event_type.capitalize()} event evolutions...")
            for i in range(n_evolution_per_type):
                self.plot_real_time_evolution(event_type=event_type)
        
        if include_early:
            print("\n7. Early Detection Performance...")
            self.plot_early_detection()
        
        if include_u0 and self.params is not None and 'binary' in self.params:
            print("\n8. u0 Dependency Analysis (Binary class only)...")
            u0_results = self.analyze_u0_dependency(n_bins=u0_bins, threshold=u0_threshold)
            if u0_results:
                self.plot_u0_dependency(u0_results, threshold=u0_threshold)
                
                u0_report_path = self.output_dir / 'u0_report.json'
                with open(u0_report_path, 'w') as f:
                    json.dump(u0_results, f, indent=2)
                print(f"  ✓ Saved: {u0_report_path.name}")
        
        print(f"\n{'='*70}")
        print(f"All visualizations saved to: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def save_results(self):
        """Save evaluation results (only on rank 0)"""
        if self.rank != 0:
            return
        
        results = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in self.metrics.items() if k not in ['classification_report', 'confusion_matrix']},
            'classification_report': self.metrics['classification_report'],
            'confusion_matrix': self.metrics['confusion_matrix'],
            'n_classes': self.n_classes,
            'n_samples': int(len(self.y)),
            'high_confidence_80': int((self.confidences >= 0.8).sum()),
            'high_confidence_90': int((self.confidences >= 0.9).sum()),
            'has_u0_analysis': self.params is not None and 'binary' in self.params
        }
        
        output_path = self.output_dir / 'evaluation_summary.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive model evaluation with DDP support'
    )
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--u0_threshold', type=float, default=0.3)
    parser.add_argument('--u0_bins', type=int, default=10)
    parser.add_argument('--no_u0', action='store_true')
    parser.add_argument('--early_detection', action='store_true',
                       help='Compute early detection curve')
    parser.add_argument('--n_evolution_per_type', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    
    args = parser.parse_args()
    
    # Setup distributed
    rank, local_rank, world_size = setup_distributed()
    
    # Find experiment directory
    results_dir = Path('../results')
    exp_dirs = sorted(results_dir.glob(f'{args.experiment_name}_*'))
    
    if not exp_dirs:
        print_rank0(f"No experiment found matching: {args.experiment_name}", rank)
        cleanup_distributed()
        return
    
    exp_dir = exp_dirs[-1]
    model_path = exp_dir / 'best_model.pt'
    normalizer_path = exp_dir / 'normalizer.pkl'
    
    if not model_path.exists():
        print_rank0(f"Model not found: {model_path}", rank)
        cleanup_distributed()
        return
    
    print_rank0(f"Using experiment: {exp_dir.name}", rank)
    output_dir = exp_dir / 'evaluation'
    
    # Synchronize before creating evaluator
    if world_size > 1:
        dist.barrier()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(
        model_path=str(model_path),
        normalizer_path=str(normalizer_path),
        data_path=args.data,
        output_dir=str(output_dir),
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        batch_size=args.batch_size,
        n_samples=args.n_samples
    )
    
    # Synchronize before plotting (only rank 0 plots)
    if world_size > 1:
        dist.barrier()
    
    # Generate plots (only on rank 0)
    evaluator.generate_all_plots(
        include_u0=not args.no_u0,
        include_early=args.early_detection,
        n_evolution_per_type=args.n_evolution_per_type,
        u0_threshold=args.u0_threshold,
        u0_bins=args.u0_bins
    )
    
    # Save results (only on rank 0)
    evaluator.save_results()
    
    # Synchronize before cleanup
    if world_size > 1:
        dist.barrier()
    
    print_rank0("\nEvaluation complete!", rank)
    
    # Cleanup distributed
    cleanup_distributed()


if __name__ == '__main__':
    main()
