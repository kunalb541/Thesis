import os
import sys
import json
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import warnings
import random

try:
    current_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(current_dir))
    from model import GodModeCausalGRU, GRUConfig
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

except ImportError:
    def count_parameters(model): return 0

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CLIP_NORM = 1.0
DEFAULT_LR = 3e-4
SEED = 42

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64, np.floating)): return float(obj)
        if isinstance(obj, (np.int32, np.int64, np.integer)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def setup_distributed():
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
            # FASTEST SETTINGS
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            return rank, local_rank, world_size, True
    return 0, 0, 1, False

def setup_logging(rank: int, output_dir: Path, debug_mode: bool):
    logger = logging.getLogger("TRAIN")
    level = logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if rank == 0:
        handlers.append(logging.FileHandler(output_dir / "training.log"))
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format='%(asctime)s | Rank %(rank)s | %(levelname)s | %(message)s' if rank!=0 else '%(asctime)s | %(message)s')
    if rank != 0: logger = logging.LoggerAdapter(logger, {'rank': rank})
    return logger

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class MicrolensingDataset(Dataset):
    def __init__(self, flux: np.ndarray, delta_t: np.ndarray, labels: np.ndarray, stats: dict = None):
        self.flux = torch.from_numpy(np.ascontiguousarray(flux)).float()
        self.delta_t = torch.from_numpy(np.ascontiguousarray(delta_t)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        self.flux = torch.nan_to_num(self.flux, nan=0.0)
        self.delta_t = torch.nan_to_num(self.delta_t, nan=0.0)
        self.padding_mask = (self.flux != 0.0)
        self.lengths = self.padding_mask.sum(dim=1).long().clamp(min=1)

        if stats is not None:
            self.flux = torch.where(
                self.padding_mask,
                (self.flux - stats['median']) / stats['iqr'],
                torch.tensor(0.0, dtype=self.flux.dtype)
            )

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx): return self.flux[idx], self.delta_t[idx], self.lengths[idx], self.labels[idx]

def train_epoch(model, loader, optimizer, scaler, class_weights, device, rank, epoch, logger, config):
    model.train()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)
    
    # Disable TQDM on workers for speed
    iterator = loader if rank != 0 else tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for flux, delta_t, lengths, labels in iterator:
        flux, delta_t, lengths, labels = flux.to(device, non_blocking=True), delta_t.to(device, non_blocking=True), lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(flux, delta_t, lengths=lengths)
        
        if config.hierarchical:
            loss = F.nll_loss(out['logits'], labels, weight=class_weights)
        else:
            loss = F.cross_entropy(out['logits'], labels, weight=class_weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()
        
        with torch.no_grad():
            total_loss += loss.detach() * labels.size(0)
            correct += (out['probs'].argmax(dim=1) == labels).sum()
            total_samples += labels.size(0)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    
    return float(total_loss / total_samples), float(correct / total_samples)

@torch.no_grad()
def evaluate(model, loader, class_weights, device, rank, config):
    model.eval()
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0.0, device=device)
    total_samples = torch.tensor(0.0, device=device)

    for flux, delta_t, lengths, labels in loader:
        flux, delta_t, lengths, labels = flux.to(device, non_blocking=True), delta_t.to(device, non_blocking=True), lengths.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(flux, delta_t, lengths=lengths)
        if config.hierarchical:
            loss = F.nll_loss(out['logits'], labels, weight=class_weights)
        else:
            loss = F.cross_entropy(out['logits'], labels, weight=class_weights)
        
        total_loss += loss * labels.size(0)
        correct += (out['probs'].argmax(dim=1) == labels).sum()
        total_samples += labels.size(0)

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
    return {'loss': float(total_loss / total_samples), 'accuracy': float(correct / total_samples)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=4)
    # Model args
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--compile', action='store_true') # Kept for API compat
    args = parser.parse_args()
    
    rank, local_rank, world_size, is_ddp = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')

    output_dir = Path(__file__).resolve().parent.parent / 'results' / f"{args.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if rank == 0: output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(rank, output_dir, False)
    set_seed(SEED)

    if rank == 0: logger.info(f"Loading data from {args.data}...")
    data = np.load(args.data, allow_pickle=True)
    flux, delta_t, labels = data['flux'].squeeze(), data['delta_t'].squeeze() if 'delta_t' in data else np.zeros_like(data['flux']), data['labels']
    
    # Simple stats
    s_flux = flux[:10000][flux[:10000] != 0]
    stats = {'median': float(np.median(s_flux)), 'iqr': float(np.subtract(*np.percentile(s_flux, [75, 25])))}

    train_ds = MicrolensingDataset(*train_test_split(flux, delta_t, labels, test_size=0.2, stratify=labels, random_state=SEED)[::2], stats=stats)
    val_ds = MicrolensingDataset(*train_test_split(flux, delta_t, labels, test_size=0.2, stratify=labels, random_state=SEED)[1::2], stats=stats)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=DistributedSampler(train_ds, shuffle=True), num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=DistributedSampler(val_ds, shuffle=False), num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    config = GRUConfig(d_model=args.d_model, n_layers=args.n_layers)
    model = GodModeCausalGRU(config, dtype=torch.bfloat16).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=DEFAULT_LR, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler()
    weights = torch.tensor([1.0, 1.0, 1.0], device=device) # Simplified

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)
        tl, ta = train_epoch(model, train_loader, optimizer, scaler, weights, device, rank, epoch, logger, config)
        if epoch % 5 == 0 or epoch == 1:
            res = evaluate(model, val_loader, weights, device, rank, config)
            if rank == 0:
                logger.info(f"Ep {epoch} | Tr Loss: {tl:.4f} Acc: {ta:.4f} | Val Loss: {res['loss']:.4f} Acc: {res['accuracy']:.4f}")
                if res['accuracy'] > best_acc:
                    best_acc = res['accuracy']
                    torch.save(model.module.state_dict(), output_dir / "best_model.pt")

    if dist.is_initialized(): dist.destroy_process_group()

if __name__ == '__main__':
    main()
