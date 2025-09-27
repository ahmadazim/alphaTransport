# --- NEW UTILS / DATASETS / LOADERS -----------------------------------------
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 0) Feature standardization computed on the union (plasma ∪ csf), per protein
def compute_feature_standardizer(df_plasma, df_csf, eps=1e-6):
    # Use float64 for stable stats, cast later
    stack = np.vstack([df_plasma.values, df_csf.values]).astype(np.float64)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0)
    std[std < eps] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)

def apply_standardizer_np(arr, mean, std):
    return ((arr.astype(np.float32) - mean) / std).astype(np.float32)

# 1) Unpaired dataset that keeps data on CPU; we move to device inside training loops
class MatrixDataset(Dataset):
    def __init__(self, X_np: np.ndarray):
        """
        X_np: (N, D) float32 numpy array (CPU memory)
        """
        assert X_np.ndim == 2, "Expected 2D array"
        self.X = torch.from_numpy(X_np)  # CPU tensor

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]  # CPU tensor; move to device at use-time

# 2) Paired dataset for pretraining (plasma, csf) matched rows
class PairedMatrixDataset(Dataset):
    def __init__(self, X0_np: np.ndarray, X1_np: np.ndarray):
        """
        X0_np: (N, D) float32 numpy array
        X1_np: (N, D) float32 numpy array
        """
        assert X0_np.shape == X1_np.shape, "Paired arrays must be same shape"
        self.X0 = torch.from_numpy(X0_np)  # CPU tensors
        self.X1 = torch.from_numpy(X1_np)

    def __len__(self):
        return self.X0.shape[0]

    def __getitem__(self, idx):
        return self.X0[idx], self.X1[idx]  # CPU; move to device later

def build_loader_from_np(X_np: np.ndarray, batch_size: int, shuffle=True, pin=False):
    ds = MatrixDataset(X_np)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin)

def build_paired_loader_from_np(X0_np: np.ndarray, X1_np: np.ndarray, batch_size: int, shuffle=True, pin=False):
    ds = PairedMatrixDataset(X0_np, X1_np)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True, pin_memory=pin)

# 3) Paired pretraining that uses matched (x0,x1) ONLY (df_ov)
def pretraining_stage_paired(model, ema_model, dl_pairs, num_steps=5000, eps=1.0, lr=1e-3, ema_decay=0.9):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist = []
    it = iter(dl_pairs)
    for step in range(num_steps):
        try:
            x0_cpu, x1_cpu = next(it)
        except StopIteration:
            it = iter(dl_pairs)
            x0_cpu, x1_cpu = next(it)
        # Move batch to device here (dataset stays on CPU to save GPU RAM)
        x0 = x0_cpu.to(device, non_blocking=True)
        x1 = x1_cpu.to(device, non_blocking=True)

        t = torch.rand(x0.shape[0], 1, device=device)
        loss = compute_bidirectional_loss(model, x0, x1, t, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(model, ema_model, ema_decay)

        loss_hist.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"[Pretraining (paired)] Step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    return loss_hist
# --- END NEW -----------------------------------------------------------------



# --- WIRING WITH REAL DATA ---------------------------------------------------
# Assumptions:
#   - df_plasma: (17587, 7290) plasma samples × proteins
#   - df_csf:    ( 1941, 7290) CSF    samples × proteins
#   - df_ov:     (~1500, 14580) matched rows, first 7290 cols = plasma, last 7290 cols = csf
#   - Columns in df_plasma and df_csf are aligned and identical
import pandas as pd

assert list(df_plasma.columns) == list(df_csf.columns), "plasma/csf columns must align 1:1"
D = df_plasma.shape[1]  # 7290

# Compute per-protein z-score params over the union (plasma ∪ csf) so domains are on a common scale
mean, std = compute_feature_standardizer(df_plasma, df_csf)

# Build standardized numpy arrays (float32, on CPU)
X_plasma_np = apply_standardizer_np(df_plasma.values, mean, std)          # (17587, D)
X_csf_np    = apply_standardizer_np(df_csf.values,    mean, std)          # ( 1941, D)

# Split df_ov into paired (x0, x1), standardize with the SAME mean/std
ov_np = df_ov.values  # shape (~1500, 2*D)
assert ov_np.shape[1] == 2*D, "df_ov must have 2*D columns (plasma then csf)"
X_ov0_np = apply_standardizer_np(ov_np[:, :D], mean, std)                 # (~1500, D)
X_ov1_np = apply_standardizer_np(ov_np[:,  D:], mean, std)                # (~1500, D)

# DataLoaders (keep large matrices on CPU; pin_memory for faster H→D transfer if CUDA)
pin = (device.type == "cuda")
dl_pairs  = build_paired_loader_from_np(X_ov0_np, X_ov1_np, batch_size=batch_size, shuffle=True, pin=pin)
dl_plasma = build_loader_from_np(X_plasma_np, batch_size=batch_size, shuffle=True, pin=pin)
dl_csf    = build_loader_from_np(X_csf_np,    batch_size=batch_size, shuffle=True, pin=pin)

# Model: same architecture as before, but adapt input/output dims to D proteins
# input to the net is [s, t, x] → dimension = 1 + 1 + D = D + 2
model = BidirectionalDriftNet(input_dim=D + 2, hidden_dim=hidden_dim, output_dim=D).to(device)
ema_model = BidirectionalDriftNet(input_dim=D + 2, hidden_dim=hidden_dim, output_dim=D).to(device)
ema_model.load_state_dict(model.state_dict())

print("----- Pretraining Stage (paired, using df_ov) -----")
pretrain_loss = pretraining_stage_paired(
    model, ema_model, dl_pairs,
    num_steps=pretrain_steps, eps=eps, lr=1e-3, ema_decay=ema_decay
)

print("----- Finetuning Stage (unpaired, full plasma/csf) -----")
finetune_loss, saved_models = finetuning_stage(
    model, ema_model, dl_plasma, dl_csf,
    num_finetune_steps=finetune_steps, eps=eps, lr=1e-4,
    alpha_step=alpha_step, ema_decay=ema_decay, save_freq=1000
)

# Optional: plot loss as you already do
plt.figure(figsize=(6,4))
plt.plot(pretrain_loss, label="Pretraining (paired)")
plt.plot(np.arange(len(pretrain_loss), len(pretrain_loss)+len(finetune_loss)), finetune_loss, label="Finetuning (unpaired)")
plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss History"); plt.legend(); plt.show()
# --- END WIRING --------------------------------------------------------------