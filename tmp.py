# ===========================
# EVALUATION: HELD-OUT PAIRED
# ===========================
import numpy as np
import torch
import matplotlib.pyplot as plt

# --- 0) Helpers: standardization / inverse ---
def apply_standardizer_np(arr, mean, std):
    return ((arr.astype(np.float32) - mean) / std).astype(np.float32)

def invert_standardizer_np(arr_std, mean, std):
    return (arr_std * std) + mean

# --- 1) Prepare X0 (plasma) / X1 (csf) from df_test ---
def prepare_test_arrays(df_test, D, mean, std, standardize=True):
    """
    df_test: DataFrame with shape (N, 2D); first D cols plasma, last D cols csf.
    D: number of proteins/features
    mean,std: arrays of shape (D,) from training
    """
    X0 = df_test.iloc[:, :D].values.astype(np.float32)
    X1 = df_test.iloc[:,  D:].values.astype(np.float32)
    if standardize:
        X0 = apply_standardizer_np(X0, mean, std)
        X1 = apply_standardizer_np(X1, mean, std)
    return X0, X1

# --- 2) Transport in batches using your existing sampler ---
@torch.no_grad()
def transport_forward_batched(model, X_np, batch_size=256, num_steps=100, eps=1.0, device=torch.device("cpu")):
    """
    Plasma -> CSF via forward SDE (uses your existing sample_forward_SDE under the hood if available).
    If you already have sample_forward_SDE(model, x0, num_steps, eps), you can call it directly here.
    """
    model.eval()
    N, D = X_np.shape
    out = np.empty_like(X_np, dtype=np.float32)
    n_batches = (N + batch_size - 1) // batch_size
    for b in range(n_batches):
        sl = slice(b * batch_size, min((b + 1) * batch_size, N))
        x0 = torch.from_numpy(X_np[sl]).to(device, non_blocking=True)
        # Reuse your sampler
        hat_x1 = sample_forward_SDE(model, x0, num_steps=num_steps, eps=eps)
        out[sl] = hat_x1.detach().cpu().numpy().astype(np.float32)
    return out

# --- 3) Metrics: MSE and R^2 along chosen axis ---
def mse_along_axis(y_true, y_pred, axis=0):
    """
    axis=0 -> per-feature across individuals (shape [D])
    axis=1 -> per-sample across features (shape [N])
    """
    diff = y_pred - y_true
    return np.mean(diff * diff, axis=axis)

def r2_along_axis(y_true, y_pred, axis=0, eps=1e-12):
    """
    R^2 = 1 - SSE/SST computed along axis.
    axis=0 -> per-feature R^2 across individuals.
    axis=1 -> per-sample R^2 across features.
    Handles zero-variance targets by returning NaN for those entries.
    """
    y_true_mean = np.mean(y_true, axis=axis, keepdims=True)
    sse = np.sum((y_pred - y_true) ** 2, axis=axis)
    sst = np.sum((y_true - y_true_mean) ** 2, axis=axis)
    r2 = 1.0 - (sse / (sst + eps))
    # Mark truly zero-variance targets as NaN (uninformative)
    zero_var = sst < eps
    if axis == 0:
        r2[zero_var] = np.nan
    else:
        r2 = np.where(zero_var, np.nan, r2)
    return r2

# --- 4) End-to-end evaluation on df_test ---
def evaluate_transport_on_test(
    model,
    df_test,
    mean,
    std,
    device,
    D,
    batch_size=256,
    num_steps=100,
    eps=0.1,
    compute_on_original_scale=False
):
    """
    Returns a dict with predictions and metrics on BOTH standardized scale (always)
    and original scale (optional).
    """
    # Prepare standardized arrays
    X0_std, X1_std = prepare_test_arrays(df_test, D, mean, std, standardize=True)
    # Transport
    hat_X1_std = transport_forward_batched(model, X0_std, batch_size=batch_size, num_steps=num_steps, eps=eps, device=device)

    # Metrics on standardized scale
    per_feat_mse_std = mse_along_axis(X1_std, hat_X1_std, axis=0)   # [D]
    per_feat_r2_std  = r2_along_axis(X1_std, hat_X1_std, axis=0)    # [D]
    per_samp_mse_std = mse_along_axis(X1_std, hat_X1_std, axis=1)   # [N]
    per_samp_r2_std  = r2_along_axis(X1_std, hat_X1_std, axis=1)    # [N]

    out = {
        "hat_X1_std": hat_X1_std,
        "X1_std": X1_std,
        "X0_std": X0_std,
        "per_feature": {
            "mse_std": per_feat_mse_std,
            "r2_std":  per_feat_r2_std,
        },
        "per_sample": {
            "mse_std": per_samp_mse_std,
            "r2_std":  per_samp_r2_std,
        },
        "summary_std": {
            "mse_mean": float(np.mean(per_samp_mse_std)),
            "r2_feature_mean": float(np.nanmean(per_feat_r2_std)),
            "r2_sample_mean":  float(np.nanmean(per_samp_r2_std)),
        }
    }

    if compute_on_original_scale:
        # De-standardize for interpretability on raw units
        X1 = invert_standardizer_np(X1_std, mean, std)
        hat_X1 = invert_standardizer_np(hat_X1_std, mean, std)
        per_feat_mse = mse_along_axis(X1, hat_X1, axis=0)
        per_feat_r2  = r2_along_axis(X1, hat_X1, axis=0)
        per_samp_mse = mse_along_axis(X1, hat_X1, axis=1)
        per_samp_r2  = r2_along_axis(X1, hat_X1, axis=1)
        out.update({
            "hat_X1": hat_X1,
            "X1": X1,
            "per_feature_raw": {
                "mse": per_feat_mse,
                "r2":  per_feat_r2,
            },
            "per_sample_raw": {
                "mse": per_samp_mse,
                "r2":  per_samp_r2,
            },
            "summary_raw": {
                "mse_mean": float(np.mean(per_samp_mse)),
                "r2_feature_mean": float(np.nanmean(per_feat_r2)),
                "r2_sample_mean":  float(np.nanmean(per_samp_r2)),
            }
        })

    return out

# --- 5) Visualization utilities ---
def plot_metric_histograms(per_feature_mse, per_feature_r2, per_sample_mse, per_sample_r2, title_suffix="(standardized)"):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    axs[0].hist(per_feature_mse, bins=60, alpha=0.8)
    axs[0].set_title("Per-feature MSE " + title_suffix)
    axs[0].set_xlabel("MSE"); axs[0].set_ylabel("# proteins"); axs[0].grid(True, ls="--", alpha=0.3)

    axs[1].hist(per_feature_r2[~np.isnan(per_feature_r2)], bins=60, alpha=0.8)
    axs[1].set_title("Per-feature R² " + title_suffix)
    axs[1].set_xlabel("R²"); axs[1].set_ylabel("# proteins"); axs[1].grid(True, ls="--", alpha=0.3)

    axs[2].hist(per_sample_mse, bins=60, alpha=0.8)
    axs[2].set_title("Per-sample MSE " + title_suffix)
    axs[2].set_xlabel("MSE"); axs[2].set_ylabel("# individuals"); axs[2].grid(True, ls="--", alpha=0.3)

    axs[3].hist(per_sample_r2[~np.isnan(per_sample_r2)], bins=60, alpha=0.8)
    axs[3].set_title("Per-sample R² " + title_suffix)
    axs[3].set_xlabel("R²"); axs[3].set_ylabel("# individuals"); axs[3].grid(True, ls="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_topk_feature_parity_scatter(X_true, X_pred, r2_per_feature, feature_names=None, k=6, max_points=3000, seed=0, title_prefix="Top-k feature parity"):
    """
    For the top-k proteins by R², plot y_true vs y_pred across individuals.
    """
    rng = np.random.default_rng(seed)
    valid = np.where(~np.isnan(r2_per_feature))[0]
    if valid.size == 0:
        print("No valid features for parity scatter.")
        return
    order = valid[np.argsort(-r2_per_feature[valid])]
    top_idx = order[:min(k, order.size)]
    ncols = min(3, len(top_idx))
    nrows = int(np.ceil(len(top_idx) / ncols))
    plt.figure(figsize=(5*ncols, 4*nrows))
    for i, j in enumerate(top_idx):
        yt = X_true[:, j]
        yp = X_pred[:, j]
        if yt.shape[0] > max_points:
            sub = rng.choice(yt.shape[0], size=max_points, replace=False)
            yt, yp = yt[sub], yp[sub]
        ax = plt.subplot(nrows, ncols, i+1)
        ax.scatter(yt, yp, s=8, alpha=0.5)
        ax.plot([yt.min(), yt.max()], [yt.min(), yt.max()], 'k--', lw=1)
        fname = feature_names[j] if (feature_names is not None) else f"Protein {j}"
        ax.set_title(f"{fname} | R²={r2_per_feature[j]:.3f}")
        ax.set_xlabel("True CSF"); ax.set_ylabel("Pred CSF"); ax.grid(True, ls="--", alpha=0.3)
    plt.suptitle(title_prefix)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_sample_parity_scatter(X_true, X_pred, sample_indices=None, max_points=5000, seed=0, title_prefix="Per-sample parity across proteins"):
    """
    For selected individuals, plot true vs predicted across ALL proteins (subsampled for readability).
    """
    rng = np.random.default_rng(seed)
    N, D = X_true.shape
    if sample_indices is None:
        sample_indices = [0]
    ncols = min(3, len(sample_indices))
    nrows = int(np.ceil(len(sample_indices) / ncols))
    plt.figure(figsize=(5*ncols, 4*nrows))
    for i, idx in enumerate(sample_indices[:ncols*nrows]):
        yt = X_true[idx]
        yp = X_pred[idx]
        if D > max_points:
            sub = rng.choice(D, size=max_points, replace=False)
            yt, yp = yt[sub], yp[sub]
        ax = plt.subplot(nrows, ncols, i+1)
        ax.scatter(yt, yp, s=6, alpha=0.5)
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_title(f"Sample {idx}")
        ax.set_xlabel("True CSF"); ax.set_ylabel("Pred CSF"); ax.grid(True, ls="--", alpha=0.3)
    plt.suptitle(title_prefix)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def ranked_feature_table(r2_per_feature, feature_names=None, top=20):
    """
    Utility: print top/bottom features by R² for quick inspection.
    """
    valid = np.where(~np.isnan(r2_per_feature))[0]
    if valid.size == 0:
        print("No valid features.")
        return
    order = valid[np.argsort(-r2_per_feature[valid])]
    top_idx = order[:min(top, order.size)]
    print("\nTop features by R²:")
    for j in top_idx:
        name = feature_names[j] if (feature_names is not None) else f"Protein {j}"
        print(f"{name:>24s}  R²={r2_per_feature[j]:.4f}")
    bot_idx = valid[np.argsort(r2_per_feature[valid])][:min(top, valid.size)]
    print("\nBottom features by R² (excluding NaN):")
    for j in bot_idx:
        name = feature_names[j] if (feature_names is not None) else f"Protein {j}"
        print(f"{name:>24s}  R²={r2_per_feature[j]:.4f}")


# Assuming:
# - ema_model (or model) is trained
# - df_test is your held-out paired DataFrame
# - mean, std are the training standardization stats (shape [D,])
# - device, eps, and num_steps are the ones you used during training/eval
D = df_test.shape[1] // 2
results = evaluate_transport_on_test(
    ema_model, df_test, mean, std, device,
    D=D, batch_size=256, num_steps=100, eps=eps,
    compute_on_original_scale=False  # set True if you want raw-unit metrics too
)

# Summaries (standardized scale)
print("=== Standardized-scale summary ===")
print(results["summary_std"])
per_feat_mse = results["per_feature"]["mse_std"]
per_feat_r2  = results["per_feature"]["r2_std"]
per_samp_mse = results["per_sample"]["mse_std"]
per_samp_r2  = results["per_sample"]["r2_std"]

# Histograms
plot_metric_histograms(per_feat_mse, per_feat_r2, per_samp_mse, per_samp_r2, title_suffix="(standardized)")

# Parity plots for top-k proteins by R²
feature_names = list(df_test.columns[:D])  # optional, if your columns are protein names
plot_topk_feature_parity_scatter(
    results["X1_std"], results["hat_X1_std"], per_feat_r2,
    feature_names=feature_names, k=6, max_points=3000,
    title_prefix="Top-6 proteins parity (standardized)"
)

# Parity across proteins for selected individuals
plot_sample_parity_scatter(
    results["X1_std"], results["hat_X1_std"],
    sample_indices=[0, 1, 2], max_points=5000,
    title_prefix="Sample-level parity across proteins (standardized)"
)

# Optional: print ranked features by R²
ranked_feature_table(per_feat_r2, feature_names=feature_names, top=15)