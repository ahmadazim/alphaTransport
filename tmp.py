# --- NEW: PCA + batched transport + metrics/plots ----------------------------
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---------- PCA without sklearn (stable SVD) ----------
def pca_fit(X_list, n_components=10, center=True):
    """
    Fit PCA on the vertical concatenation of arrays in X_list (each shape [N_i, D]).
    Returns dict with 'mean', 'components', 'explained_var', and transform function.
    """
    X = np.vstack(X_list).astype(np.float32)
    mean = X.mean(axis=0) if center else np.zeros(X.shape[1], dtype=np.float32)
    Xc = X - mean
    # economical SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]                    # [n_components, D]
    # explained variances of each PC
    explained_var = (S**2) / (X.shape[0] - 1)
    explained_var = explained_var[:n_components].astype(np.float32)

    def pca_transform(X_in):
        return (X_in.astype(np.float32) - mean) @ components.T  # [N, n_components]

    return {
        "mean": mean.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_var": explained_var,
        "transform": pca_transform,
    }

# ---------- Batched forward transport using your drift/SDE ----------
@torch.no_grad()
def transport_forward_batched(model, X_np, batch_size=256, num_steps=100, eps=1.0, device=torch.device("cpu")):
    """
    Transport plasma -> CSF via forward SDE (EMA model recommended).
    X_np: (N, D) float32 numpy array (already standardized consistently with training)
    Returns: hat_X1_np of shape (N, D)
    """
    model.eval()
    N, D = X_np.shape
    out = np.empty_like(X_np, dtype=np.float32)
    n_batches = (N + batch_size - 1) // batch_size
    for b in range(n_batches):
        sl = slice(b * batch_size, min((b + 1) * batch_size, N))
        x0 = torch.from_numpy(X_np[sl]).to(device, non_blocking=True)
        # reuse your sampler
        hat_x1 = sample_forward_SDE(model, x0, num_steps=num_steps, eps=eps)  # [B, D] on device
        out[sl] = hat_x1.detach().cpu().numpy().astype(np.float32)
    return out

# ---------- Simple RBF MMD (with median heuristic) in a feature space ----------
def _pairwise_sq_dists(A, B):
    # A:[n,d], B:[m,d] -> [n,m]
    A2 = (A*A).sum(1, keepdims=True)
    B2 = (B*B).sum(1, keepdims=True).T
    return A2 + B2 - 2.0 * (A @ B.T)

def _rbf_kernel(A, B, sigma):
    d2 = _pairwise_sq_dists(A, B)
    return np.exp(-d2 / (2.0 * sigma**2))

def mmd_rbf(X, Y, sigma=None, max_samples=2000, seed=0):
    """
    Unbiased MMD^2 with RBF kernel on (optionally) subsampled data.
    """
    rng = np.random.default_rng(seed)
    n = min(max_samples, X.shape[0])
    m = min(max_samples, Y.shape[0])
    Xs = X[rng.choice(X.shape[0], n, replace=False)]
    Ys = Y[rng.choice(Y.shape[0], m, replace=False)]
    if sigma is None:
        # median heuristic on pooled pairwise distances (downsample again for speed)
        pool = np.vstack([Xs, Ys])
        idx = rng.choice(pool.shape[0], size=min(1000, pool.shape[0]), replace=False)
        Z = pool[idx]
        d2 = _pairwise_sq_dists(Z, Z)
        med = np.median(d2[np.triu_indices_from(d2, k=1)])
        sigma = np.sqrt(0.5 * med + 1e-8)

    Kxx = _rbf_kernel(Xs, Xs, sigma); np.fill_diagonal(Kxx, 0.0)
    Kyy = _rbf_kernel(Ys, Ys, sigma); np.fill_diagonal(Kyy, 0.0)
    Kxy = _rbf_kernel(Xs, Ys, sigma)
    mmd2 = Kxx.sum()/(n*(n-1)) + Kyy.sum()/(m*(m-1)) - 2.0*Kxy.mean()
    return float(mmd2), float(sigma)

# ---------- 1-NN two-sample test accuracy (in a feature space) ----------
def one_nn_two_sample_accuracy(X, Y, max_samples=2000, seed=0):
    """
    Concatenate X (label 0) and Y (label 1), predict each point's class
    using the label of its nearest neighbor among all other points.
    Accuracy -> closer to 0.5 indicates better alignment.
    """
    rng = np.random.default_rng(seed)
    n = min(max_samples, X.shape[0])
    m = min(max_samples, Y.shape[0])
    Xs = X[rng.choice(X.shape[0], n, replace=False)]
    Ys = Y[rng.choice(Y.shape[0], m, replace=False)]
    Z = np.vstack([Xs, Ys])
    labels = np.concatenate([np.zeros(n, dtype=int), np.ones(m, dtype=int)])
    d2 = _pairwise_sq_dists(Z, Z)
    np.fill_diagonal(d2, np.inf)
    nn_idx = d2.argmin(axis=1)
    pred = labels[nn_idx]
    acc = (pred == labels).mean()
    return float(acc)

# ---------- Paired evaluation on held-out matched subset ----------
@torch.no_grad()
def paired_eval_forward(model, X0_np, X1_np, num_steps=100, eps=1.0, batch_size=256, device=torch.device("cpu")):
    """
    Transport paired plasma X0 -> hat_X1, compare to true CSF X1 (both standardized).
    Returns dict of per-sample MSE and cosine similarity plus overall means.
    """
    hat_X1 = transport_forward_batched(model, X0_np, batch_size=batch_size, num_steps=num_steps, eps=eps, device=device)
    diff = hat_X1 - X1_np
    mse_per = (diff**2).mean(axis=1)
    # cosine
    num = (hat_X1 * X1_np).sum(axis=1)
    den = np.linalg.norm(hat_X1, axis=1) * np.linalg.norm(X1_np, axis=1) + 1e-8
    cos_per = num / den
    return {
        "hat_X1": hat_X1,
        "mse_per": mse_per,
        "cos_per": cos_per,
        "mse_mean": float(mse_per.mean()),
        "cos_mean": float(cos_per.mean()),
    }

# ---------- Plotting in PCA space ----------
def plot_pca_scatter_with_arrows(pca, X0, X1, X0_to_1, n_points_arrows=200, title="PCA Projection (PC1 vs PC2)", alpha_bg=0.15, s_bg=6, seed=0):
    """
    X0: plasma (N0,D), X1: csf (N1,D), X0_to_1: transported plasma (N0,D)
    Projects all to PCA(2) and plots scatter; draws arrows for a random subset of X0 -> X0_to_1.
    """
    rng = np.random.default_rng(seed)
    Z0 = pca["transform"](X0)[:, :2]
    Z1 = pca["transform"](X1)[:, :2]
    Zt = pca["transform"](X0_to_1)[:, :2]

    # pick subset for arrows
    n_ar = min(n_points_arrows, Z0.shape[0])
    idx = rng.choice(Z0.shape[0], n_ar, replace=False)

    plt.figure(figsize=(7,6))
    plt.scatter(Z0[:,0], Z0[:,1], s=s_bg, alpha=alpha_bg, label="Plasma (X0)", color="tab:red")
    plt.scatter(Z1[:,0], Z1[:,1], s=s_bg, alpha=alpha_bg, label="CSF (X1)", color="tab:blue")
    plt.scatter(Zt[:,0], Zt[:,1], s=s_bg, alpha=0.25, label="Transported (X0→ĤX1)", color="tab:green")

    # arrows
    for i in idx:
        plt.arrow(Z0[i,0], Z0[i,1], (Zt[i,0]-Z0[i,0]), (Zt[i,1]-Z0[i,1]),
                  head_width=0.04, length_includes_head=True, alpha=0.5, color="k")

    ev = pca["explained_var"]
    exp1 = 100.0 * ev[0] / ev.sum()
    exp2 = 100.0 * ev[1] / ev.sum()
    plt.xlabel(f"PC1 ({exp1:.1f}% var)")
    plt.ylabel(f"PC2 ({exp2:.1f}% var)")
    plt.title(title)
    plt.legend()
    plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_pca_density_comparison(pca, X1, X0_to_1, bins=80, title="PC1 Marginals: CSF vs Transported"):
    """
    Compare 1D marginals along PC1 (and optionally PC2) for CSF vs Transported.
    """
    Z1 = pca["transform"](X1)[:, :2]
    Zt = pca["transform"](X0_to_1)[:, :2]
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    ax[0].hist(Z1[:,0], bins=bins, alpha=0.5, density=True, label="CSF (PC1)")
    ax[0].hist(Zt[:,0], bins=bins, alpha=0.5, density=True, label="Transported (PC1)")
    ax[0].set_title("PC1 marginals"); ax[0].legend(); ax[0].grid(True, ls="--", alpha=0.3)

    ax[1].hist(Z1[:,1], bins=bins, alpha=0.5, density=True, label="CSF (PC2)")
    ax[1].hist(Zt[:,1], bins=bins, alpha=0.5, density=True, label="Transported (PC2)")
    ax[1].set_title("PC2 marginals"); ax[1].legend(); ax[1].grid(True, ls="--", alpha=0.3)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

# ---------- Optional: show a few trajectory paths in PCA(2) ----------
@torch.no_grad()
def plot_forward_paths_in_pca(model, X0_np, pca, num_traj=8, num_steps=40, eps=1.0, device=torch.device("cpu"), seed=0):
    """
    Simulate deterministic Euler (no noise) paths for a small subset and plot in PCA(2).
    Useful for sanity-checking directionality in the low-d projection.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(X0_np.shape[0], size=min(num_traj, X0_np.shape[0]), replace=False)
    X0_subset = torch.from_numpy(X0_np[idx]).to(device)
    ts = torch.linspace(0.0, 1.0, num_steps, device=device).view(-1, 1)
    dt = 1.0 / (num_steps - 1)
    s_val = torch.ones(X0_subset.shape[0], 1, device=device)

    # collect trajectory points for each sample
    paths = [[] for _ in range(X0_subset.shape[0])]
    x = X0_subset.clone()
    for k, t in enumerate(ts):
        X_cpu = x.detach().cpu().numpy()
        Z = pca["transform"](X_cpu)[:, :2]
        for j in range(Z.shape[0]):
            paths[j].append(Z[j])
        if k < num_steps - 1:
            drift = model(s_val, t.expand(x.shape[0], 1), x)
            x = x + drift * dt  # deterministic preview (no noise)

    # plot
    plt.figure(figsize=(7,6))
    for j in range(len(paths)):
        P = np.stack(paths[j], axis=0)
        plt.plot(P[:,0], P[:,1], "--", alpha=0.8)
        plt.scatter(P[0,0], P[0,1], s=30, color="tab:red", alpha=0.7)
        plt.scatter(P[-1,0], P[-1,1], s=30, color="tab:green", alpha=0.7)
    plt.title("Forward Euler paths (PC1 vs PC2)")
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True, ls="--", alpha=0.3)
    plt.tight_layout(); plt.show()
# --- END NEW -----------------------------------------------------------------