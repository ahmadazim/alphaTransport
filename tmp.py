import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
import torch

# 0) Helpers for standardization (same as earlier)
def compute_feature_standardizer(df_plasma, df_csf, eps=1e-6):
    stack = np.vstack([df_plasma.values, df_csf.values]).astype(np.float32)
    mean = stack.mean(axis=0)
    std  = stack.std(axis=0)
    std[std < eps] = 1.0
    return mean, std

def apply_standardizer_np(arr, mean, std):
    return ((arr.astype(np.float32) - mean) / std).astype(np.float32)

# 1) Per-protein logistic screening on paired df_ov
def select_proteins_logistic(df_ov, D, alpha_fdr=0.05, min_keep=256, seed=0):
    """
    For each protein j, binarize CSF_j by median and fit Logit(y_j ~ Plasma_j).
    Return selected indices by FDR on slope p-values (two-sided).
    """
    rng = np.random.default_rng(seed)
    X0 = df_ov.iloc[:, :D].values.astype(np.float32)  # plasma
    X1 = df_ov.iloc[:,  D:].values.astype(np.float32) # csf

    # Standardize both sides (per-protein z-score) for stable fits
    mu = np.mean(np.vstack([X0, X1]), axis=0)
    sd = np.std(np.vstack([X0, X1]), axis=0); sd[sd < 1e-6] = 1.0
    Z0 = (X0 - mu) / sd
    Z1 = (X1 - mu) / sd

    pvals = np.ones(D, dtype=np.float64)
    betas = np.zeros(D, dtype=np.float64)

    for j in range(D):
        y = (Z1[:, j] > np.median(Z1[:, j])).astype(np.float32)
        if y.mean() < 0.05 or y.mean() > 0.95:   # too imbalanced -> skip
            pvals[j] = 1.0
            continue
        x = sm.add_constant(Z0[:, j])
        try:
            fit = sm.Logit(y, x).fit(disp=0, maxiter=200, method='lbfgs')
            # slope is second coef; two-sided p-value from summary
            betas[j] = fit.params[1]
            pvals[j] = fit.pvalues[1]
        except Exception:
            pvals[j] = 1.0

    # FDR (BH)
    reject, qvals, _, _ = multipletests(pvals, alpha=alpha_fdr, method='fdr_bh')
    sel = np.where(reject)[0]

    # Fallback: ensure we keep at least min_keep by smallest p
    if sel.size < min_keep:
        order = np.argsort(pvals)
        sel = order[:min_keep]

    return sel, pvals, betas

# 2) Subset dfs to selected proteins
def subset_dfs(df_plasma, df_csf, df_ov, sel_idx):
    D = df_plasma.shape[1]
    df_plasma_sel = df_plasma.iloc[:, sel_idx]
    df_csf_sel    = df_csf.iloc[:, sel_idx]
    # df_ov: concat selected plasma cols and the corresponding CSF cols in the second half
    df_ov_sel = pd.concat(
        [df_ov.iloc[:, sel_idx], df_ov.iloc[:, D + np.array(sel_idx)]],
        axis=1
    )
    return df_plasma_sel, df_csf_sel, df_ov_sel

# 3) End-to-end: select -> subset -> standardize -> loaders -> run SB
def run_sb_with_selected_features(
    df_plasma, df_csf, df_ov,
    model_ctor,          # callable: lambda x_dim: model instance (uses your drift arch)
    pretrain_steps, finetune_steps,
    batch_size, eps, ema_decay, alpha_step, lr_pre=1e-3, lr_fine=1e-4,
    alpha_fdr=0.05, min_keep=256, device=torch.device("cpu"),
    save_freq=1000, sde_steps=100
):
    # (a) select proteins
    D_full = df_plasma.shape[1]
    sel_idx, pvals, betas = select_proteins_logistic(df_ov, D_full, alpha_fdr=alpha_fdr, min_keep=min_keep)
    print(f"[FeatureSelect] Selected {sel_idx.size} / {D_full} proteins")

    # (b) subset dataframes
    df_plasma_sel, df_csf_sel, df_ov_sel = subset_dfs(df_plasma, df_csf, df_ov, sel_idx)
    D = df_plasma_sel.shape[1]

    # (c) standardize on union (selected features only)
    mean, std = compute_feature_standardizer(df_plasma_sel, df_csf_sel)
    X_plasma_np = apply_standardizer_np(df_plasma_sel.values, mean, std)
    X_csf_np    = apply_standardizer_np(df_csf_sel.values,    mean, std)
    X_ov0_np    = apply_standardizer_np(df_ov_sel.iloc[:, :D].values, mean, std)
    X_ov1_np    = apply_standardizer_np(df_ov_sel.iloc[:,  D:].values, mean, std)

    # (d) loaders (uses your existing helpers)
    pin = (device.type == "cuda")
    dl_pairs  = build_paired_loader_from_np(X_ov0_np, X_ov1_np, batch_size=batch_size, shuffle=True, pin=pin)
    dl_plasma = build_loader_from_np(X_plasma_np,     batch_size=batch_size, shuffle=True, pin=pin)
    dl_csf    = build_loader_from_np(X_csf_np,        batch_size=batch_size, shuffle=True, pin=pin)

    # (e) model/init (keep your EMA + same DSBM functions)
    model = model_ctor(D).to(device)
    ema_model = model_ctor(D).to(device)
    ema_model.load_state_dict(model.state_dict())

    # (f) run pretraining + finetuning (unchanged)
    print("----- Pretraining (paired, selected proteins) -----")
    _ = pretraining_stage_paired(
        model, ema_model, dl_pairs,
        num_steps=pretrain_steps, eps=eps, lr=lr_pre, ema_decay=ema_decay
    )

    print("----- Finetuning (unpaired, selected proteins) -----")
    _, saved_models = finetuning_stage(
        model, ema_model, dl_plasma, dl_csf,
        num_finetune_steps=finetune_steps, eps=eps, lr=lr_fine,
        alpha_step=alpha_step, ema_decay=ema_decay, save_freq=save_freq
    )

    out = {
        "sel_idx": sel_idx,
        "pvals": pvals,
        "betas": betas,
        "mean": mean,
        "std": std,
        "model": model,
        "ema_model": ema_model,
        "saved_models": saved_models,
        "df_plasma_sel": df_plasma_sel,
        "df_csf_sel": df_csf_sel,
        "df_ov_sel": df_ov_sel,
    }
    return out







result = run_sb_with_selected_features(
    df_plasma=df_plasma,
    df_csf=df_csf,
    df_ov=df_ov,
    model_ctor=make_basic_drift,          # or make_gated_drift
    pretrain_steps=pretrain_steps,
    finetune_steps=finetune_steps,
    batch_size=batch_size,
    eps=eps,
    ema_decay=ema_decay,
    alpha_step=alpha_step,
    lr_pre=1e-3, lr_fine=1e-4,
    alpha_fdr=0.05, min_keep=256,
    device=device,
    save_freq=1000
)

print(f"Kept {len(result['sel_idx'])} proteins.")