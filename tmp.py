# ---- NEW: High-capacity feature-gated, FiLM-conditioned residual MLP drift ----
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# (Optional) Low-rank Linear to reduce params for huge D
class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        if rank is None or rank >= min(in_features, out_features):
            self.A = None
            self.core = nn.Linear(in_features, out_features, bias=bias)
        else:
            self.A = nn.Linear(in_features, rank, bias=False)
            self.B = nn.Linear(rank, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        if self.A is None:
            # use default init
            return
        # Xavier for low-rank pieces
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)
        if self.B.bias is not None:
            nn.init.zeros_(self.B.bias)

    def forward(self, x):
        if self.A is None:
            return self.core(x)
        return self.B(self.A(x))

# SwiGLU FFN block
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w12 = nn.Linear(d_model, d_ff * 2)
        self.w3  = nn.Linear(d_ff, d_model)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)  # [B, d_ff] x2
        return self.w3(F.silu(a) * b)

class FiLMResBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, gamma, beta):
        """
        gamma, beta: [B, d_model] FiLM parameters
        """
        h = self.ln(x)
        h = self.ff(h)
        # FiLM on the block output
        h = h * (1.0 + gamma) + beta
        h = self.drop(h)
        return x + h

def fourier_time_embedding(t, n_f=16):
    """
    t: [B,1] in [0,1]
    returns [B, 2*n_f] with sin/cos features
    """
    device = t.device
    freqs = torch.linspace(1.0, 2.0**(n_f-1), n_f, device=device) * (2*math.pi)
    ang = t * freqs  # [B, n_f]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # [B, 2*n_f]

class GatedFiLMResNetDrift(nn.Module):
    """
    High-capacity drift network for high-D proteomics.
    - Per-feature gates on x with L1 regularization
    - Sinusoidal time embedding + direction embedding
    - FiLM-conditioned residual MLP blocks with SwiGLU
    - Optional low-rank input/output to keep params manageable
    """
    def __init__(
        self,
        x_dim: int,            # number of proteins (or latent dims if training in PCA space)
        d_model: int = 512,
        n_blocks: int = 6,
        ff_mult: int = 4,
        dropout: float = 0.1,
        time_fourier: int = 16,
        dir_emb_dim: int = 8,
        in_rank: int | None = None,   # e.g., 256 to reduce params from x_dim->d_model
        out_rank: int | None = None   # e.g., 256 for d_model->x_dim
    ):
        super().__init__()
        self.x_dim = x_dim

        # --- Feature gates (sigmoid(g_logits) in (0,1)) ---
        init_gate = 0.5
        init_logit = math.log(init_gate / (1.0 - init_gate))
        self.g_logits = nn.Parameter(torch.full((x_dim,), float(init_logit)))

        # --- Input projection (optionally low-rank) ---
        self.x_in = LowRankLinear(x_dim, d_model, rank=in_rank, bias=True)

        # --- Direction embedding (two categories: 0 backward, 1 forward) ---
        self.s_emb = nn.Embedding(2, dir_emb_dim)

        # --- Conditioning network to produce FiLM params for each block ---
        cond_dim = 2 * time_fourier + dir_emb_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # one gamma & beta per block, each of size d_model
        self.to_film = nn.Linear(d_model, n_blocks * 2 * d_model)

        # --- Residual stack ---
        d_ff = ff_mult * d_model
        self.blocks = nn.ModuleList([FiLMResBlock(d_model, d_ff, dropout=dropout) for _ in range(n_blocks)])

        # --- Output head (optionally low-rank) ---
        self.out_ln = nn.LayerNorm(d_model)
        self.out = LowRankLinear(d_model, x_dim, rank=out_rank, bias=True)

        # Scale residuals a bit for stability
        self.register_buffer("res_scale", torch.tensor(1.0 / math.sqrt(n_blocks)))

    def gate_l1(self):
        # L1 on gate magnitudes (post-sigmoid). Encourages sparsity.
        return torch.sigmoid(self.g_logits).abs().sum()

    def forward(self, s, t, x):
        """
        s: [B,1] (float 0/1)    direction
        t: [B,1]                time
        x: [B, x_dim]           input features
        Returns: drift [B, x_dim]
        """
        if t.dim() == 1: t = t.unsqueeze(1)
        # Apply feature gates
        g = torch.sigmoid(self.g_logits)              # [x_dim]
        xg = x * g                                     # [B, x_dim]

        # Project to model dim
        h = self.x_in(xg)                              # [B, d_model]

        # Build conditioning (time fourier + direction embedding)
        t_feat = fourier_time_embedding(t, n_f=self.cond_mlp[0].in_features // 2)  # but we set time_fourier, not used here
        # fix: compute t_feat with the chosen time_fourier explicitly
                # Recompute t_feat correctly with explicit time_fourier
        # (work around static access to constructor arg)
        # We'll infer n_f from cond input size: cond_mlp first layer in_features = 2*time_fourier + dir_emb_dim
        cond_in = self.cond_mlp[0].in_features
        # We stored dir_emb_dim as s_emb.embedding_dim; so:
        dir_dim = self.s_emb.embedding_dim
        n_f = (cond_in - dir_dim) // 2
        t_feat = fourier_time_embedding(t, n_f=n_f)    # [B, 2*n_f]

        s_idx = s.round().long().clamp(0, 1).view(-1)  # [B]
        s_feat = self.s_emb(s_idx)                     # [B, dir_emb_dim]

        cond = torch.cat([t_feat, s_feat], dim=1)      # [B, 2*n_f + dir_emb_dim]
        c = self.cond_mlp(cond)                        # [B, d_model]
        film_all = self.to_film(c)                     # [B, n_blocks*2*d_model]
        B, _ = film_all.shape
        # reshape to per-block gamma/beta
        n_blocks = len(self.blocks)
        d_model = self.blocks[0].ln.normalized_shape[0]
        film_all = film_all.view(B, n_blocks, 2, d_model)
        gammas = film_all[:, :, 0, :]                  # [B, n_blocks, d_model]
        betas  = film_all[:, :, 1, :]                  # [B, n_blocks, d_model]

        # Residual stack with FiLM
        for i, blk in enumerate(self.blocks):
            h = h + self.res_scale * blk(h, gammas[:, i, :], betas[:, i, :])

        # Output head
        h = self.out_ln(h)
        drift = self.out(h)                            # [B, x_dim]
        return drift
# ---- END NEW ----------------------------------------------------------------


D = X_plasma_np.shape[1]  # 7290
model = GatedFiLMResNetDrift(
    x_dim=D,
    d_model=512,
    n_blocks=8,
    ff_mult=4,
    dropout=0.1,
    time_fourier=16,
    dir_emb_dim=8,
    in_rank=256,   # try None if you want full-rank; 256 keeps params lighter
    out_rank=256,
).to(device)

ema_model = GatedFiLMResNetDrift(
    x_dim=D,
    d_model=512,
    n_blocks=8,
    ff_mult=4,
    dropout=0.1,
    time_fourier=16,
    dir_emb_dim=8,
    in_rank=256,
    out_rank=256,
).to(device)
ema_model.load_state_dict(model.state_dict())
