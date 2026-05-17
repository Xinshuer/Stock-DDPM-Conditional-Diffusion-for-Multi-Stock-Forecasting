import math
import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────
# β schedule
# ──────────────────────────────────────────────
def quadratic_beta_schedule(beta_start: float = 1e-4, beta_end: float = 0.5, n_steps: int = 50):
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_steps) ** 2


# ──────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────
class EMA:
    """Shadow-copy parameter EMA. Apply/restore on sampling."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self.backup = None

    @torch.no_grad()
    def update(self, model: nn.Module):
        sd = model.state_dict()
        for k in self.shadow:
            self.shadow[k].mul_(self.decay).add_(sd[k].detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module):
        sd = model.state_dict()
        self.backup = {k: sd[k].clone() for k in self.shadow}
        merged = {**sd, **self.shadow}
        model.load_state_dict(merged, strict=False)

    def restore(self, model: nn.Module):
        assert self.backup is not None
        sd = model.state_dict()
        merged = {**sd, **self.backup}
        model.load_state_dict(merged, strict=False)
        self.backup = None


# ──────────────────────────────────────────────
# Core DDPM
# ──────────────────────────────────────────────
class StockDDPM:
    def __init__(
        self,
        model: nn.Module,
        n_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.5,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.n_steps = n_steps
        self.device = device

        self.beta = quadratic_beta_schedule(beta_start, beta_end, n_steps).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    # ─── Forward diffusion ─────────────────────
    def forward_diffusion(self, y0: torch.Tensor, t: torch.Tensor):
        ab = self.alpha_bar[t]
        sqrt_ab = ab.sqrt()[:, None, None]
        sqrt_one_minus_ab = (1.0 - ab).sqrt()[:, None, None]
        noise = torch.randn_like(y0)
        y_t = sqrt_ab * y0 + sqrt_one_minus_ab * noise
        return y_t, noise

    # ─── DDPM sampling ─────────────────────────
    @torch.no_grad()
    def sample_ddpm(self, x_cond: torch.Tensor, n_samples: int = 100, clip_value: float = 10.0):
        """
        Standard DDPM reverse sampling with optional numerical clamp.
        Target data is Z-score normalized (unit variance), so |y_0| > 10 is almost
        certainly a divergence artefact from an under-trained model.
        """
        self.model.eval()
        D = x_cond.shape[1]
        pred_len = self.model.pred_len

        if x_cond.shape[0] == 1:
            x_cond = x_cond.expand(n_samples, -1, -1)

        y_t = torch.randn(n_samples, D, pred_len, device=self.device)

        for i in reversed(range(self.n_steps)):
            t_batch = torch.full((n_samples,), i, device=self.device, dtype=torch.long)
            eps = self.model(y_t, t_batch, x_cond)

            alpha_t = self.alpha[i]
            alpha_bar_t = self.alpha_bar[i]
            beta_t = self.beta[i]

            coeff = beta_t / (1.0 - alpha_bar_t).sqrt()
            mean = (y_t - coeff * eps) / alpha_t.sqrt()

            if i > 0:
                z = torch.randn_like(y_t)
                y_t = mean + beta_t.sqrt() * z
            else:
                y_t = mean

            if clip_value is not None:
                y_t = y_t.clamp(min=-clip_value, max=clip_value)

        return y_t

    # ─── DDIM sampling ─────────────────────────
    @torch.no_grad()
    def sample_ddim(
        self,
        x_cond: torch.Tensor,
        n_samples: int = 100,
        n_ddim_steps: int = 10,
        eta: float = 0.0,
    ):
        self.model.eval()
        D = x_cond.shape[1]
        pred_len = self.model.pred_len

        if x_cond.shape[0] == 1:
            x_cond = x_cond.expand(n_samples, -1, -1)

        step_idx = torch.linspace(0, self.n_steps - 1, n_ddim_steps, dtype=torch.long).tolist()

        y = torch.randn(n_samples, D, pred_len, device=self.device)

        for i in reversed(range(len(step_idx))):
            t = step_idx[i]
            t_prev = step_idx[i - 1] if i > 0 else -1

            ab_t = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=self.device)

            t_batch = torch.full((n_samples,), t, device=self.device, dtype=torch.long)
            eps = self.model(y, t_batch, x_cond)

            y0_hat = (y - (1.0 - ab_t).sqrt() * eps) / ab_t.sqrt()

            sigma = eta * (((1.0 - ab_prev) / (1.0 - ab_t)) * (1.0 - ab_t / ab_prev)).clamp(min=0).sqrt() \
                    if t_prev >= 0 else torch.tensor(0.0, device=self.device)

            dir_term = (1.0 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            noise = sigma * torch.randn_like(y) if t_prev >= 0 else 0.0

            y = ab_prev.sqrt() * y0_hat + dir_term + noise

        return y


# ──────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────
def save_checkpoint(path, ddpm, optimizer, scheduler, ema, history, epoch):
    """Save full training state for resumption or sampling."""
    state = {
        "epoch": epoch,
        "model_state": ddpm.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "ema_shadow": ema.shadow if ema is not None else None,
        "ema_decay": ema.decay if ema is not None else None,
        "history": history,
        "n_steps": ddpm.n_steps,
    }
    torch.save(state, path)


def load_checkpoint(path, ddpm, optimizer=None, scheduler=None, ema=None):
    """Load state into model / optimizer / scheduler / EMA (any None is skipped).
    Returns (start_epoch, history)."""
    state = torch.load(path, map_location=ddpm.device, weights_only=False)
    ddpm.model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state"):
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state"):
        scheduler.load_state_dict(state["scheduler_state"])
    if ema is not None and state.get("ema_shadow") is not None:
        ema.shadow = {k: v.to(ddpm.device) for k, v in state["ema_shadow"].items()}
    return state["epoch"], state.get("history", {"train_loss": [], "val_crps": [], "val_crps_sum": []})


def train(
    ddpm: StockDDPM,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-6,
    lr_milestones=(150, 180),
    lr_gamma: float = 0.1,
    grad_clip: float = 1.0,
    use_ema: bool = True,
    ema_decay: float = 0.999,
    log_every_epoch: int = 5,
    val_every_epoch: int = 20,
    val_n_samples: int = 50,
    ckpt_dir: str = None,
    ckpt_every_epoch: int = 20,
    resume_from: str = None,
):
    from crps import crps as crps_fn, crps_sum as crps_sum_fn

    optimizer = torch.optim.Adam(
        ddpm.model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(lr_milestones), gamma=lr_gamma
    )
    loss_fn = nn.MSELoss()
    ema = EMA(ddpm.model, decay=ema_decay) if use_ema else None

    device = ddpm.device
    history = {"train_loss": [], "val_loss": [], "val_crps": [], "val_crps_sum": []}

    start_epoch = 1
    if resume_from is not None and os.path.isfile(resume_from):
        start_epoch, history = load_checkpoint(resume_from, ddpm, optimizer, scheduler, ema)
        start_epoch += 1
        print(f"Resumed from {resume_from} at epoch {start_epoch}")

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(start_epoch, epochs + 1):
        ddpm.model.train()
        total_loss = 0.0
        n_batches = 0
        grad_norm_sum = 0.0

        for x_cond, y0, _, _ in train_loader:
            x_cond = x_cond.to(device)
            y0 = y0.to(device)
            B = x_cond.shape[0]

            t = torch.randint(0, ddpm.n_steps, (B,), device=device)
            y_t, noise = ddpm.forward_diffusion(y0, t)
            eps_pred = ddpm.model(y_t, t, x_cond)

            loss = loss_fn(eps_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(ddpm.model.parameters(), grad_clip)
            optimizer.step()

            if ema is not None:
                ema.update(ddpm.model)

            total_loss += loss.item()
            grad_norm_sum += float(gn)
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_gn = grad_norm_sum / n_batches
        history["train_loss"].append(avg_loss)

        # Per-epoch val MSE (cheap: one forward pass per val batch, no sampling)
        if val_loader is not None:
            ddpm.model.eval()
            with torch.no_grad():
                v_total, v_n = 0.0, 0
                for x_cond, y0, _, _ in val_loader:
                    x_cond = x_cond.to(device)
                    y0 = y0.to(device)
                    Bv = y0.shape[0]
                    tv = torch.randint(0, ddpm.n_steps, (Bv,), device=device)
                    y_t_v, noise_v = ddpm.forward_diffusion(y0, tv)
                    eps_v = ddpm.model(y_t_v, tv, x_cond)
                    v_total += loss_fn(eps_v, noise_v).item() * Bv
                    v_n += Bv
            val_loss = v_total / max(v_n, 1)
            history["val_loss"].append(val_loss)
        else:
            val_loss = None

        if epoch % log_every_epoch == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            vstr = f"  val_loss={val_loss:.5f}" if val_loss is not None else ""
            print(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.5f}{vstr}  |grad|={avg_gn:.2f}  lr={lr_now:.1e}")

        # Validation CRPS
        if val_loader is not None and (epoch % val_every_epoch == 0 or epoch == epochs):
            val_crps, val_crps_sum = evaluate_crps(
                ddpm, val_loader, n_samples=val_n_samples, ema=ema
            )
            history["val_crps"].append((epoch, val_crps))
            history["val_crps_sum"].append((epoch, val_crps_sum))
            print(f"             val_CRPS={val_crps:.4f}  val_CRPS_sum={val_crps_sum:.4f}")

        # Checkpoint
        if ckpt_dir is not None and (epoch % ckpt_every_epoch == 0 or epoch == epochs):
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch:04d}.pt")
            save_checkpoint(ckpt_path, ddpm, optimizer, scheduler, ema, history, epoch)
            latest = os.path.join(ckpt_dir, "latest.pt")
            save_checkpoint(latest, ddpm, optimizer, scheduler, ema, history, epoch)
            print(f"             saved checkpoint: {ckpt_path}")

    return history, ema


# ──────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────
@torch.no_grad()
def evaluate_crps(
    ddpm: StockDDPM,
    loader: DataLoader,
    n_samples: int = 50,
    sampler: str = "ddpm",
    n_ddim_steps: int = 10,
    ema: EMA = None,
    max_batches: int = None,
):
    """
    Memory-bounded CRPS evaluation.
    For each batch of B cond-windows, loop n_samples times calling sample_ddpm/ddim
    with batch size B (one forecast per cond per call). Peak GPU batch = B, never B*n_samples.
    """
    from crps import crps as crps_fn, crps_sum as crps_sum_fn

    if ema is not None:
        ema.apply_to(ddpm.model)
    ddpm.model.eval()

    all_targets = []
    all_forecasts = []

    for b_idx, (x_cond, y0, _, _) in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        x_cond = x_cond.to(ddpm.device)
        y0 = y0.to(ddpm.device)
        B, D, L = y0.shape

        # One-forecast-per-cond, repeated n_samples times (memory bounded to B)
        forecasts = []
        for _ in range(n_samples):
            if sampler == "ddpm":
                fc_one = ddpm.sample_ddpm(x_cond, n_samples=B)
            elif sampler == "ddim":
                fc_one = ddpm.sample_ddim(x_cond, n_samples=B, n_ddim_steps=n_ddim_steps)
            else:
                raise ValueError(sampler)
            forecasts.append(fc_one.cpu())

        # (n_samples, B, D, L)
        fc = torch.stack(forecasts, dim=0)

        all_targets.append(y0.cpu())
        all_forecasts.append(fc)

    target = torch.cat(all_targets, dim=0)
    forecast = torch.cat(all_forecasts, dim=1)

    # Diagnostic stats (helps detect sampling divergence / CRPS anomalies)
    tgt_abs_mean = float(target.abs().mean())
    fc_std = float(forecast.std())
    fc_abs_mean = float(forecast.abs().mean())
    print(
        f"             [diag] target |mean|={tgt_abs_mean:.3f} "
        f"forecast std={fc_std:.3f} |mean|={fc_abs_mean:.3f}"
    )

    c = crps_fn(target, forecast)
    c_sum = crps_sum_fn(target, forecast)

    if ema is not None:
        ema.restore(ddpm.model)

    return c, c_sum

