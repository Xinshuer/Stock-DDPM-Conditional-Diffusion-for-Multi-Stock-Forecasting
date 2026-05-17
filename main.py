import argparse
import math
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from generate_data import generate_multi_seed_log_returns
from dataset import MultiSeriesStockDataset, COND_LEN, PRED_LEN
from model import ConditionalDenoiser
from ddpm import StockDDPM, train, evaluate_crps, EMA


def plot_training_curves(history, out_dir="."):
    """Two separate charts: (1) Train+Val MSE, (2) Val CRPS + CRPS_sum."""
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    # ── Chart 1: Train MSE vs Val MSE (traditional style) ─────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], color="tab:blue", lw=1.2, label="Train MSE")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], color="tab:orange", lw=1.5, label="Val MSE")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (log)")
    ax.set_title("Training vs Validation Loss (MSE on noise prediction)")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    fn1 = f"{out_dir}/training_vs_validation.png"
    plt.savefig(fn1, dpi=120)
    plt.close(fig)

    # ── Chart 2: Val CRPS + Val CRPS_sum ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    if history.get("val_crps"):
        ec = [e for e, _ in history["val_crps"]]
        vc = [v for _, v in history["val_crps"]]
        ecs = [e for e, _ in history["val_crps_sum"]]
        vcs = [v for _, v in history["val_crps_sum"]]
        ax.plot(ec, vc, "o-", color="tab:orange", lw=1.8, markersize=9, label="Val CRPS")
        ax.plot(ecs, vcs, "s--", color="tab:red", lw=1.5, markersize=8, label="Val CRPS_sum")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CRPS (lower is better)")
    ax.set_title("Validation CRPS — Probabilistic Forecast Quality")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fn2 = f"{out_dir}/crps_curve.png"
    plt.savefig(fn2, dpi=120)
    plt.close(fig)

    return fn1, fn2


def plot_actual_vs_generated(ddpm, dataset, device, n_windows=10, n_paths=500, out_dir="."):
    """
    For `n_windows` evenly-spaced test windows, plot actual vs generated paths
    for ALL 10 stocks in one PNG per window.
    """
    n = len(dataset)
    indices = np.linspace(0, n - 1, n_windows, dtype=int)
    saved = []

    for sample_i, idx in enumerate(indices):
        x_cond, y_true, x_mean, x_std = dataset[int(idx)]
        x_cond_batch = x_cond.unsqueeze(0).to(device)

        samples = ddpm.sample_ddpm(x_cond_batch, n_samples=n_paths)
        samples_np = samples.cpu().numpy()

        x_mean_np = x_mean.numpy()
        x_std_np = x_std.numpy()
        x_cond_raw = x_cond.numpy() * x_std_np[:, None] + x_mean_np[:, None]        # (10, 60)
        y_true_raw = y_true.numpy() * x_std_np[:, None] + x_mean_np[:, None]        # (10, 20)
        gen_raw = samples_np * x_std_np[None, :, None] + x_mean_np[None, :, None]   # (n_paths, 10, 20)

        D = x_cond_raw.shape[0]
        fig, axes = plt.subplots(2, 5, figsize=(22, 8.5))
        for stock in range(D):
            ax = axes.flat[stock]
            # History cumulative price (normalized to 1 at t=0)
            hist_cum = np.concatenate([[0.0], np.cumsum(x_cond_raw[stock])])
            hist_price = np.exp(hist_cum)
            last = hist_price[-1]

            # True future
            true_cum = np.cumsum(y_true_raw[stock])
            true_price = np.concatenate([[last], last * np.exp(true_cum)])

            # Generated paths
            gen_cum = np.cumsum(gen_raw[:, stock, :], axis=1)
            gen_price = last * np.exp(gen_cum)
            gen_price = np.concatenate([np.full((gen_raw.shape[0], 1), last), gen_price], axis=1)

            t_hist = np.arange(len(hist_price))
            t_future = np.arange(len(hist_price) - 1, len(hist_price) - 1 + len(true_price))

            ax.plot(t_hist, hist_price, color="black", lw=1.5, label="history")
            for j in range(min(200, len(gen_price))):
                ax.plot(t_future, gen_price[j], color="steelblue", alpha=0.04, lw=0.5)

            p5 = np.percentile(gen_price, 5, axis=0)
            p50 = np.percentile(gen_price, 50, axis=0)
            p95 = np.percentile(gen_price, 95, axis=0)
            ax.fill_between(t_future, p5, p95, alpha=0.15, color="steelblue")
            ax.plot(t_future, p50, color="steelblue", lw=1.2, linestyle="--", label="gen median")

            ax.plot(t_future, true_price, color="crimson", lw=1.8, label="actual")

            ax.axvline(len(hist_price) - 1, color="gray", linestyle=":", alpha=0.5)
            ax.set_title(f"Stock {stock}", fontsize=10)
            ax.grid(alpha=0.3)
            if stock == 0:
                ax.legend(loc="upper left", fontsize=8)

        fig.suptitle(f"Sample {sample_i:02d} (test window idx={int(idx)}) — actual (red) vs generated (blue fan)",
                     fontsize=12)
        plt.tight_layout()
        fn = f"{out_dir}/sample_actual_vs_gen_{sample_i:02d}.png"
        plt.savefig(fn, dpi=100)
        plt.close(fig)
        saved.append(fn)

    return saved


def plot_fan_chart(x_cond_raw, generated_paths, stock_idx, seed_idx, sampler_name, out_dir="."):
    """x_cond_raw: (cond_len,) log returns;  generated_paths: (n_samples, pred_len)"""
    cond_len = len(x_cond_raw)
    pred_len = generated_paths.shape[1]

    hist_cum = np.concatenate([[0.0], np.cumsum(x_cond_raw)])
    hist_price = np.exp(hist_cum)
    last_price = hist_price[-1]

    future_cum = np.cumsum(generated_paths, axis=1)
    future_price = last_price * np.exp(future_cum)
    branch = np.full((future_price.shape[0], 1), last_price)
    future_price = np.concatenate([branch, future_price], axis=1)

    t_hist = np.arange(cond_len + 1)
    t_future = np.arange(cond_len, cond_len + pred_len + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(t_hist, hist_price, color="black", linewidth=2, label="History (60)")

    for i in range(min(200, len(future_price))):
        ax.plot(t_future, future_price[i], color="steelblue", alpha=0.05, linewidth=0.5)

    p5 = np.percentile(future_price, 5, axis=0)
    p25 = np.percentile(future_price, 25, axis=0)
    p50 = np.percentile(future_price, 50, axis=0)
    p75 = np.percentile(future_price, 75, axis=0)
    p95 = np.percentile(future_price, 95, axis=0)
    ax.fill_between(t_future, p5, p95, alpha=0.15, color="steelblue", label="5-95 pctl")
    ax.fill_between(t_future, p25, p75, alpha=0.25, color="steelblue", label="25-75 pctl")
    ax.plot(t_future, p50, color="steelblue", linestyle="--", label="Median")

    ax.axvline(x=cond_len, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Day")
    ax.set_ylabel("Normalized Price")
    ax.set_title(f"Test seed {seed_idx}, stock {stock_idx} — {sampler_name} ({len(future_price)} paths)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fn = f"{out_dir}/fan_{sampler_name}_seed{seed_idx}_stock{stock_idx}.png"
    plt.savefig(fn, dpi=120)
    plt.close(fig)
    return fn


def main():
    parser = argparse.ArgumentParser()
    # Compressed-default config (ship-ready for ~2-3 hour GPU run).
    # Override individual flags for larger runs, or use --smoke for quick sanity.
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (5 seeds, 10 epochs)")
    parser.add_argument("--n_seeds", type=int, default=20)
    parser.add_argument("--T", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_samples_eval", type=int, default=30)
    parser.add_argument("--val_every_epoch", type=int, default=20)
    parser.add_argument("--ckpt_every_epoch", type=int, default=20)
    parser.add_argument("--ckpt_dir", type=str, default="g:/project3/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--out_dir", type=str, default=".")
    args = parser.parse_args()

    if args.smoke:
        args.n_seeds = 5
        args.T = 800
        args.epochs = 10
        args.n_samples_eval = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=" * 70)
    print(f"CSDI+TimeGrad Hybrid Reproduction")
    print(f"Device: {device}   |   smoke={args.smoke}")
    print(f"n_seeds={args.n_seeds}  T={args.T}  epochs={args.epochs}  bs={args.batch_size}")
    print(f"=" * 70)

    # ── 1. Data generation ────────────────────────
    t0 = time.time()
    print(f"\n[1/6] Generating {args.n_seeds} seeds x T={args.T} days...")
    series = generate_multi_seed_log_returns(n_seeds=args.n_seeds, T_per_seed=args.T, seed_start=0)
    print(f"      generated {len(series)} series, each of shape {tuple(series[0].shape)}")
    print(f"      elapsed: {time.time()-t0:.1f}s")

    # ── 2. Split ──────────────────────────────────
    n = args.n_seeds
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    n_train = n - n_val - n_test
    assert n_train >= 1, f"n_seeds={n} too small: need at least 3"
    train_series = series[:n_train]
    val_series = series[n_train : n_train + n_val]
    test_series = series[n_train + n_val :]
    print(f"\n[2/6] Split: train={n_train}, val={n_val}, test={n_test}")

    # ── 3. Datasets / Loaders ─────────────────────
    train_ds = MultiSeriesStockDataset(train_series, COND_LEN, PRED_LEN)
    val_ds = MultiSeriesStockDataset(val_series, COND_LEN, PRED_LEN)
    test_ds = MultiSeriesStockDataset(test_series, COND_LEN, PRED_LEN)
    print(f"      train={len(train_ds)} windows, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print(f"      batches/epoch: {len(train_loader)}")

    # ── 4. Model + DDPM ───────────────────────────
    D = series[0].shape[1]
    model = ConditionalDenoiser(d_stocks=D, cond_len=COND_LEN, pred_len=PRED_LEN)
    ddpm = StockDDPM(model, n_steps=50, beta_start=1e-4, beta_end=0.5, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[3/6] Model: ConditionalDenoiser, {n_params:,} params")
    print(f"      Schedule: quadratic beta, N=50, ab_N={ddpm.alpha_bar[-1]:.2e}")

    # ── 5. Train ──────────────────────────────────
    print(f"\n[4/6] Training {args.epochs} epochs...")
    import os
    os.makedirs(args.ckpt_dir, exist_ok=True)

    milestones = (int(args.epochs * 0.75), int(args.epochs * 0.90))
    history, ema = train(
        ddpm,
        train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=1e-6,
        lr_milestones=milestones,
        grad_clip=1.0,
        use_ema=True,
        val_every_epoch=args.val_every_epoch,
        val_n_samples=args.n_samples_eval,
        ckpt_dir=args.ckpt_dir,
        ckpt_every_epoch=args.ckpt_every_epoch,
        resume_from=args.resume,
    )

    # ── 6. Final eval on test set ─────────────────
    print(f"\n[5/6] Final evaluation on test set...")
    # Limit eval batches for speed: ~20 batches × bs=128 = 2560 windows
    max_eval_batches = min(len(test_loader), 20)

    crps_ddpm = evaluate_crps(
        ddpm, test_loader, n_samples=args.n_samples_eval,
        sampler="ddpm", ema=ema, max_batches=max_eval_batches,
    )
    print(f"      DDPM-50  CRPS={crps_ddpm[0]:.4f}  CRPS_sum={crps_ddpm[1]:.4f}")

    crps_ddim = evaluate_crps(
        ddpm, test_loader, n_samples=args.n_samples_eval,
        sampler="ddim", n_ddim_steps=10, ema=ema, max_batches=max_eval_batches,
    )
    print(f"      DDIM-10  CRPS={crps_ddim[0]:.4f}  CRPS_sum={crps_ddim[1]:.4f}")

    # ── 7. Training curves + 10 actual-vs-generated samples ───
    print(f"\n[6/6] Generating visualizations...")
    fn1, fn2 = plot_training_curves(history, out_dir=args.out_dir)
    print(f"      saved {fn1}")
    print(f"      saved {fn2}")

    if ema is not None:
        ema.apply_to(ddpm.model)
    ddpm.model.eval()

    saved = plot_actual_vs_generated(
        ddpm, test_ds, device,
        n_windows=10, n_paths=500, out_dir=args.out_dir,
    )
    for fn in saved:
        print(f"      saved {fn}")

    if ema is not None:
        ema.restore(ddpm.model)

    print(f"\n{'='*70}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")
    print(f"FINAL CRPS  — DDPM-50: {crps_ddpm[0]:.4f}, DDIM-10: {crps_ddim[0]:.4f}")
    print(f"FINAL CRPS_sum — DDPM-50: {crps_ddpm[1]:.4f}, DDIM-10: {crps_ddim[1]:.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
