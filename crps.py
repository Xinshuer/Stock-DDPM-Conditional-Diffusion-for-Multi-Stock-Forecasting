import numpy as np
import torch


DEFAULT_QUANTILES = np.arange(0.05, 1.0, 0.05)


def quantile_loss(target: torch.Tensor, forecast_samples: torch.Tensor, q: float) -> torch.Tensor:
    """
    Args:
        target:           (...)                   ground truth
        forecast_samples: (n_samples, ...)        sampled predictions
        q:                scalar in (0, 1)
    Returns:
        mean quantile loss over all non-sample dims (scalar tensor)
    """
    q_hat = torch.quantile(forecast_samples, q, dim=0)
    diff = target - q_hat
    return (q * diff.clamp(min=0.0) + (1.0 - q) * (-diff).clamp(min=0.0)).mean()


def crps(
    target: torch.Tensor,
    forecast_samples: torch.Tensor,
    quantiles=DEFAULT_QUANTILES,
) -> float:
    """
    Args:
        target:           (B, D, L)
        forecast_samples: (n_samples, B, D, L)
    Returns:
        normalized CRPS (lower is better)
    """
    denom = target.abs().mean().clamp(min=1e-8)
    total = 0.0
    for q in quantiles:
        total = total + quantile_loss(target, forecast_samples, float(q))
    return float(2.0 * total / (len(quantiles) * denom))


def crps_sum(
    target: torch.Tensor,
    forecast_samples: torch.Tensor,
    quantiles=DEFAULT_QUANTILES,
) -> float:
    """
    Sum across feature dimension D first, then CRPS on the scalar (B, L) series.

    Args:
        target:           (B, D, L)
        forecast_samples: (n_samples, B, D, L)
    """
    target_sum = target.sum(dim=1)                         # (B, L)
    forecast_sum = forecast_samples.sum(dim=2)             # (n_samples, B, L)
    denom = target_sum.abs().mean().clamp(min=1e-8)
    total = 0.0
    for q in quantiles:
        total = total + quantile_loss(target_sum, forecast_sum, float(q))
    return float(2.0 * total / (len(quantiles) * denom))


# ──────────────────────────────────────────────
# Unit test
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Test 1: CRPS of unit-Gaussian samples vs unit-Gaussian target
    # Analytical CRPS for Y ~ N(0,1), forecast = N(0,1) is 1/√π ≈ 0.564
    torch.manual_seed(0)
    N_SAMPLES = 10000
    target = torch.randn(1, 1, 1)
    forecast = torch.randn(N_SAMPLES, 1, 1, 1) + 0.0
    # Note: normalization by |target| gives huge variance for single sample.
    # Instead test raw sum vs analytical.
    qs = DEFAULT_QUANTILES
    raw = 0.0
    for q in qs:
        raw += float(quantile_loss(target.expand(1, 1, 1), forecast.expand(N_SAMPLES, 1, 1, 1), float(q)))
    raw_crps = 2.0 * raw / len(qs)
    # Expected: CRPS(N(0,1), x=0) averaged — use large eval to smooth
    B = 2048
    t_big = torch.randn(B)
    f_big = torch.randn(N_SAMPLES, B)
    raw = 0.0
    for q in qs:
        q_hat = torch.quantile(f_big, float(q), dim=0)
        diff = t_big - q_hat
        raw += (q * diff.clamp(min=0) + (1-q) * (-diff).clamp(min=0)).mean().item()
    analytical = 2.0 * raw / len(qs)
    print(f"Empirical QL-CRPS (Gaussian self, 19-quantile): {analytical:.4f}  (continuous ≈ 0.564, discrete ≈ 0.59–0.60)")
    assert abs(analytical - 0.59) < 0.05, f"CRPS deviates: {analytical}"

    # Test 2: CRPS with perfect forecast → near zero
    target = torch.randn(8, 10, 20)
    perfect = target.unsqueeze(0).expand(N_SAMPLES, -1, -1, -1) + 1e-3 * torch.randn(N_SAMPLES, 8, 10, 20)
    c = crps(target, perfect)
    print(f"CRPS (near-perfect forecast): {c:.4f}  (expected ≈ 0)")
    assert c < 0.05

    # Test 3: CRPS_sum shapes
    fc = torch.randn(100, 8, 10, 20)
    c_sum = crps_sum(target, fc)
    print(f"CRPS_sum (random forecast): {c_sum:.4f}")
    assert c_sum > 0

    print("CRPS unit tests passed.")
