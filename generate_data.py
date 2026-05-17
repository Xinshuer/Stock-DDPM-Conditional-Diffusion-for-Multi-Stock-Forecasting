import numpy as np
import torch
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────
# 1. Parameters
# ──────────────────────────────────────────────
D = 10          # number of stocks
T = 1000         # number of trading days
dt = 1 / 252    # daily time step (annualized)
S0 = np.full(D, 100.0)  # initial prices

# Annualized drift per stock (small positive)
mu = np.array([0.05, 0.06, 0.04, 0.05,   # sector A (tech-like)
               0.03, 0.04, 0.05,           # sector B (finance-like)
               0.02, 0.06, 0.03])          # sector C + hedge stock

# Annualized volatility per stock
sigma = np.array([0.20, 0.25, 0.18, 0.22,
                  0.15, 0.20, 0.30,
                  0.35, 0.28, 0.16])

# ──────────────────────────────────────────────
# 2. Correlation Matrix (10x10)
# ──────────────────────────────────────────────
# Sector A: stocks 0-3 (high positive ~0.7)
# Sector B: stocks 4-6 (moderate positive ~0.6)
# Sector C: stocks 7-8 (moderate positive ~0.5)
# Stock 9:  "hedge" asset, negatively correlated with most others
corr = np.array([
    # 0     1     2     3     4     5     6     7     8     9
    [1.00, 0.70, 0.65, 0.72, 0.30, 0.25, 0.20, 0.15, 0.10,-0.30],  # 0
    [0.70, 1.00, 0.68, 0.66, 0.28, 0.22, 0.18, 0.12, 0.08,-0.25],  # 1
    [0.65, 0.68, 1.00, 0.60, 0.25, 0.20, 0.15, 0.10, 0.12,-0.20],  # 2
    [0.72, 0.66, 0.60, 1.00, 0.32, 0.28, 0.22, 0.18, 0.14,-0.35],  # 3
    [0.30, 0.28, 0.25, 0.32, 1.00, 0.60, 0.55, 0.20, 0.18,-0.15],  # 4
    [0.25, 0.22, 0.20, 0.28, 0.60, 1.00, 0.58, 0.22, 0.20,-0.10],  # 5
    [0.20, 0.18, 0.15, 0.22, 0.55, 0.58, 1.00, 0.25, 0.22,-0.20],  # 6
    [0.15, 0.12, 0.10, 0.18, 0.20, 0.22, 0.25, 1.00, 0.50,-0.30],  # 7
    [0.10, 0.08, 0.12, 0.14, 0.18, 0.20, 0.22, 0.50, 1.00,-0.25],  # 8
    [-0.30,-0.25,-0.20,-0.35,-0.15,-0.10,-0.20,-0.30,-0.25, 1.00],  # 9
])

# ──────────────────────────────────────────────
# 3. Generate Correlated GBM Paths
# ──────────────────────────────────────────────
def generate_gbm_paths(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    T: int,
    dt: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate correlated GBM price paths via Cholesky decomposition.

    Returns:
        prices: ndarray of shape (T+1, D) — daily prices including t=0.
    """
    rng = np.random.default_rng(seed)
    D = len(S0)
    L = np.linalg.cholesky(corr)  # (D, D)

    # iid standard normals → correlated normals
    Z_iid = rng.standard_normal((T, D))   # (T, D)
    Z_corr = Z_iid @ L.T                  # (T, D)

    # GBM: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    drift = (mu - 0.5 * sigma ** 2) * dt          # (D,)
    diffusion = sigma * np.sqrt(dt) * Z_corr       # (T, D)
    log_increments = drift[np.newaxis, :] + diffusion  # (T, D)

    log_prices = np.zeros((T + 1, D))
    log_prices[0] = np.log(S0)
    log_prices[1:] = np.log(S0) + np.cumsum(log_increments, axis=0)

    return np.exp(log_prices)


# ──────────────────────────────────────────────
# 3b. Multi-seed generation (for large-scale training)
# ──────────────────────────────────────────────
def generate_multi_seed_log_returns(
    n_seeds: int = 50,
    T_per_seed: int = 2000,
    seed_start: int = 0,
):
    """
    Generate `n_seeds` independent GBM realizations sharing the SAME correlation
    structure (same Cholesky of `corr`), each of length T_per_seed days.

    Returns:
        list of torch.Tensor, each of shape (T_per_seed, D) — log returns.
    """
    assert np.linalg.eigvalsh(corr).min() > 0, "corr matrix must be positive definite"
    series = []
    for s in range(n_seeds):
        prices = generate_gbm_paths(S0, mu, sigma, corr, T_per_seed, dt, seed=seed_start + s)
        series.append(prices_to_log_returns(prices))
    return series


# ──────────────────────────────────────────────
# 4. Log Returns
# ──────────────────────────────────────────────
def prices_to_log_returns(prices: np.ndarray) -> torch.Tensor:
    """
    Convert price matrix to log returns tensor.

    Args:
        prices: ndarray of shape (T+1, D)
    Returns:
        log_returns: torch.Tensor of shape (T, D)
    """
    log_ret = np.log(prices[1:] / prices[:-1])
    return torch.tensor(log_ret, dtype=torch.float32)


# ──────────────────────────────────────────────
# 5. Visualization
# ──────────────────────────────────────────────
def plot_cumulative_returns(prices: np.ndarray, n_stocks: int = 5, n_days: int = 20):
    """
    Plot cumulative price paths (normalized to 1.0 at t=0) for the first
    `n_stocks` stocks over the first `n_days` days.
    """
    subset = prices[:n_days + 1, :n_stocks]        # (n_days+1, n_stocks)
    normalized = subset / subset[0:1, :]            # normalize to 1.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_stocks):
        ax.plot(normalized[:, i], label=f"Stock {i}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Return (normalized)")
    ax.set_title(f"Simulated GBM Paths — First {n_stocks} Stocks, {n_days} Days")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cumulative_returns.png", dpi=150)
    print("Plot saved to cumulative_returns.png")
    plt.show()


# ──────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    prices = generate_gbm_paths(S0, mu, sigma, corr, T, dt)
    log_returns = prices_to_log_returns(prices)

    print(f"Prices shape:      {prices.shape}")       # (101, 10)
    print(f"Log returns shape:  {log_returns.shape}")  # (100, 10)
    print(f"Log returns dtype:  {log_returns.dtype}")  # float32

    # Verify correlation structure
    empirical_corr = np.corrcoef(log_returns.numpy(), rowvar=False)
    print("\n— Target Correlation (first 5x5) —")
    print(np.round(corr[:5, :5], 2))
    print("\n— Empirical Correlation (first 5x5) —")
    print(np.round(empirical_corr[:5, :5], 2))

    plot_cumulative_returns(prices, n_stocks=5, n_days=20)
