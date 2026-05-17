import torch
from torch.utils.data import Dataset, DataLoader
from generate_data import generate_gbm_paths, prices_to_log_returns, S0, mu, sigma, corr, T, dt

# ──────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────
WINDOW = 80        # total window length
COND_LEN = 60      # condition (history) length
PRED_LEN = 20      # prediction (target) length
BATCH_SIZE = 128

assert COND_LEN + PRED_LEN == WINDOW


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class StockReturnDataset(Dataset):
    """
    Sliding-window dataset over log-return matrix.

    Args:
        log_returns: Tensor of shape (T, D)
        cond_len:    length of the condition window
        pred_len:    length of the prediction window
    """

    def __init__(self, log_returns: torch.Tensor, cond_len: int = COND_LEN, pred_len: int = PRED_LEN):
        super().__init__()
        self.data = log_returns          # (T, D)
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.window = cond_len + pred_len
        self.n_samples = len(log_returns) - self.window + 1
        assert self.n_samples > 0, f"Not enough data: {len(log_returns)} days < {self.window} window"

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        # Extract full window: (window, D)
        window = self.data[idx : idx + self.window]

        # Split into condition / target: both (?, D)
        x_raw = window[: self.cond_len]    # (60, D)
        y_raw = window[self.cond_len :]    # (20, D)

        # Z-score using condition statistics (per-stock)
        x_mean = x_raw.mean(dim=0)         # (D,)
        x_std = x_raw.std(dim=0).clamp(min=1e-8)  # (D,)

        x_norm = ((x_raw - x_mean) / x_std).T   # (D, 60)
        y_norm = ((y_raw - x_mean) / x_std).T   # (D, 20)

        return x_norm, y_norm, x_mean, x_std


# ──────────────────────────────────────────────
# Multi-series Dataset (for large-scale training)
# ──────────────────────────────────────────────
class MultiSeriesStockDataset(Dataset):
    """
    Sliding-window dataset over MULTIPLE independent log-return series.

    Windows never cross a series boundary (each window indexed by (series_idx, offset)).
    Z-score normalization uses each sample's own 60-day condition stats.

    Args:
        series_list: list of Tensors, each shape (T_i, D)
        cond_len, pred_len: same as StockReturnDataset
    """

    def __init__(self, series_list, cond_len: int = COND_LEN, pred_len: int = PRED_LEN):
        super().__init__()
        self.series = series_list
        self.cond_len = cond_len
        self.pred_len = pred_len
        self.window = cond_len + pred_len

        self.index = []
        for s_idx, s in enumerate(series_list):
            n = len(s) - self.window + 1
            assert n > 0, f"Series {s_idx} too short: {len(s)} < {self.window}"
            for off in range(n):
                self.index.append((s_idx, off))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        s_idx, off = self.index[idx]
        window = self.series[s_idx][off : off + self.window]   # (80, D)

        x_raw = window[: self.cond_len]    # (60, D)
        y_raw = window[self.cond_len :]    # (20, D)

        x_mean = x_raw.mean(dim=0)
        x_std = x_raw.std(dim=0).clamp(min=1e-8)

        x_norm = ((x_raw - x_mean) / x_std).T   # (D, 60)
        y_norm = ((y_raw - x_mean) / x_std).T   # (D, 20)

        return x_norm, y_norm, x_mean, x_std


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # Generate data
    prices = generate_gbm_paths(S0, mu, sigma, corr, T, dt)
    log_returns = prices_to_log_returns(prices)
    print(f"Log returns: {log_returns.shape}")  # (1000, 10)

    # Build dataset & loader
    dataset = StockReturnDataset(log_returns)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batches per epoch: {len(loader)}")

    # Verify one batch
    x, y, mean, std = next(iter(loader))
    print(f"\nBatch shapes:")
    print(f"  X (condition): {x.shape}")    # [128, 10, 60]
    print(f"  Y (target):    {y.shape}")    # [128, 10, 20]
    print(f"  mean:          {mean.shape}")  # [128, 10]
    print(f"  std:           {std.shape}")   # [128, 10]

    # Sanity check: X should be ~ zero-mean, unit-var per stock
    print(f"\nX mean per stock (should ≈ 0): {x.mean(dim=2)[0].tolist()[:3]}...")
    print(f"X std  per stock (should ≈ 1): {x.std(dim=2)[0].tolist()[:3]}...")
