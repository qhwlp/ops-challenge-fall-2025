import numpy as np
import polars as pl
from math import fsum
from numba import njit


@njit
def _rolling_beta_symbol(
    low: np.ndarray,
    close: np.ndarray,
    window: int,
    beta_out: np.ndarray,
    var_out: np.ndarray,
) -> None:
    n = low.shape[0]
    for i in range(n):
        start = i - window + 1
        if start < 0:
            start = 0

        count = i - start + 1
        if count < 2:
            beta_out[i] = 0.0
            var_out[i] = 0.0
            continue

        sum_low = 0.0
        sum_close = 0.0
        sum_low_low = 0.0
        sum_low_close = 0.0

        for j in range(start, i + 1):
            val_low = low[j]
            val_close = close[j]
            sum_low += val_low
            sum_close += val_close
            sum_low_low += val_low * val_low
            sum_low_close += val_low * val_close

        count_f = float(count)
        cov_num = sum_low_close - (sum_low * sum_close) / count_f
        var_num = sum_low_low - (sum_low * sum_low) / count_f

        denom = count_f - 1.0
        if denom <= 0.0:
            beta_out[i] = 0.0
            var_out[i] = 0.0
            continue

        var = var_num / denom
        var_out[i] = var
        if var < 1e-6:
            beta_out[i] = 0.0
            continue

        cov = cov_num / denom
        beta_out[i] = cov / var


def _refine_beta(
    low: np.ndarray,
    close: np.ndarray,
    start: int,
    end: int,
) -> float:
    window_low = low[start:end]
    window_close = close[start:end]

    count = window_low.shape[0]
    if count < 2:
        return 0.0

    count_f = float(count)
    sum_low = fsum(float(x) for x in window_low)
    sum_close = fsum(float(x) for x in window_close)
    mean_low = sum_low / count_f
    mean_close = sum_close / count_f

    var = fsum((float(x) - mean_low) * (float(x) - mean_low) for x in window_low)
    if var <= 0.0:
        return 0.0

    cov = fsum(
        (float(window_low[k]) - mean_low) * (float(window_close[k]) - mean_close)
        for k in range(count)
    )

    denom = count_f - 1.0
    var /= denom
    if var < 1e-6:
        return 0.0

    cov /= denom
    return cov / var


def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    df = (
        pl.scan_parquet(input_path)
        .select(
            [
                pl.col("symbol").cast(pl.Categorical).alias("symbol"),
                pl.col("Low").cast(pl.Float64).alias("Low"),
                pl.col("Close").cast(pl.Float64).alias("Close"),
            ]
        )
        .collect()
    )

    if df.is_empty():
        return np.empty((0, 1), dtype=np.float64)

    symbols = df["symbol"]
    codes = np.asarray(symbols.to_physical().to_numpy(), dtype=np.int64)
    low = np.asarray(df["Low"].to_numpy(), dtype=np.float64)
    close = np.asarray(df["Close"].to_numpy(), dtype=np.float64)

    order = np.argsort(codes, kind="stable")
    sorted_low = low[order]
    sorted_close = close[order]
    sorted_codes = codes[order]

    num_symbols = len(symbols.cat.get_categories())
    counts = np.bincount(sorted_codes, minlength=num_symbols)

    sorted_beta = np.empty_like(sorted_low)
    sorted_var = np.empty_like(sorted_low)

    offsets = np.empty(counts.size + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])

    for g in range(num_symbols):
        start = offsets[g]
        end = offsets[g + 1]
        if end > start:
            _rolling_beta_symbol(
                sorted_low[start:end],
                sorted_close[start:end],
                int(window),
                sorted_beta[start:end],
                sorted_var[start:end],
            )

    threshold = 1e-7
    mask = (sorted_var > 0.0) & (sorted_var < threshold)
    indices = np.nonzero(mask)[0]
    if indices.size:
        for idx in indices:
            sym = np.searchsorted(offsets, idx, side="right") - 1
            sym_start = offsets[sym]
            start = max(sym_start, idx - window + 1)
            end = idx + 1
            sorted_beta[idx] = _refine_beta(sorted_low, sorted_close, start, end)

    result = np.empty_like(sorted_beta)
    result[order] = sorted_beta
    return result.reshape(-1, 1)
