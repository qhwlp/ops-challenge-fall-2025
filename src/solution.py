import numpy as np
import polars as pl
from numba import njit


@njit
def _compute_beta(cov: np.ndarray, var: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    out = np.empty(n, dtype=np.float64)

    for i in range(n):
        cov_val = cov[i]
        var_val = var[i]

        if np.isnan(cov_val) or np.isnan(var_val):
            out[i] = np.nan
            continue

        if var_val < 1e-6:
            out[i] = 0.0
            continue

        out[i] = cov_val / var_val

    return out


def ops_rolling_regbeta(input_path: str, window: int = 20) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be a positive integer")

    lazy_df = (
        pl.scan_parquet(input_path)
        .with_columns(
            [
                pl.col("symbol").cast(pl.Categorical),
                pl.col("Low").cast(pl.Float64),
                pl.col("Close").cast(pl.Float64),
            ]
        )
        .select(
            [
                pl.col("symbol"),
                pl.rolling_cov(
                    pl.col("Low"),
                    pl.col("Close"),
                    window_size=window,
                    ddof=1,
                    min_samples=2,
                )
                .over("symbol")
                .alias("cov"),
                pl.col("Low")
                .rolling_var(window_size=window, ddof=1, min_samples=2)
                .over("symbol")
                .alias("var"),
            ]
        )
    )

    df = lazy_df.collect()
    if df.is_empty():
        return np.empty((0, 1), dtype=np.float64)

    cov = np.asarray(df["cov"].to_numpy(), dtype=np.float64)
    var = np.asarray(df["var"].to_numpy(), dtype=np.float64)

    beta = _compute_beta(cov, var)
    return beta.reshape(-1, 1)
