import numpy as np
import polars as pl
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "src"))
from solution import ops_rolling_regbeta, _rolling_beta_symbol
