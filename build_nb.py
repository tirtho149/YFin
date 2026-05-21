"""Builder: assemble DS2010_Final.ipynb.

Stage 1 of the research ladder: pure-OHLC vs finance-informed feature comparison.
"""
import json
import os

cells = []
def md(src):   cells.append(("markdown", src))
def code(src): cells.append(("code", src))

# ── Cell 1: title ─────────────────────────────────────────────────────────────
md(r'''
# Finance GPU ML Benchmark

Benchmarks four model families for stock **direction / return / volatility**
prediction: **classical ML** (XGBoost, RandomForest, Ridge / LogisticRegression),
**feed-forward MLPs**, **sequence models** (LSTM, Transformer), and a
**latent world model** (VAE encoder → latent transition → prediction head) —
across two feature representations:

- **Branch 1 — Pure OHLC**: compact features derived only from price & volume.
- **Branch 2 — Finance-Informed**: 91 technical indicators (RSI, MACD, Bollinger,
  ATR, ADX, Hurst, ...) that inject explicit financial inductive bias.

Research questions: *does finance domain knowledge improve prediction over
generic price-only learning?* and *which model family — classical, MLP,
sequence, or latent world model — wins?* Universe: ~325 US equities across
12 sectors.

> **Research ladder — complete.** This notebook covers all five stages:
> (1) feature comparison, (2) classical ML baselines, (3) sequence models,
> (4) a latent world model, and (5) walk-forward validation with regime-split
> testing and calibration metrics.

Run the cells top to bottom. Works on Google Colab with a GPU runtime, or
locally on CPU/GPU.
''')

# ── Cell 2: install ───────────────────────────────────────────────────────────
code(r'''
# Install dependencies (safe to re-run; -q keeps the output quiet).
!pip install -q yfinance numpy pandas scikit-learn scipy torch xgboost matplotlib
''')

# ── Cell 3: imports header ────────────────────────────────────────────────────
md(r'''## 1. Imports''')

# ── Cell 4: imports ───────────────────────────────────────────────────────────
code(r'''
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, roc_auc_score,
    balanced_accuracy_score, f1_score, matthews_corrcoef,
)
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
''')

# ── Cell 5: configuration header ──────────────────────────────────────────────
md(r'''## 2. Configuration

Every tunable setting — tickers, paths, date splits and hyperparameters — lives
in the cell below.''')

# ── Cell 6: configuration ─────────────────────────────────────────────────────
code(r'''
# --- Ticker universe (12 sectors) ---------------------------------------------
TICKER_UNIVERSE = {
    "Technology": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "ADBE", "CRM", "ORCL", "INTC",
        "CSCO", "IBM", "QCOM", "TXN", "AVGO", "AMD", "MU", "AMAT", "LRCX", "KLAC",
        "SNPS", "CDNS", "FTNT", "PANW", "CRWD", "ZS", "DDOG", "SNOW", "PLTR", "NET",
        "SHOP", "SQ", "PYPL", "WDAY", "VEEV", "TEAM", "DOCU", "MDB", "ANET", "SMCI",
        "HPQ", "DELL", "STX", "WDC", "NTAP", "PSTG", "OKTA", "ZM", "TWLO", "BILL",
    ],
    "Financials": [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "COF",
        "BLK", "SCHW", "USB", "PNC", "TFC", "FITB", "KEY", "RF", "HBAN", "CFG",
        "BK", "STT", "NTRS", "TROW", "IVZ", "PRU", "MET", "AFL", "ALL", "TRV",
        "HIG", "CB", "AON", "MMC", "ICE", "CME", "CBOE", "NDAQ", "HOOD", "SOFI",
    ],
    "Healthcare": [
        "JNJ", "UNH", "ABT", "TMO", "PFE", "MRK", "LLY", "BMY", "GILD", "AMGN",
        "BIIB", "REGN", "VRTX", "MRNA", "BSX", "SYK", "MDT", "BAX", "BDX", "ISRG",
        "IDXX", "IQV", "CRL", "ILMN", "ALGN", "HOLX", "HSIC", "CVS", "CI", "HUM",
    ],
    "Consumer_Discretionary": [
        "AMZN", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX", "ROST", "BKNG",
        "MAR", "HLT", "CCL", "RCL", "MGM", "WYNN", "ABNB", "UBER", "DASH", "EXPE",
        "AZO", "ORLY", "F", "GM", "RIVN", "NIO", "RL", "BBY", "ETSY", "EBAY",
    ],
    "Consumer_Staples": [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "KHC", "GIS",
        "K", "CPB", "HRL", "MKC", "CAG", "COKE", "KDP", "STZ", "MNST", "CELH",
    ],
    "Industrials": [
        "CAT", "DE", "HON", "GE", "BA", "RTX", "LMT", "NOC", "GD", "MMM",
        "EMR", "ETN", "ROK", "PH", "DOV", "ITW", "AME", "ROP", "FDX", "UPS",
        "CHRW", "EXPD", "JBHT", "KNX", "URI", "RSG", "WM", "UNP", "CSX", "NSC",
    ],
    "Energy": [
        "XOM", "CVX", "COP", "EOG", "PXD", "DVN", "MPC", "VLO", "PSX", "SLB",
        "HAL", "BKR", "OXY", "APA", "MRO", "HES", "EQT", "AR", "RRC", "CTRA",
        "ET", "EPD", "OKE", "WMB", "LNG", "NEE", "AES", "SO", "DUK", "PCG",
    ],
    "Materials": [
        "LIN", "APD", "SHW", "ECL", "PPG", "RPM", "DOW", "DD", "LYB", "NEM",
        "GOLD", "AEM", "FCX", "AA", "NUE", "STLD", "CMC", "RS", "ATI", "HWM",
    ],
    "Real_Estate": [
        "AMT", "PLD", "CCI", "EQIX", "PSA", "DLR", "O", "WELL", "AVB", "EQR",
        "MAA", "UDR", "CPT", "NNN", "ADC", "STAG", "IIPR", "VNO", "BXP", "SPG",
    ],
    "Communication_Services": [
        "GOOG", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "SNAP",
        "PINS", "RDDT", "BMBL", "MTCH", "TTWO", "EA", "RBLX", "SPOT", "NYT", "AMCX",
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "PCG", "PEG", "ED",
        "XEL", "ES", "EIX", "DTE", "PPL", "AEE", "LNT", "WEC", "AWK", "WTRG",
    ],
    "ETFs_Indices": [
        "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "GLD", "SLV", "TLT", "IEF",
        "AGG", "BND", "LQD", "HYG", "XLF", "XLK", "XLV", "XLE", "XLI", "VNQ",
    ],
}

# Flatten the universe into a deduplicated ticker list + ticker -> sector map.
TICKERS = []
TICKER_SECTOR = {}
for _sector, _syms in TICKER_UNIVERSE.items():
    for _t in _syms:
        if _t not in TICKER_SECTOR:
            TICKERS.append(_t)
            TICKER_SECTOR[_t] = _sector

# --- Compute device -----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Directories --------------------------------------------------------------
BASE_DIR    = "/content" if os.path.isdir("/content") else os.getcwd()
DATA_DIR    = os.path.join(BASE_DIR, "finance_output")
CORR_DIR    = os.path.join(BASE_DIR, "correlation_analysis")
OUT_DIR     = os.path.join(BASE_DIR, "comparison_report")
RESULTS_DIR = os.path.join(OUT_DIR, "gpu_multi_model_benchmark")
for _d in (DATA_DIR, CORR_DIR, OUT_DIR, RESULTS_DIR):
    os.makedirs(_d, exist_ok=True)

# Data files written by the download step and read by the rest of the notebook.
RETURNS_PATH = os.path.join(CORR_DIR, "returns_matrix.csv")
REGIME_PATH  = os.path.join(CORR_DIR, "phase5_regime_labels.csv")

# --- Data ---------------------------------------------------------------------
DATA_START_DATE  = "2015-01-01"   # first date of downloaded history
MIN_HISTORY_ROWS = 300            # skip tickers with fewer rows than this

# --- Feature engineering ------------------------------------------------------
DIRECTION_THRESHOLD = 0.005       # |next-day return| band for the neutral class
EPS = 1e-12                       # numerical-stability epsilon

# --- Train / validation / test date splits ------------------------------------
TRAIN_END = "2024-06-30"
VAL_END   = "2025-03-31"
TEST_END  = "2026-03-31"

# --- Prediction targets -------------------------------------------------------
TARGET_COLS = ["LogReturn_Next", "Direction", "Volatility_Next"]

# --- Model training: deep MLPs ------------------------------------------------
BATCH_SIZE    = 4096
N_EPOCHS      = 20
LEARNING_RATE = 1e-3
MODEL_SPECS   = {"small": (128, 64), "medium": (256, 128, 64)}

# --- Classical ML baselines (Stage 2) -----------------------------------------
# Fixed, sensible defaults -- these are baselines, not tuned models. RF defaults
# are deliberately Colab-friendly (large leaves, half-sample bootstrap, capped
# depth): RF is the slowest baseline on the ~325-ticker panel. XGBoost rides the
# GPU runtime when one is present via device=XGB_DEVICE.
XGB_DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
RF_PARAMS     = dict(n_estimators=150, max_depth=14, min_samples_leaf=50,
                     max_samples=0.5, n_jobs=-1, random_state=42)
XGB_PARAMS    = dict(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                     device=XGB_DEVICE, n_jobs=-1, random_state=42)
RIDGE_PARAMS  = dict(alpha=1.0)
LOGREG_PARAMS = dict(C=1.0, max_iter=2000)

# --- Model training: sequence models (Stage 3) --------------------------------
# LSTM / Transformer over a SEQ_LEN-day look-back window of per-day features.
# Sequence models are the heaviest stage -- lower SEQ_EPOCHS / SEQ_LEN to trade
# accuracy for runtime.
SEQ_LEN        = 30       # look-back window length in trading days
SEQ_EPOCHS     = 15
SEQ_BATCH_SIZE = 2048
SEQ_SPECS = {
    "LSTM":        dict(kind="lstm", hidden_size=128, num_layers=2, dropout=0.2),
    "Transformer": dict(kind="transformer", d_model=128, nhead=4,
                        num_layers=2, dropout=0.2),
}

# --- Latent world model (Stage 4) ---------------------------------------------
# VAE encoder -> GRU latent transition -> prediction head, over the same
# look-back windows as the sequence models.
WM_LATENT_DIM = 24
WM_HIDDEN     = 128
WM_EPOCHS     = 15
WM_BETA       = 0.5    # KL weight (beta-VAE)
WM_GAMMA      = 1.0    # latent-transition (next-latent prediction) weight
WM_SUP_WEIGHT = 5.0    # supervised prediction-head weight

# --- Walk-forward & regime evaluation (Stage 5) -------------------------------
WF_N_FOLDS        = 6      # expanding-window walk-forward folds
WF_MIN_TRAIN_FRAC = 0.40   # first 40% of dates reserved as the initial train floor

print(f"Device  : {DEVICE}")
print(f"Base    : {BASE_DIR}")
print(f"Tickers : {len(TICKERS)} across {len(TICKER_UNIVERSE)} sectors")
''')

# ── Cell 7: data download header ──────────────────────────────────────────────
md(r'''## 3. Data Download

Fetches OHLCV from Yahoo Finance (cached as one CSV per ticker), then builds the
returns matrix and default regime labels.''')

# ── Cell 8: data download ─────────────────────────────────────────────────────
code(r'''
def fetch_ohlcv(ticker, retries=3):
    """Download one ticker's OHLCV history, retrying transient API errors."""
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=DATA_START_DATE,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def download_data(pause=0.3):
    """Download OHLCV for every ticker in TICKERS, then build the returns matrix
    and default regime labels. Cached CSVs in DATA_DIR are reused when present,
    so re-running resumes where a previous run stopped."""
    print(f"Downloading OHLCV for {len(TICKERS)} tickers from Yahoo Finance ...")
    closes, failed = {}, []
    for i, ticker in enumerate(TICKERS, 1):
        path = os.path.join(DATA_DIR, f"{ticker}_historical.csv")
        try:
            if os.path.isfile(path):
                df = pd.read_csv(path, index_col=0)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()].sort_index()
            else:
                df = fetch_ohlcv(ticker)
                df.to_csv(path)
                time.sleep(pause)   # be gentle with the API across many tickers

            if "Close" not in df.columns or len(df) < MIN_HISTORY_ROWS:
                failed.append(ticker)
            else:
                closes[ticker] = df["Close"].astype(float)
        except Exception as ex:
            failed.append(ticker)
            print(f"  fail {ticker}: {ex}")
        if i % 25 == 0:
            print(f"  {i}/{len(TICKERS)} processed ...")

    print(f"Downloaded {len(closes)}/{len(TICKERS)} tickers"
          + (f"  ({len(failed)} skipped/failed)" if failed else ""))
    if not closes:
        raise RuntimeError("No ticker data downloaded. Check your connection.")

    # Returns matrix: daily log returns, one column per ticker.
    close_df = pd.DataFrame(closes).sort_index()
    ret_matrix = np.log(close_df / close_df.shift(1))
    ret_matrix.to_csv(RETURNS_PATH)
    print(f"Saved returns matrix -> {RETURNS_PATH}  shape={ret_matrix.shape}")

    # Regime labels: this notebook has no regime model, so default to 'medium'.
    pd.Series(["medium"] * len(ret_matrix.index), index=ret_matrix.index,
              name="rolling_mean_r").to_csv(REGIME_PATH)
    print(f"Saved regime labels  -> {REGIME_PATH}")


download_data()
''')

# ── Cell 9: feature engineering header ────────────────────────────────────────
md(r'''## 4. Feature Engineering

Two feature builders define the comparison; both emit the **same three targets**
so their results are directly comparable.

- `ohlc_features` — **Branch 1**, ~9 price/volume features only (no indicators).
- `finance_features` — **Branch 2**, 91 technical indicators across 9 families
  (returns, trend, momentum, volatility, volume, microstructure, regime,
  temporal, interactions).

`FEATURE_SETS` registers both so the benchmark can loop over them.''')

# ── Cell 10: feature engineering ──────────────────────────────────────────────
code(r'''
def _ema(s, n):
    """Exponential moving average."""
    return s.ewm(span=n, adjust=False).mean()


def _sma(s, n):
    """Simple moving average."""
    return s.rolling(n).mean()


def _true_range(df):
    """Wilder's True Range from an OHLC frame."""
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _add_targets(out, logret, direction_threshold):
    """Attach the three forward-looking targets and drop warm-up / tail NaN rows.

    LogReturn_Next  -- next-day log return
    Volatility_Next -- forward 5-day std of log returns
    Direction       -- {-1, 0, 1} from next-day return vs direction_threshold
    """
    out["LogReturn_Next"]  = logret.shift(-1)
    out["Volatility_Next"] = logret.shift(-1).rolling(5).std()
    nxt = out["LogReturn_Next"]
    out["Direction"] = np.where(nxt > direction_threshold, 1,
                                np.where(nxt < -direction_threshold, -1, 0))
    return out.dropna()


def ohlc_features(raw: pd.DataFrame,
                  direction_threshold: float = DIRECTION_THRESHOLD) -> pd.DataFrame:
    """Branch 1 -- 'pure OHLC' baseline feature set.

    A compact, model-agnostic set derived only from price and volume: lagged log
    returns, momentum, rolling volatility and intraday range -- no technical
    indicators or financial priors.
    """
    df = raw.copy()
    if "Close" not in df.columns:
        return pd.DataFrame()
    df = df.sort_index()
    c = df["Close"].astype(float)
    logret = np.log(c / c.shift(1))
    out = pd.DataFrame(index=df.index)

    out["Ret1"]   = logret.shift(1)
    out["Ret2"]   = logret.shift(2)
    out["Ret5"]   = logret.shift(5)
    out["Mom10"]  = (c / c.shift(10) - 1.0)
    out["Mom20"]  = (c / c.shift(20) - 1.0)
    out["Vol5"]   = logret.rolling(5).std().shift(1)
    out["Vol20"]  = logret.rolling(20).std().shift(1)
    out["Range1"] = (
        ((df["High"].astype(float) - df["Low"].astype(float)) / c).shift(1)
        if "High" in df.columns and "Low" in df.columns else np.nan
    )
    out["VolChg"] = (out["Vol5"] / (out["Vol20"] + EPS) - 1.0)

    return _add_targets(out, logret, direction_threshold)


def finance_features(raw: pd.DataFrame,
                     direction_threshold: float = DIRECTION_THRESHOLD) -> pd.DataFrame:
    """Branch 2 -- finance-informed feature set (91 technical indicators).

    Nine families injecting explicit financial inductive bias. Trend/MA features
    are scale-free (distance from price); volatility features are price-
    normalised; calendar features use cyclical sin/cos encodings.
    """
    raw = raw.copy()
    if "Close" not in raw.columns:
        return pd.DataFrame()
    if hasattr(raw.index, "tz") and raw.index.tz is not None:
        raw.index = raw.index.tz_localize(None)
    raw = raw.sort_index()
    raw.index = pd.DatetimeIndex(raw.index)
    c, h, l, o, v = (raw["Close"].astype(float), raw["High"].astype(float),
                     raw["Low"].astype(float), raw["Open"].astype(float),
                     raw["Volume"].astype(float))
    frames = []
    logret = np.log(c / c.shift(1))

    # F1 -- Returns
    frames.append(pd.DataFrame({
        "Ret1": c.pct_change(), "LogRet1": logret,
        "LogRet2": np.log(c / c.shift(2)), "LogRet5": np.log(c / c.shift(5)),
        "LogRet10": np.log(c / c.shift(10)), "LogRet20": np.log(c / c.shift(20)),
        "GapRet": np.log(o / c.shift(1)), "IntradayRet": np.log(c / o),
        "Cum5Ret": c.pct_change(5)}, index=raw.index))

    # F2 -- Trend (moving averages used only to derive scale-free distances)
    ma = {}
    for n in [5, 10, 20, 50, 200]:
        ma[f"SMA{n}"] = _sma(c, n)
        ma[f"EMA{n}"] = _ema(c, n)
    e10, e20, e12 = _ema(c, 10), _ema(c, 20), _ema(c, 12)
    ma["DEMA10"] = 2 * e10 - _ema(e10, 10)
    ma["DEMA20"] = 2 * e20 - _ema(e20, 20)
    ma["TEMA12"] = 3 * e12 - 3 * _ema(e12, 12) + _ema(_ema(e12, 12), 12)
    w10, w20 = np.arange(1, 11, dtype=float), np.arange(1, 21, dtype=float)
    ma["WMA10"] = c.rolling(10).apply(lambda x: np.dot(x, w10) / w10.sum(), raw=True)
    ma["WMA20"] = c.rolling(20).apply(lambda x: np.dot(x, w20) / w20.sum(), raw=True)
    ma["VWMA20"] = (c * v).rolling(20).sum() / v.rolling(20).sum()
    dist = {}
    for n in [10, 20, 50, 200]:
        dist[f"Dist_SMA{n}"] = (c - ma[f"SMA{n}"]) / c
        dist[f"Dist_EMA{n}"] = (c - ma[f"EMA{n}"]) / c
    dist["Cross_EMA5_EMA20"]   = (ma["EMA5"] - ma["EMA20"]) / c
    dist["Cross_EMA10_EMA50"]  = (ma["EMA10"] - ma["EMA50"]) / c
    dist["Cross_SMA50_SMA200"] = (ma["SMA50"] - ma["SMA200"]) / c
    dist["Dist_VWMA20"]        = (c - ma["VWMA20"]) / c
    frames.append(pd.DataFrame(dist, index=raw.index))

    # F3 -- Momentum
    def _rsi(s, n=14):
        d = s.diff()
        gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        loss = (-d).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
        return (100 - 100 / (1 + gain / (loss + 1e-9))) / 100
    e26 = _ema(c, 26)
    macd_line = (e12 - e26) / c
    macd_signal = _ema(macd_line, 9)
    tp = (h + l + c) / 3
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    lo21, hi21 = l.rolling(21).min(), h.rolling(21).max()
    mfv = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    mfr = (mfv.where(tp > tp.shift(1), 0).rolling(14).sum() /
           (mfv.where(tp < tp.shift(1), 0).rolling(14).sum().abs() + 1e-9))
    stp = _sma(tp, 20)
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    frames.append(pd.DataFrame({
        "RSI14": _rsi(c, 14), "RSI7": _rsi(c, 7),
        "MACD_line": macd_line, "MACD_signal": macd_signal,
        "MACD_hist": macd_line - macd_signal,
        "ROC5": c.pct_change(5), "ROC10": c.pct_change(10), "ROC20": c.pct_change(20),
        "Stoch_K14": (c - lo14) / (hi14 - lo14 + 1e-9),
        "Stoch_D14": ((c - lo14) / (hi14 - lo14 + 1e-9)).rolling(3).mean(),
        "Stoch_K21": (c - lo21) / (hi21 - lo21 + 1e-9),
        "Stoch_D21": ((c - lo21) / (hi21 - lo21 + 1e-9)).rolling(3).mean(),
        "WilliamsR": (hi14 - c) / (hi14 - lo14 + 1e-9),
        "CCI20": ((tp - stp) / (0.015 * mad + 1e-9)) / 200,
        "MFI14": (100 - 100 / (1 + mfr)) / 100,
        "DPO10": (c.shift(6) - _sma(c, 10)) / c}, index=raw.index))

    # F4 -- Volatility
    tr = _true_range(raw)
    mid20, std20 = _sma(c, 20), c.rolling(20).std()
    atr20 = tr.ewm(span=20, adjust=False).mean()
    log_hl, log_co = np.log(h / l) ** 2, np.log(c / o) ** 2
    gk = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    log_oc = np.log(o / c.shift(1)) ** 2
    rs_c = log_hl * 0.5 - log_co * (2 * np.log(2) - 1)
    k = 0.34 / (1.34 + 11 / 9)
    yz = log_oc + k * log_co + (1 - k) * rs_c
    r1 = c.pct_change()
    r2 = r1 ** 2
    v5, v20 = r1.rolling(5).std(), r1.rolling(20).std()
    frames.append(pd.DataFrame({
        "ATR7": tr.ewm(span=7, adjust=False).mean() / c,
        "ATR14": tr.ewm(span=14, adjust=False).mean() / c,
        "ATR21": tr.ewm(span=21, adjust=False).mean() / c,
        "BB_Width20": (4 * std20) / (mid20 + 1e-9),
        "BB_Pct20": (c - (mid20 - 2 * std20)) / (4 * std20 + 1e-9),
        "KC_Pct": (c - (_ema(c, 20) - 2 * atr20)) / (4 * atr20 + 1e-9),
        "GK_Vol10": np.sqrt(gk.rolling(10).mean().clip(lower=0)),
        "YZ_Vol10": np.sqrt(yz.rolling(10).mean().clip(lower=0)),
        "RolStd5": v5, "RolStd10": r1.rolling(10).std(),
        "RolStd20": v20, "RolStd60": r1.rolling(60).std(),
        "GARCH_proxy": 0.85 * r2.ewm(span=20, adjust=False).mean() + 0.1 * r2,
        "VolRatio_5_20": v5 / (v20 + 1e-9)}, index=raw.index))

    # F5 -- Volume
    obv = (np.sign(c.diff()) * v).cumsum()
    vwap = ((h + l + c) / 3 * v).rolling(20).sum() / v.rolling(20).sum()
    mfv2 = ((c - l) - (h - c)) / (h - l + 1e-9) * v
    ad = mfv2.cumsum()
    pvt = (c.pct_change() * v).cumsum()
    fi = c.diff() * v
    frames.append(pd.DataFrame({
        "OBV_Diff": obv.diff(),
        "OBV_ZScore": (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-9),
        "VWAP_Dist": (c - vwap) / c,
        "CMF20": mfv2.rolling(20).sum() / (v.rolling(20).sum() + 1e-9),
        "AD_ZScore": (ad - ad.rolling(20).mean()) / (ad.rolling(20).std() + 1e-9),
        "PVT_ZScore": (pvt - pvt.rolling(20).mean()) / (pvt.rolling(20).std() + 1e-9),
        "ForceIndex13": _ema(fi, 13) / (c * v.rolling(20).mean() + 1e-9),
        "Vol_ZScore": (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9),
        "VolRatio20": v / v.rolling(20).mean()}, index=raw.index))

    # F6 -- Microstructure
    rng = h - l + 1e-9
    body_top = pd.concat([c, o], axis=1).max(axis=1)
    body_bot = pd.concat([c, o], axis=1).min(axis=1)
    frames.append(pd.DataFrame({
        "HL_Range": (h - l) / c, "Body_Pct": (c - o).abs() / rng,
        "Upper_Shadow": (h - body_top) / rng, "Lower_Shadow": (body_bot - l) / rng,
        "Close_Position": (c - l) / rng, "GapOpen": (o - c.shift(1)) / c.shift(1),
        "IsDoji": ((c - o).abs() / rng < 0.1).astype(float),
        "Spread_Proxy": (h - l) / o}, index=raw.index))

    # F7 -- Regime
    hi252, lo252 = c.rolling(252).max(), c.rolling(252).min()
    up_move, dn_move = h - h.shift(1), l.shift(1) - l
    pdm = up_move.where((up_move > dn_move) & (up_move > 0), 0)
    ndm = dn_move.where((dn_move > up_move) & (dn_move > 0), 0)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    pdi = _ema(pdm, 14) / (atr14 + 1e-9)
    mdi = _ema(ndm, 14) / (atr14 + 1e-9)
    dx = (pdi - mdi).abs() / (pdi + mdi + 1e-9)

    def _hurst(s, n=20):
        def rs(x):
            x = np.asarray(x)
            cs = np.cumsum(x - x.mean())
            spread = cs.max() - cs.min()
            sd = x.std() + 1e-9
            return np.log(spread / sd) / np.log(n) if spread > 0 else 0.5
        return s.rolling(n).apply(rs, raw=True)

    frames.append(pd.DataFrame({
        "Pct_From_52Hi": (c - hi252) / hi252,
        "Pct_From_52Lo": (c - lo252) / lo252,
        "ZScore20": (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9),
        "ZScore60": (c - c.rolling(60).mean()) / (c.rolling(60).std() + 1e-9),
        "Drawdown20": (c - c.rolling(20).max()) / (c.rolling(20).max() + 1e-9),
        "ADX14": _ema(dx, 14), "PlusDI14": pdi, "MinusDI14": mdi,
        "Hurst20": _hurst(c.pct_change().fillna(0), 20)}, index=raw.index))

    # F8 -- Temporal (cyclical calendar encodings)
    idx = pd.DatetimeIndex(raw.index)
    dow, mon = idx.dayofweek.values, idx.month.values
    woy = idx.isocalendar().week.values.astype(float)
    frames.append(pd.DataFrame({
        "DOW_sin": np.sin(2 * np.pi * dow / 5), "DOW_cos": np.cos(2 * np.pi * dow / 5),
        "Month_sin": np.sin(2 * np.pi * mon / 12), "Month_cos": np.cos(2 * np.pi * mon / 12),
        "WOY_sin": np.sin(2 * np.pi * woy / 52), "WOY_cos": np.cos(2 * np.pi * woy / 52),
        "Quarter_sin": np.sin(2 * np.pi * ((mon - 1) // 3) / 4),
        "Quarter_cos": np.cos(2 * np.pi * ((mon - 1) // 3) / 4)}, index=raw.index))

    df = pd.concat(frames, axis=1)

    # F9 -- Interactions
    df["RSI_x_ATR"]     = df["RSI14"] * df["ATR14"]
    df["MACD_x_Vol"]    = df["MACD_hist"] * df["RolStd20"]
    df["VolSurp_x_Ret"] = df["Vol_ZScore"] * df["LogRet1"]
    df["BB_RSI"]        = df["BB_Pct20"] * (df["RSI14"] - 0.5)
    df["ADX_x_MACD"]    = df["ADX14"] * np.sign(df["MACD_line"])
    df["Body_x_VolZ"]   = df["Body_Pct"] * df["Vol_ZScore"]

    return _add_targets(df, df["LogRet1"], direction_threshold)


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return the feature column names of an engineered frame (targets excluded)."""
    ignore = set(TARGET_COLS)
    return [c for c in df.columns if c not in ignore]


# Registry of the feature sets compared by the benchmark (Branch 1 vs Branch 2).
FEATURE_SETS = {
    "ohlc": ohlc_features,
    "finance": finance_features,
}
''')

# ── Cell 11: data loading header ──────────────────────────────────────────────
md(r'''## 5. Data Loading & Panel Construction

Assembles the cross-sectional (date, ticker) panel for a chosen feature set and
the train / validation / test splits.''')

# ── Cell 12: data loading ─────────────────────────────────────────────────────
code(r'''
def load_returns_and_meta():
    """Load the returns matrix, regime labels, correlation matrix and
    cluster/sector maps produced by the data-download step.

    Returns ``(ret, regime_int, corr, cluster, sector, tickers, dates)``.
    """
    ret = pd.read_csv(RETURNS_PATH, index_col=0)
    ret.index = pd.to_datetime(ret.index, errors="coerce")
    ret = ret[~ret.index.isna()].sort_index()
    tickers = list(ret.columns)
    dates = ret.index

    regime_raw = pd.read_csv(REGIME_PATH, index_col=0, parse_dates=True).squeeze()
    if hasattr(regime_raw, "columns") and "rolling_mean_r" in regime_raw.columns:
        regime_raw = regime_raw["rolling_mean_r"]
    regime_raw = regime_raw.reindex(dates).ffill().bfill()
    regime_map = {"low": 0, "medium": 1, "high": 2}
    regime_int = regime_raw.map(regime_map).fillna(1).astype(int)

    corr_path = os.path.join(CORR_DIR, "correlation_matrix_full.csv")
    corr = pd.read_csv(corr_path, index_col=0) if os.path.isfile(corr_path) else ret.corr()
    corr = corr.reindex(index=tickers, columns=tickers).fillna(0)

    cluster_path = os.path.join(CORR_DIR, "phase3_cluster_membership.csv")
    cluster, sector = {}, {}
    if os.path.isfile(cluster_path):
        cm = pd.read_csv(cluster_path)
        for _, row in cm.iterrows():
            t = row.get("ticker")
            if t in tickers:
                cluster[t] = int(row.get("empirical_cluster", 0))
                sector[t]  = str(row.get("GICS_sector", "Other"))
    for t in tickers:
        sector.setdefault(t, TICKER_SECTOR.get(t, "Other"))
        cluster.setdefault(t, 0)

    return ret, regime_int, corr, cluster, sector, tickers, dates


def build_feature_panel(tickers, dates, feature_fn) -> pd.DataFrame:
    """Build a (date, ticker) panel of engineered features and targets using
    ``feature_fn`` (an entry of FEATURE_SETS). Only rows whose date is present
    in ``dates`` are kept."""
    keep = set(pd.DatetimeIndex(pd.to_datetime(pd.Index(dates), errors="coerce")).dropna())
    frames, skipped = [], 0
    for i, sym in enumerate(tickers, 1):
        if i % 50 == 0:
            print(f"  building features {i}/{len(tickers)} ...")
        path = os.path.join(DATA_DIR, f"{sym}_historical.csv")
        if not os.path.isfile(path):
            skipped += 1
            continue
        try:
            raw = pd.read_csv(path, index_col=0)
            raw.index = pd.to_datetime(raw.index, errors="coerce")
            raw = raw[~raw.index.isna()].sort_index()
            if "Close" not in raw.columns or len(raw) < MIN_HISTORY_ROWS:
                skipped += 1
                continue
            feat = feature_fn(raw)
            if feat.empty or "LogReturn_Next" not in feat.columns:
                skipped += 1
                continue
            feat = feat[feat.index.isin(keep)]
            if feat.empty:
                skipped += 1
                continue
            feat = feat.reset_index()
            feat = feat.rename(columns={feat.columns[0]: "date"})
            feat.insert(1, "ticker", sym)
            frames.append(feat)
        except Exception:
            skipped += 1

    if skipped:
        print(f"  skipped {skipped} tickers")
    if not frames:
        raise RuntimeError("No feature rows built. Check the finance_output CSVs.")
    return pd.concat(frames, ignore_index=True)


def zscore_cross_sectional(panel: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Z-score each feature cross-sectionally (across tickers) within each day."""
    out = panel.copy()
    for _, grp in out.groupby("date"):
        for c in feature_cols:
            if c not in grp.columns:
                continue
            s = grp[c].astype(float)
            mu, std = s.mean(), s.std()
            out.loc[grp.index, c] = (s - mu) / std if (std and std > EPS) else 0.0
    return out


def encode_sector(sector_map: dict, tickers: list) -> dict:
    """Deterministically integer-encode sector labels with a LabelEncoder."""
    le = LabelEncoder()
    labels = [sector_map.get(t, "Other") for t in tickers]
    le.fit(labels)
    return {t: int(le.transform([sector_map.get(t, "Other")])[0]) for t in tickers}


def make_splits(panel, feature_cols, regime_ser, sector_enc):
    """Split the panel into train/val/test by date, appending regime and sector
    columns to the feature matrix. Returns three ``get_split`` tuples."""
    p = panel.set_index(["date", "ticker"])
    all_dates = sorted(p.index.get_level_values(0).unique())

    train_dates = [d for d in all_dates if d <= pd.Timestamp(TRAIN_END)]
    val_dates   = [d for d in all_dates if pd.Timestamp(TRAIN_END) < d <= pd.Timestamp(VAL_END)]
    test_dates  = [d for d in all_dates if pd.Timestamp(VAL_END)   < d <= pd.Timestamp(TEST_END)]

    def get_split(date_list):
        """Assemble (X, y_ret, y_dir, y_vol, index) for the given dates."""
        mask = p.index.get_level_values(0).isin(date_list)
        P = p.loc[mask].copy().dropna(subset=feature_cols, how="all")
        X   = P[feature_cols].fillna(0).values.astype(np.float32)
        sec = np.array([sector_enc.get(t, 0) for t in P.index.get_level_values(1)],
                       dtype=np.float32).reshape(-1, 1)
        reg = regime_ser.reindex(P.index.get_level_values(0)).values.reshape(-1, 1).astype(np.float32)
        X   = np.hstack([X, reg, sec])
        y_ret = P["LogReturn_Next"].values  if "LogReturn_Next"  in P.columns else None
        y_dir = P["Direction"].values       if "Direction"       in P.columns else None
        y_vol = P["Volatility_Next"].values if "Volatility_Next" in P.columns else None
        return X, y_ret, y_dir, y_vol, P.index

    return get_split(train_dates), get_split(val_dates), get_split(test_dates)


def daily_rank_ic(dates_arr, y_true, y_pred):
    """Mean daily cross-sectional Spearman rank IC and its ICIR (mean / std)."""
    if y_true is None or y_pred is None:
        return np.nan, np.nan, pd.Series(dtype=float)
    dates, yt, yp = np.asarray(dates_arr), np.asarray(y_true), np.asarray(y_pred)
    daily_ics = {}
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 5:
            continue
        rho, _ = spearmanr(yt[mask], yp[mask])
        if np.isfinite(rho):
            daily_ics[d] = rho
    if not daily_ics:
        return np.nan, np.nan, pd.Series(dtype=float)
    s = pd.Series(daily_ics)
    mean_ic, std_ic = s.mean(), s.std()
    icir = mean_ic / std_ic if (std_ic and std_ic > EPS) else np.nan
    return float(mean_ic), float(icir), s


def merge_train_val(train_split, val_split):
    """Concatenate the train and validation splits into one fit set.

    Every model -- the MLPs (which do their own internal early-stopping) and the
    classical baselines (which do not) -- fits on train+val and is scored on the
    held-out test split, so deep vs classical is a fair, identical-data
    comparison. Returns ``(X_full, y_ret, y_dir, y_vol)``.
    """
    X_tr, yr_tr, yd_tr, yv_tr, _ = train_split
    X_va, yr_va, yd_va, yv_va, _ = val_split
    X_full  = np.vstack([X_tr, X_va])
    yr_full = np.concatenate([yr_tr, yr_va]) if yr_tr is not None else None
    yd_full = np.concatenate([yd_tr, yd_va]) if yd_tr is not None else None
    yv_full = np.concatenate([yv_tr, yv_va]) if yv_tr is not None else None
    return X_full, yr_full, yd_full, yv_full
''')

# ── Cell 13: metrics header ───────────────────────────────────────────────────
md(r'''## 6. Metrics''')

# ── Cell 14: metrics ──────────────────────────────────────────────────────────
code(r'''
def regression_metrics(y_true, y_pred):
    """Return MSE and MAE for a regression target as a dict."""
    if y_true is None or y_pred is None:
        return {}
    yt = np.asarray(y_true).ravel().astype(float)
    yp = np.asarray(y_pred).ravel().astype(float)
    return {
        "mse": float(mean_squared_error(yt, yp)),
        "mae": float(mean_absolute_error(yt, yp)),
    }


def classification_metrics(y_true, y_pred):
    """Return a dict of classifier metrics: accuracy, balanced accuracy, macro
    precision / recall / F1, Matthews correlation (MCC) and AUC.

    Balanced accuracy, F1 and MCC are robust to the class imbalance of the
    Direction target (added in Stage 5)."""
    if y_true is None or y_pred is None:
        return {}
    yt = np.asarray(y_true).ravel().astype(int)
    yp = np.asarray(y_pred).ravel().astype(int)
    try:
        auc = roc_auc_score(yt, yp) if len(np.unique(yt)) == 2 else 0.5
    except Exception:
        auc = 0.5
    return {
        "accuracy":          float(accuracy_score(yt, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
        "precision":         float(precision_score(yt, yp, average="macro", zero_division=0)),
        "recall":            float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1":                float(f1_score(yt, yp, average="macro", zero_division=0)),
        "mcc":               float(matthews_corrcoef(yt, yp)),
        "auc":               float(auc),
    }
''')

# ── Cell 15: deep models header ───────────────────────────────────────────────
md(r'''## 7. Models — Deep (MLP)

Simple feed-forward MLPs plus their training loops.''')

# ── Cell 16: models ───────────────────────────────────────────────────────────
code(r'''
class MLPRegressor(nn.Module):
    """Feed-forward MLP with ReLU hidden layers and a single scalar output."""

    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class MLPClassifier(nn.Module):
    """Feed-forward MLP with ReLU hidden layers and n_classes logit outputs."""

    def __init__(self, input_dim, hidden_sizes, n_classes=3):
        super().__init__()
        layers, last = [], input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_loaders(X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE):
    """Wrap train/val arrays in PyTorch DataLoaders.

    Direction labels arrive as {-1, 0, 1}; they are remapped to {0, 1, 2} so
    they are valid class indices for CrossEntropyLoss.
    """
    X_tr = torch.from_numpy(X_train.astype(np.float32))
    X_va = torch.from_numpy(X_val.astype(np.float32))
    if y_train is None:
        return None, None
    # Remap signed direction labels {-1,0,1} -> {0,1,2} for CrossEntropyLoss.
    if y_train.dtype.kind in "fi" and np.any(y_train < 0):
        y_train = y_train + 1
        y_val   = y_val + 1
    if y_train.dtype.kind in "iu" or np.issubdtype(y_train.dtype, np.integer):
        y_tr = torch.from_numpy(y_train.astype(np.int64))
        y_va = torch.from_numpy(y_val.astype(np.int64))
    else:
        y_tr = torch.from_numpy(y_train.astype(np.float32))
        y_va = torch.from_numpy(y_val.astype(np.float32))
    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=batch_size, shuffle=False)
    return tr_loader, va_loader


def train_regressor(model, train_loader, val_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train a regression MLP with Adam + MSE loss; keep the best-val-MSE weights."""
    model.to(DEVICE)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_state, best_val = None, float("inf")
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        se, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                se += torch.sum((model(xb) - yb) ** 2).item()
                n  += yb.numel()
        val_mse = se / max(n, 1)
        if val_mse < best_val:
            best_val   = val_mse
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model


def train_classifier(model, train_loader, val_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train a classifier MLP with Adam + cross-entropy; keep best-val-loss weights."""
    model.to(DEVICE)
    optim   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_state, best_val = None, float("inf")
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            loss_fn(model(xb), yb).backward()
            optim.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += loss_fn(model(xb), yb).item() * yb.size(0)
                n     += yb.size(0)
        val_loss = total / max(n, 1)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model
''')

# ── Cell 17: classical baselines header ───────────────────────────────────────
md(r'''## 8. Classical ML Baselines

Tree ensembles and linear models — **XGBoost**, **RandomForest**, and **Ridge**
(regression) / **LogisticRegression** (Direction) — trained on the *same* panels
and splits as the MLPs. They establish the non-deep-learning floor and answer:
*do the deep MLPs actually beat off-the-shelf classical ML?*

`benchmark_classical` emits metric rows in the same schema as the deep
benchmark, tagged `model_type="classical"` so the two are directly comparable.''')

# ── Cell 18: classical baselines ──────────────────────────────────────────────
code(r'''
# Classical ML baselines (Stage 2 of the research ladder).
#
# sklearn / xgboost estimators train directly on the numpy arrays the splits
# already produce -- no DataLoaders. Each model fits on train+val (via
# merge_train_val) and is scored on the held-out test split, exactly like the
# deep MLPs, so deep vs classical is a fair, identical-data comparison.

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression

try:
    from xgboost import XGBRegressor, XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    print("xgboost not available -- XGBoost baselines will be skipped.")


def classical_regressors():
    """Name -> factory for the classical regression baselines.

    Each value builds a *fresh* estimator, so the return and volatility targets
    get independent fits. XGBoost is included only when the package imports.
    """
    models = {
        "RandomForest": lambda: RandomForestRegressor(**RF_PARAMS),
        "Ridge":        lambda: Ridge(**RIDGE_PARAMS),
    }
    if _HAS_XGB:
        models["XGBoost"] = lambda: XGBRegressor(**XGB_PARAMS)
    return models


def classical_classifiers():
    """Name -> factory for the classical Direction classifiers (fresh each call)."""
    models = {
        "RandomForest":       lambda: RandomForestClassifier(**RF_PARAMS),
        "LogisticRegression": lambda: LogisticRegression(**LOGREG_PARAMS),
    }
    if _HAS_XGB:
        models["XGBoost"] = lambda: XGBClassifier(**XGB_PARAMS)
    return models


def benchmark_classical(feature_set, splits):
    """Train every classical baseline on each target for one feature set.

    Returns metric rows in the same schema as ``benchmark_feature_set`` (the
    deep MLPs), tagged ``model_type="classical"`` so both flow into one results
    CSV and one comparison report.
    """
    train_split, val_split, test_split = splits
    X_full, yr_full, yd_full, yv_full  = merge_train_val(train_split, val_split)
    X_te, yr_te, yd_te, yv_te, idx_te  = test_split
    test_dates = idx_te.get_level_values(0)
    rows = []

    def record(model_name, target, metrics, extra=None):
        row = {"feature_set": feature_set, "model_name": model_name,
               "model_type": "classical", "target": target}
        if extra:
            row.update(extra)
        if metrics:
            row.update(metrics)
        rows.append(row)

    # -- Regression targets: return (with rank IC) and volatility --
    for target, stub, y_full, y_te, want_ic in [
        ("LogReturn_Next",  "ret", yr_full, yr_te, True),
        ("Volatility_Next", "vol", yv_full, yv_te, False),
    ]:
        if y_full is None:
            continue
        for cname, factory in classical_regressors().items():
            name = f"{cname}_{stub}"
            try:
                print(f"  [REG] {feature_set}/{name} -> {target}")
                mdl = factory()
                mdl.fit(X_full, y_full)
                preds = mdl.predict(X_te)
                met = regression_metrics(y_te, preds)
                if want_ic:
                    ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                    met.update({"return_ic": ic, "return_icir": icir})
                record(name, target, met)
            except Exception as e:
                record(name, target, {}, {"error": str(e)})
                print(f"    ERROR: {e}")

    # -- Direction classification (labels remapped {-1,0,1} -> {0,1,2}) --
    if yd_full is not None:
        yd_full_int = yd_full.astype(int) + 1
        yd_te_int   = yd_te.astype(int) + 1
        for cname, factory in classical_classifiers().items():
            name = f"{cname}_dir"
            try:
                print(f"  [CLF] {feature_set}/{name} -> Direction")
                mdl = factory()
                mdl.fit(X_full, yd_full_int)
                preds = mdl.predict(X_te)
                record(name, "Direction", classification_metrics(yd_te_int, preds))
            except Exception as e:
                record(name, "Direction", {}, {"error": str(e)})
                print(f"    ERROR: {e}")
    return rows
''')

# ── Cell 19: sequence models header ───────────────────────────────────────────
md(r'''## 9. Sequence Models (LSTM / Transformer)

**LSTM** and **Transformer** encoders over a `SEQ_LEN`-day look-back window of
the same per-day features the MLPs use — adding an explicit temporal axis the
cross-sectional MLPs and classical models lack.

Windows are built **lazily** from the z-scored panel (a materialised 3-D tensor
would be multi-GB) and split by each window's last (prediction) date. Like every
other model they fit on train+val and are scored on the held-out test split.

`benchmark_sequence` emits metric rows in the shared schema, tagged
`model_type="sequence"`.''')

# ── Cell 20: sequence models ──────────────────────────────────────────────────
code(r'''
# Sequence models (Stage 3 of the research ladder).

class LSTMNet(nn.Module):
    """LSTM encoder over a feature sequence; the last hidden state feeds a
    regression (n_outputs=1) or classification (n_outputs=n_classes) head."""

    def __init__(self, input_dim, hidden_size=128, num_layers=2,
                 n_outputs=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        out, _ = self.lstm(x)                  # (B, T, H)
        y = self.head(out[:, -1, :])           # last timestep -> (B, n_outputs)
        return y.squeeze(-1) if self.n_outputs == 1 else y


class TransformerNet(nn.Module):
    """Transformer-encoder over a feature sequence with a learnable positional
    embedding; the time-mean-pooled encoding feeds a regression/classification
    head. No causal mask -- every timestep in the window is past data."""

    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=2,
                 n_outputs=1, dropout=0.2, max_len=256):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos  = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        z = self.proj(x) + self.pos[:, :x.size(1), :]
        z = self.encoder(z)                    # (B, T, d_model)
        y = self.head(z.mean(dim=1))           # time-mean pool -> (B, n_outputs)
        return y.squeeze(-1) if self.n_outputs == 1 else y


def build_sequence_model(spec, input_dim, n_outputs):
    """Construct an LSTM or Transformer from a SEQ_SPECS entry."""
    kw = {k: v for k, v in spec.items() if k != "kind"}
    if spec["kind"] == "lstm":
        return LSTMNet(input_dim, n_outputs=n_outputs, **kw)
    if spec["kind"] == "transformer":
        return TransformerNet(input_dim, n_outputs=n_outputs, **kw)
    raise ValueError(f"unknown sequence model kind: {spec['kind']}")


class SequenceWindowDataset(torch.utils.data.Dataset):
    """Lazily yields a (SEQ_LEN, F) feature window and its scalar target.

    Holds one 2-D feature array per ticker; __getitem__ slices a window so the
    full 3-D tensor is never materialised.
    """

    def __init__(self, feat_by_ticker, target_by_ticker, index, seq_len):
        self.feat    = feat_by_ticker
        self.target  = target_by_ticker
        self.index   = index            # list of (ticker, end_row) tuples
        self.seq_len = seq_len

    def __len__(self):
        return len(self.index)

    def __getitem__(self, k):
        ticker, i = self.index[k]
        w = self.feat[ticker][i - self.seq_len + 1: i + 1]
        return torch.from_numpy(w), torch.tensor(self.target[ticker][i])


def build_sequence_panel(panel, feature_cols, regime_ser, sector_enc, seq_len):
    """Turn the z-scored panel into per-ticker feature / target / date arrays
    plus a window index split into train / val / test by each window's last
    (prediction) date.

    The per-timestep feature vector matches the MLP input: feature_cols + regime
    + sector. Returns a dict with keys ``feat``, ``targets``, ``dates``, ``index``.
    """
    feat_by_ticker, targets_by_ticker, dates_by_ticker = {}, {}, {}
    index = {"train": [], "val": [], "test": []}
    train_end = pd.Timestamp(TRAIN_END)
    val_end   = pd.Timestamp(VAL_END)
    test_end  = pd.Timestamp(TEST_END)

    for ticker, grp in panel.groupby("ticker", sort=False):
        grp = grp.sort_values("date")
        if len(grp) < seq_len:
            continue
        feats = grp[feature_cols].fillna(0).values.astype(np.float32)
        dts   = pd.DatetimeIndex(grp["date"].values)
        reg   = (regime_ser.reindex(dts).ffill().bfill().fillna(1)
                 .values.reshape(-1, 1).astype(np.float32))
        sec   = np.full((len(grp), 1), float(sector_enc.get(ticker, 0)),
                        dtype=np.float32)
        feat_by_ticker[ticker]    = np.hstack([feats, reg, sec])
        dates_by_ticker[ticker]   = dts.values
        targets_by_ticker[ticker] = {
            "LogReturn_Next":  grp["LogReturn_Next"].values.astype(np.float32),
            "Volatility_Next": grp["Volatility_Next"].values.astype(np.float32),
            "Direction":       grp["Direction"].values.astype(np.int64),
        }
        for i in range(seq_len - 1, len(grp)):
            d = dts[i]
            if   d <= train_end: index["train"].append((ticker, i))
            elif d <= val_end:   index["val"].append((ticker, i))
            elif d <= test_end:  index["test"].append((ticker, i))

    return {"feat": feat_by_ticker, "targets": targets_by_ticker,
            "dates": dates_by_ticker, "index": index}


def make_sequence_loaders(seq_bundle, target_name, seq_len):
    """Build (fit_loader, val_loader, test_loader, test_dates, y_test) for one
    target. Fit = train+val windows (shuffled); val = the val windows (the
    early-stopping monitor); test = held-out windows. Direction labels are
    remapped {-1,0,1} -> {0,1,2} for CrossEntropyLoss."""
    feat, targets = seq_bundle["feat"], seq_bundle["targets"]
    dates, index  = seq_bundle["dates"], seq_bundle["index"]
    remap = target_name == "Direction"
    tgt = {}
    for t, per_target in targets.items():
        arr = per_target[target_name]
        tgt[t] = (arr.astype(np.int64) + 1) if remap else arr.astype(np.float32)

    def loader(idx_list, shuffle):
        ds = SequenceWindowDataset(feat, tgt, idx_list, seq_len)
        return DataLoader(ds, batch_size=SEQ_BATCH_SIZE, shuffle=shuffle)

    fit_loader  = loader(index["train"] + index["val"], True)
    val_loader  = loader(index["val"],  False)
    test_loader = loader(index["test"], False)
    test_dates  = np.array([dates[t][i] for t, i in index["test"]])
    y_test      = np.array([tgt[t][i]   for t, i in index["test"]])
    return fit_loader, val_loader, test_loader, test_dates, y_test


def benchmark_sequence(feature_set, bundle):
    """Train every SEQ_SPECS sequence model (LSTM / Transformer) on each target
    for one feature set. Returns metric rows tagged ``model_type="sequence"``,
    in the same schema as the deep and classical benchmarks."""
    seq_bundle = bundle["seq_bundle"]
    n_tr, n_te = len(seq_bundle["index"]["train"]), len(seq_bundle["index"]["test"])
    if n_tr == 0 or n_te == 0:
        print(f"  no sequence windows (SEQ_LEN={SEQ_LEN}) -- skipping")
        return []
    input_dim = next(iter(seq_bundle["feat"].values())).shape[1]
    print(f"  sequence windows: {n_tr} train+val, {n_te} test  (SEQ_LEN={SEQ_LEN})")
    rows = []

    def record(model_name, target, metrics, extra=None):
        row = {"feature_set": feature_set, "model_name": model_name,
               "model_type": "sequence", "target": target}
        if extra:
            row.update(extra)
        if metrics:
            row.update(metrics)
        rows.append(row)

    for target, stub, is_clf, want_ic in [
        ("LogReturn_Next",  "ret", False, True),
        ("Volatility_Next", "vol", False, False),
        ("Direction",       "dir", True,  False),
    ]:
        fit_l, val_l, test_l, test_dates, y_te = make_sequence_loaders(
            seq_bundle, target, SEQ_LEN)
        for sname, spec in SEQ_SPECS.items():
            name = f"{sname}_{stub}"
            try:
                print(f"  [{'CLF' if is_clf else 'REG'}] {feature_set}/{name} "
                      f"-> {target}")
                n_out = 3 if is_clf else 1
                model = build_sequence_model(spec, input_dim, n_out)
                trainer = train_classifier if is_clf else train_regressor
                model = trainer(model, fit_l, val_l,
                                n_epochs=SEQ_EPOCHS, lr=LEARNING_RATE)
                model.eval()
                chunks = []
                with torch.no_grad():
                    for xb, _ in test_l:
                        chunks.append(model(xb.to(DEVICE)).cpu().numpy())
                preds = np.concatenate(chunks) if chunks else np.array([])
                if is_clf:
                    preds = np.argmax(preds, axis=1)
                    record(name, target, classification_metrics(y_te, preds))
                else:
                    met = regression_metrics(y_te, preds)
                    if want_ic:
                        ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                        met.update({"return_ic": ic, "return_icir": icir})
                    record(name, target, met)
            except Exception as e:
                record(name, target, {}, {"error": str(e)})
                print(f"    ERROR: {e}")
    return rows
''')

# ── Cell 21: latent world model header ────────────────────────────────────────
md(r'''## 10. Latent World Model

A **latent world model** — the "latent market dynamics" rung of the research
question. Each day's feature vector is encoded by a **VAE** into a latent
distribution; a **GRU latent-transition** module rolls the sampled latent
sequence into a world state and predicts the next latent (a learned dynamics
model); a **decoder** reconstructs the observation; and a **prediction head**
reads the target off the final world state.

Trained with a combined objective — reconstruction + KL (β-VAE) + next-latent
transition + supervised loss. Reuses the sequence look-back windows; rows are
tagged `model_type="worldmodel"`.''')

# ── Cell 22: latent world model ───────────────────────────────────────────────
code(r'''
# Latent world model (Stage 4 of the research ladder).

class WorldModel(nn.Module):
    """VAE encoder -> GRU latent transition -> prediction head.

    forward(x) returns ``(pred, recon, mu, logvar, z, z_next_pred)``:
      * per-timestep VAE encoder      x_t      -> q(z_t | x_t) = N(mu_t, sigma_t)
      * GRU latent-transition module  z_1..z_T -> world states h_1..h_T
      * next-latent predictor         h_t      -> z_{t+1}   (learned dynamics)
      * decoder                       z_t      -> x_hat_t   (reconstruction)
      * prediction head               h_T      -> target
    """

    def __init__(self, input_dim, latent_dim=24, hidden=128, n_outputs=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_outputs  = n_outputs
        # VAE encoder: per-timestep observation -> latent distribution
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.to_mu     = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        # Latent transition: GRU rolls the latent sequence into a world state
        self.transition = nn.GRU(latent_dim, hidden, batch_first=True)
        self.to_next    = nn.Linear(hidden, latent_dim)   # predicts the next latent
        # Observation decoder
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim))
        # Prediction head off the final world state
        self.head = nn.Linear(hidden, n_outputs)

    def reparameterize(self, mu, logvar):
        """Sample z ~ N(mu, sigma) when training; use the mean at eval time."""
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h = self.enc(x)                                # (B, T, hidden)
        mu, logvar = self.to_mu(h), self.to_logvar(h)  # (B, T, latent)
        z = self.reparameterize(mu, logvar)            # (B, T, latent)
        recon = self.dec(z)                            # (B, T, input_dim)
        world, _ = self.transition(z)                  # (B, T, hidden)
        z_next_pred = self.to_next(world[:, :-1, :])    # (B, T-1, latent) -> z[1:]
        pred = self.head(world[:, -1, :])              # (B, n_outputs)
        pred = pred.squeeze(-1) if self.n_outputs == 1 else pred
        return pred, recon, mu, logvar, z, z_next_pred


def train_world_model(model, train_loader, val_loader, is_clf,
                      n_epochs=WM_EPOCHS, lr=LEARNING_RATE):
    """Train the world model on the combined ELBO + transition + supervised
    objective; keep the weights with the best validation supervised loss
    (the benchmarked objective)."""
    model.to(DEVICE)
    optim    = torch.optim.Adam(model.parameters(), lr=lr)
    sup_loss = nn.CrossEntropyLoss() if is_clf else nn.MSELoss()
    best_state, best_val = None, float("inf")
    for _ in range(n_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred, recon, mu, logvar, z, z_next = model(xb)
            recon_l = ((recon - xb) ** 2).mean()
            kl_l    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            trans_l = ((z_next - z[:, 1:, :].detach()) ** 2).mean()
            sup_l   = sup_loss(pred, yb)
            loss = (WM_SUP_WEIGHT * sup_l + recon_l
                    + WM_BETA * kl_l + WM_GAMMA * trans_l)
            loss.backward()
            optim.step()
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                total += sup_loss(model(xb)[0], yb).item() * yb.size(0)
                n     += yb.size(0)
        val = total / max(n, 1)
        if val < best_val:
            best_val, best_state = val, model.state_dict()
    if best_state:
        model.load_state_dict(best_state)
    return model


def benchmark_worldmodel(feature_set, bundle):
    """Stage 4: train the latent world model on each target for one feature set,
    reusing the sequence look-back windows. One model per target; rows tagged
    ``model_type="worldmodel"`` in the shared metric schema."""
    seq_bundle = bundle["seq_bundle"]
    n_tr = len(seq_bundle["index"]["train"])
    n_te = len(seq_bundle["index"]["test"])
    if n_tr == 0 or n_te == 0:
        print(f"  no sequence windows (SEQ_LEN={SEQ_LEN}) -- world model skipped")
        return []
    input_dim = next(iter(seq_bundle["feat"].values())).shape[1]
    rows = []

    def record(model_name, target, metrics, extra=None):
        row = {"feature_set": feature_set, "model_name": model_name,
               "model_type": "worldmodel", "target": target}
        if extra:
            row.update(extra)
        if metrics:
            row.update(metrics)
        rows.append(row)

    for target, stub, is_clf, want_ic in [
        ("LogReturn_Next",  "ret", False, True),
        ("Volatility_Next", "vol", False, False),
        ("Direction",       "dir", True,  False),
    ]:
        fit_l, val_l, test_l, test_dates, y_te = make_sequence_loaders(
            seq_bundle, target, SEQ_LEN)
        name = f"WorldModel_{stub}"
        try:
            print(f"  [{'CLF' if is_clf else 'REG'}] {feature_set}/{name} -> {target}")
            n_out = 3 if is_clf else 1
            model = WorldModel(input_dim, WM_LATENT_DIM, WM_HIDDEN, n_out)
            model = train_world_model(model, fit_l, val_l, is_clf,
                                      n_epochs=WM_EPOCHS, lr=LEARNING_RATE)
            model.eval()
            chunks = []
            with torch.no_grad():
                for xb, _ in test_l:
                    chunks.append(model(xb.to(DEVICE))[0].cpu().numpy())
            preds = np.concatenate(chunks) if chunks else np.array([])
            if is_clf:
                preds = np.argmax(preds, axis=1)
                record(name, target, classification_metrics(y_te, preds))
            else:
                met = regression_metrics(y_te, preds)
                if want_ic:
                    ic, icir, _ = daily_rank_ic(test_dates, y_te, preds)
                    met.update({"return_ic": ic, "return_icir": icir})
                record(name, target, met)
        except Exception as e:
            record(name, target, {}, {"error": str(e)})
            print(f"    ERROR: {e}")
    return rows
''')

# ── Cell 23: run benchmark header ─────────────────────────────────────────────
md(r'''## 11. Run Benchmark

For each feature set in `FEATURE_SETS` (pure-OHLC and finance-informed), builds
the panel and trains every classical baseline, deep MLP (`MODEL_SPECS`), sequence
model (`SEQ_SPECS`) and the latent world model on all three targets. Prints an
**OHLC-vs-finance comparison** and a **model-type comparison**, then saves a
tagged results CSV plus plots under `RESULTS_DIR`.''')

# ── Cell 24: run benchmark ────────────────────────────────────────────────────
code(r'''
def build_panel_and_splits(feature_fn, sector_map, regime_int, tickers, dates):
    """Build and z-score the panel for one feature set. Returns the
    (train, val, test) splits plus a ``bundle`` holding the pre-built sequence
    look-back windows that the sequence models and world model both reuse."""
    panel = build_feature_panel(tickers, dates, feature_fn)
    feature_cols = [c for c in get_feature_cols(panel) if c not in ("date", "ticker")]
    panel = zscore_cross_sectional(panel, feature_cols)
    sector_enc  = encode_sector(sector_map, tickers)
    panel_dates = sorted(panel["date"].unique())
    regime_ser  = regime_int.reindex(panel_dates).ffill().bfill().fillna(1).astype(int)
    splits = make_splits(panel, feature_cols, regime_ser, sector_enc)
    seq_bundle = build_sequence_panel(panel, feature_cols, regime_ser,
                                      sector_enc, SEQ_LEN)
    bundle = {"seq_bundle": seq_bundle, "n_feat": len(feature_cols)}
    return splits, bundle


def benchmark_feature_set(feature_set, splits):
    """Train every MODEL_SPECS deep MLP on each target for one feature set;
    return a list of metric rows tagged with the feature-set name."""
    train_split, val_split, test_split = splits
    X_va, yr_va, yd_va, yv_va, _      = val_split
    X_te, yr_te, yd_te, yv_te, idx_te = test_split
    X_full, yr_full, yd_full, yv_full = merge_train_val(train_split, val_split)
    input_dim = X_full.shape[1]
    rows = []

    def record(model_name, target, metrics, extra=None):
        row = {"feature_set": feature_set, "model_name": model_name,
               "model_type": "deep", "target": target}
        if extra:
            row.update(extra)
        if metrics:
            row.update(metrics)
        rows.append(row)

    # -- Return regression --
    if yr_full is not None:
        for size, hidden in MODEL_SPECS.items():
            name = f"MLPReg_ret_{size}"
            try:
                print(f"  [REG] {feature_set}/{name} -> LogReturn_Next")
                tr_l, va_l = make_loaders(X_full, yr_full, X_va, yr_va, BATCH_SIZE)
                mdl = train_regressor(MLPRegressor(input_dim, hidden), tr_l, va_l)
                mdl.eval()
                with torch.no_grad():
                    preds = mdl(torch.from_numpy(X_te).to(DEVICE)).cpu().numpy()
                ic, icir, _ = daily_rank_ic(idx_te.get_level_values(0), yr_te, preds)
                met = regression_metrics(yr_te, preds)
                met.update({"return_ic": ic, "return_icir": icir})
                record(name, "LogReturn_Next", met)
            except Exception as e:
                record(name, "LogReturn_Next", {}, {"error": str(e)})
                print(f"    ERROR: {e}")

    # -- Volatility regression --
    if yv_full is not None:
        for size, hidden in MODEL_SPECS.items():
            name = f"MLPReg_vol_{size}"
            try:
                print(f"  [REG] {feature_set}/{name} -> Volatility_Next")
                tr_l, va_l = make_loaders(X_full, yv_full, X_va, yv_va, BATCH_SIZE)
                mdl = train_regressor(MLPRegressor(input_dim, hidden), tr_l, va_l)
                mdl.eval()
                with torch.no_grad():
                    preds = mdl(torch.from_numpy(X_te).to(DEVICE)).cpu().numpy()
                record(name, "Volatility_Next", regression_metrics(yv_te, preds))
            except Exception as e:
                record(name, "Volatility_Next", {}, {"error": str(e)})
                print(f"    ERROR: {e}")

    # -- Direction classification (labels remapped {-1,0,1} -> {0,1,2}) --
    if yd_full is not None:
        yd_full_int = yd_full.astype(int) + 1
        yd_va_int   = yd_va.astype(int) + 1
        yd_te_int   = yd_te.astype(int) + 1
        for size, hidden in MODEL_SPECS.items():
            name = f"MLPClf_dir_{size}"
            try:
                print(f"  [CLF] {feature_set}/{name} -> Direction")
                tr_l, va_l = make_loaders(X_full, yd_full_int, X_va, yd_va_int, BATCH_SIZE)
                mdl = train_classifier(MLPClassifier(input_dim, hidden, n_classes=3), tr_l, va_l)
                mdl.eval()
                with torch.no_grad():
                    logits = mdl(torch.from_numpy(X_te).to(DEVICE)).cpu().numpy()
                    preds  = np.argmax(logits, axis=1)
                record(name, "Direction", classification_metrics(yd_te_int, preds))
            except Exception as e:
                record(name, "Direction", {}, {"error": str(e)})
                print(f"    ERROR: {e}")
    return rows


def run_gpu_benchmark():
    """Run the benchmark: for each feature set, build the panel, train every
    classical, MLP and sequence model on all three targets, save a tagged
    results CSV, and print the feature-set and model-type comparisons."""
    print(f"GPU multi-model benchmark  (device={DEVICE})")
    print("\nLoading returns / regime / sector metadata ...")
    _ret, regime_int, _corr, _cluster, sector_map, tickers, dates = load_returns_and_meta()
    print(f"  tickers: {len(tickers)},  dates: {len(dates)}")

    all_rows = []
    for feature_set, feature_fn in FEATURE_SETS.items():
        print(f"\n=== Feature set: {feature_set} ===")
        try:
            splits, bundle = build_panel_and_splits(
                feature_fn, sector_map, regime_int, tickers, dates)
            print(f"  {bundle['n_feat']} features -> training models ...")
            all_rows += benchmark_feature_set(feature_set, splits)
            all_rows += benchmark_classical(feature_set, splits)
            all_rows += benchmark_sequence(feature_set, bundle)
            all_rows += benchmark_worldmodel(feature_set, bundle)
        except Exception as e:
            print(f"  feature set '{feature_set}' failed: {e}")

    if not all_rows:
        print("\nNo results produced.")
        return

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULTS_DIR, f"benchmark_{ts}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")

    # -- Feature-set comparison: best model per target -------------------------
    print("\n" + "-" * 64)
    print("FEATURE-SET COMPARISON  (best model per target)")
    print("-" * 64)
    for target, metric, higher in [("LogReturn_Next",  "return_ic", True),
                                   ("Direction",       "accuracy", True),
                                   ("Volatility_Next", "mse",      False)]:
        if metric not in df.columns:
            continue
        print(f"  {target}  [{metric}, {'higher' if higher else 'lower'} better]")
        best = {}
        for fs in FEATURE_SETS:
            sub = df[(df["target"] == target) & (df["feature_set"] == fs)]
            sub = sub.dropna(subset=[metric])
            if sub.empty:
                print(f"    {fs:<10s}  n/a")
                continue
            row = sub.sort_values(metric, ascending=not higher).iloc[0]
            best[fs] = row[metric]
            print(f"    {fs:<10s}  {row[metric]:+.4f}  ({row['model_name']})")
        if len(best) == 2:
            a, b = list(best)
            winner = b if ((best[b] - best[a] > 0) == higher) else a
            print(f"    -> {winner} feature set wins")

    # -- Model-type comparison: best of each model type, per target -----------
    if "model_type" in df.columns:
        print("\n" + "-" * 64)
        print("MODEL-TYPE COMPARISON  (best of each model type, per target)")
        print("-" * 64)
        mt_order = [mt for mt in ("classical", "deep", "sequence", "worldmodel")
                    if mt in set(df["model_type"].dropna())]
        for target, metric, higher in [("LogReturn_Next",  "return_ic", True),
                                       ("Direction",       "accuracy", True),
                                       ("Volatility_Next", "mse",      False)]:
            if metric not in df.columns:
                continue
            print(f"  {target}  [{metric}, {'higher' if higher else 'lower'} better]")
            best = {}
            for mt in mt_order:
                sub = df[(df["target"] == target) & (df["model_type"] == mt)]
                sub = sub.dropna(subset=[metric])
                if sub.empty:
                    print(f"    {mt:<10s}  n/a")
                    continue
                row = sub.sort_values(metric, ascending=not higher).iloc[0]
                best[mt] = row[metric]
                print(f"    {mt:<10s}  {row[metric]:+.4f}  "
                      f"({row['feature_set']}/{row['model_name']})")
            if best:
                winner = (max if higher else min)(best, key=best.get)
                print(f"    -> {winner} wins")

    # -- Summary plots ---------------------------------------------------------
    df["label"] = df["feature_set"] + "/" + df["model_name"]

    def plot_bar(df_sub, metric, title, fname):
        """Save a horizontal bar chart of models ranked by metric."""
        if metric not in df_sub.columns:
            return
        dfm = df_sub.dropna(subset=[metric]).sort_values(metric, ascending=False)
        if dfm.empty:
            return
        plt.figure(figsize=(8, max(4, 0.4 * len(dfm))))
        plt.barh(dfm["label"], dfm[metric])
        plt.xlabel(metric)
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        out_path = os.path.join(RESULTS_DIR, fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved -> {out_path}")

    plot_bar(df[df["target"] == "LogReturn_Next"], "return_ic",
             "Return IC -- OHLC vs finance", "cmp_ret_ic.png")
    plot_bar(df[df["target"] == "Direction"], "accuracy",
             "Direction accuracy -- OHLC vs finance", "cmp_dir_acc.png")
    df_vol = df[df["target"] == "Volatility_Next"].copy()
    if not df_vol.empty and "mse" in df_vol.columns:
        df_vol["neg_mse"] = -df_vol["mse"]
        plot_bar(df_vol, "neg_mse", "Volatility -MSE -- OHLC vs finance", "cmp_vol_mse.png")

    print("\nDone.")


run_gpu_benchmark()
''')

# ── Cell 25: walk-forward evaluation header ───────────────────────────────────
md(r'''## 12. Walk-Forward & Regime Evaluation

**Stage 5** — rigorous, time-aware evaluation. The single train/val/test split
above is one snapshot; this section re-trains a representative model (**XGBoost**
on **Direction**) on **expanding walk-forward folds** spanning the full history,
so performance is measured across many out-of-sample periods.

Each test day is labelled **bull / sideways / volatile** from the market's trend
and volatility, giving **regime-split** metrics — the only setting where regime
testing is meaningful here, since the single-split test window does not reach
back to past crises such as COVID. Reports per-fold and per-regime **balanced
accuracy / F1 / MCC** plus a **Brier** calibration score.''')

# ── Cell 26: walk-forward evaluation ──────────────────────────────────────────
code(r'''
# Walk-forward validation, regime-split testing and calibration (Stage 5).

def label_market_regimes(ret):
    """Label each trading day bull / sideways / volatile from the equal-weight
    market return: high rolling volatility -> 'volatile' (captures crises such
    as the COVID crash), else a positive quarterly trend -> 'bull', else
    'sideways'."""
    mkt   = ret.mean(axis=1)            # equal-weight market daily return
    trend = mkt.rolling(63).sum()       # ~quarterly cumulative return
    vol   = mkt.rolling(21).std()       # ~monthly volatility
    labels = pd.Series("sideways", index=ret.index)
    labels[trend > 0.03] = "bull"
    labels[vol > vol.quantile(0.80)] = "volatile"   # volatile overrides trend
    return labels


def make_walkforward_folds(all_dates, n_folds, min_train_frac):
    """Expanding-window folds: reserve the first ``min_train_frac`` of the
    timeline as the initial training floor, then tile the remainder into
    ``n_folds`` contiguous test blocks. Fold i trains on every date before its
    test block begins."""
    all_dates = sorted(pd.to_datetime(all_dates))
    floor  = int(len(all_dates) * min_train_frac)
    blocks = [b for b in np.array_split(all_dates[floor:], n_folds) if len(b)]
    folds = []
    for blk in blocks:
        test_dates  = list(pd.to_datetime(blk))
        train_dates = [d for d in all_dates if d < test_dates[0]]
        if train_dates and test_dates:
            folds.append((train_dates, test_dates))
    return folds


def wf_features(panel, feature_cols, regime_ser, sector_enc, date_set):
    """Assemble (X, y_dir, dates) for the panel rows on the given dates -- the
    same feature layout (feature_cols + regime + sector) the benchmark uses.
    Direction labels are remapped {-1,0,1} -> {0,1,2}."""
    sub = panel[panel["date"].isin(date_set)]
    if sub.empty:
        return (np.empty((0, len(feature_cols) + 2), np.float32),
                np.empty(0, int), np.empty(0))
    X   = sub[feature_cols].fillna(0).values.astype(np.float32)
    reg = (regime_ser.reindex(sub["date"]).ffill().bfill().fillna(1)
           .values.reshape(-1, 1).astype(np.float32))
    sec = (sub["ticker"].map(lambda t: sector_enc.get(t, 0))
           .values.reshape(-1, 1).astype(np.float32))
    X = np.hstack([X, reg, sec])
    y = sub["Direction"].values.astype(int) + 1
    return X, y, sub["date"].values


def run_walkforward():
    """Stage 5: walk-forward validation + regime-split testing + calibration for
    a representative model (XGBoost on the Direction target).

    Re-trains on expanding folds spanning the data, labels every test day
    bull / sideways / volatile, and reports per-fold metrics, per-regime metrics
    and a Brier calibration score."""
    print("=" * 64)
    print("STAGE 5 -- WALK-FORWARD & REGIME EVALUATION  (XGBoost / Direction)")
    print("=" * 64)
    if not _HAS_XGB:
        print("xgboost unavailable -- walk-forward skipped.")
        return

    ret, regime_int, _corr, _cluster, sector_map, tickers, dates = load_returns_and_meta()
    regimes    = label_market_regimes(ret)
    sector_enc = encode_sector(sector_map, tickers)

    fold_rows, regime_acc = [], []
    for feature_set, feature_fn in FEATURE_SETS.items():
        print(f"\n=== Feature set: {feature_set} ===")
        panel = build_feature_panel(tickers, dates, feature_fn)
        feature_cols = [c for c in get_feature_cols(panel)
                        if c not in ("date", "ticker")]
        panel = zscore_cross_sectional(panel, feature_cols)
        panel_dates = sorted(panel["date"].unique())
        regime_ser  = regime_int.reindex(panel_dates).ffill().bfill().fillna(1).astype(int)
        folds = make_walkforward_folds(panel_dates, WF_N_FOLDS, WF_MIN_TRAIN_FRAC)

        for fi, (train_dates, test_dates) in enumerate(folds, 1):
            X_tr, y_tr, _    = wf_features(panel, feature_cols, regime_ser,
                                           sector_enc, set(train_dates))
            X_te, y_te, d_te = wf_features(panel, feature_cols, regime_ser,
                                           sector_enc, set(test_dates))
            if len(X_tr) == 0 or len(X_te) == 0:
                continue
            try:
                mdl = XGBClassifier(**XGB_PARAMS)
                mdl.fit(X_tr, y_tr)
                proba = mdl.predict_proba(X_te)
                preds = np.argmax(proba, axis=1)
                met   = classification_metrics(y_te, preds)
                # Brier score: mean squared error of the predicted probabilities
                onehot = np.eye(proba.shape[1])[y_te]
                brier  = float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
            except Exception as e:
                print(f"  fold {fi}: ERROR {e}")
                continue
            row = {"feature_set": feature_set, "fold": fi,
                   "test_start": str(pd.Timestamp(min(test_dates)).date()),
                   "test_end":   str(pd.Timestamp(max(test_dates)).date()),
                   "n_test": int(len(y_te)), "brier": brier}
            row.update(met)
            fold_rows.append(row)
            print(f"  fold {fi}  {row['test_start']}..{row['test_end']}  "
                  f"acc={met['accuracy']:.3f}  bal_acc={met['balanced_accuracy']:.3f}  "
                  f"mcc={met['mcc']:+.3f}  brier={brier:.3f}")
            reg_lbl = regimes.reindex(pd.DatetimeIndex(d_te)).fillna("sideways").values
            regime_acc.append((y_te, preds, reg_lbl))

    if not fold_rows:
        print("\nNo walk-forward folds produced.")
        return

    wf_df = pd.DataFrame(fold_rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wf_path = os.path.join(RESULTS_DIR, f"walkforward_{ts}.csv")
    wf_df.to_csv(wf_path, index=False)
    print(f"\nWalk-forward results saved -> {wf_path}")

    # -- Walk-forward summary: mean / std across folds ------------------------
    print("\n" + "-" * 64)
    print("WALK-FORWARD SUMMARY  (mean +/- std across folds)")
    print("-" * 64)
    for fs in FEATURE_SETS:
        sub = wf_df[wf_df["feature_set"] == fs]
        if sub.empty:
            continue
        print(f"  {fs:<10s}  "
              f"acc={sub['accuracy'].mean():.3f}+/-{sub['accuracy'].std():.3f}  "
              f"bal_acc={sub['balanced_accuracy'].mean():.3f}  "
              f"mcc={sub['mcc'].mean():+.3f}  brier={sub['brier'].mean():.3f}")

    # -- Regime-split: metrics pooled across folds, by market regime ---------
    print("\n" + "-" * 64)
    print("REGIME-SPLIT TESTING  (pooled across walk-forward folds)")
    print("-" * 64)
    Y = np.concatenate([a[0] for a in regime_acc])
    P = np.concatenate([a[1] for a in regime_acc])
    R = np.concatenate([a[2] for a in regime_acc])
    for reg in ("bull", "sideways", "volatile"):
        m = R == reg
        if m.sum() == 0:
            print(f"  {reg:<10s}  (no test days)")
            continue
        met = classification_metrics(Y[m], P[m])
        print(f"  {reg:<10s}  n={int(m.sum()):>8d}  acc={met['accuracy']:.3f}  "
              f"bal_acc={met['balanced_accuracy']:.3f}  mcc={met['mcc']:+.3f}  "
              f"f1={met['f1']:.3f}")

    # -- Plot: accuracy across folds -----------------------------------------
    try:
        plt.figure(figsize=(8, 4))
        for fs in FEATURE_SETS:
            sub = wf_df[wf_df["feature_set"] == fs]
            if not sub.empty:
                plt.plot(sub["fold"], sub["accuracy"], marker="o", label=fs)
        plt.xlabel("walk-forward fold")
        plt.ylabel("accuracy")
        plt.title("Walk-forward accuracy by fold -- XGBoost / Direction")
        plt.legend()
        plt.tight_layout()
        p = os.path.join(RESULTS_DIR, "walkforward_accuracy.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved -> {p}")
    except Exception as e:
        print(f"  plot skipped: {e}")

    print("\nWalk-forward evaluation done.")


run_walkforward()
''')

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for i, (ctype, src) in enumerate(cells, 1):
    src = src.strip("\n")
    lines = src.splitlines(keepends=True)
    cell = {"cell_type": ctype, "id": f"cell{i:02d}", "metadata": {}, "source": lines}
    if ctype == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    nb["cells"].append(cell)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "DS2010_Final.ipynb")
with open(out_path, "w") as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print(f"Wrote notebook with {len(nb['cells'])} cells "
      f"({sum(c['cell_type']=='code' for c in nb['cells'])} code, "
      f"{sum(c['cell_type']=='markdown' for c in nb['cells'])} markdown).")
