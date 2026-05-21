"""Ticker universe, OHLCV download, and metadata loading.

``download_data`` needs internet — run it on a cluster login node (or anywhere
with network access) before submitting the GPU job. It caches one CSV per
ticker, so re-runs and the offline GPU job just read the cache.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .logging_utils import get_logger

# --- Ticker universe (12 sectors) --------------------------------------------
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


def build_universe() -> tuple[list[str], dict[str, str]]:
    """Flatten ``TICKER_UNIVERSE`` into a deduplicated ticker list and a
    ticker -> sector map (first occurrence wins)."""
    tickers: list[str] = []
    sector: dict[str, str] = {}
    for sec, syms in TICKER_UNIVERSE.items():
        for t in syms:
            if t not in sector:
                tickers.append(t)
                sector[t] = sec
    return tickers, sector


def fetch_ohlcv(ticker: str, start: str, retries: int = 3) -> pd.DataFrame:
    """Download one ticker's OHLCV history, retrying transient API errors."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def download_data(cfg) -> None:
    """Download OHLCV for every ticker, cache one CSV each, then write the
    returns matrix and (placeholder) regime labels. Cached CSVs are reused, so
    re-running only fetches what is missing."""
    log = get_logger()
    cfg.ensure_dirs()
    tickers, _sector = build_universe()
    log.info("downloading OHLCV for %d tickers -> %s", len(tickers), cfg.raw_dir)

    closes: dict[str, pd.Series] = {}
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        path = cfg.raw_dir / f"{ticker}.csv"
        try:
            if path.is_file():
                df = pd.read_csv(path, index_col=0)
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()].sort_index()
            else:
                df = fetch_ohlcv(ticker, cfg.data_start_date)
                df.to_csv(path)
                time.sleep(cfg.download_pause)   # be gentle with the API
            if "Close" not in df.columns or len(df) < cfg.min_history_rows:
                failed.append(ticker)
            else:
                closes[ticker] = df["Close"].astype(float)
        except Exception as ex:               # noqa: BLE001
            failed.append(ticker)
            log.warning("  fail %s: %s", ticker, ex)
        if i % 25 == 0:
            log.info("  %d/%d processed ...", i, len(tickers))

    log.info("downloaded %d/%d tickers (%d skipped/failed)",
             len(closes), len(tickers), len(failed))
    if not closes:
        raise RuntimeError("No ticker data downloaded — check network access.")

    # Returns matrix: daily log returns, one column per ticker.
    close_df = pd.DataFrame(closes).sort_index()
    ret_matrix = np.log(close_df / close_df.shift(1))
    ret_matrix.to_csv(cfg.returns_path)
    log.info("saved returns matrix -> %s  shape=%s", cfg.returns_path, ret_matrix.shape)

    # Placeholder regime labels (the real market regimes are computed at
    # walk-forward time by finbench.walkforward.label_market_regimes).
    pd.Series(["medium"] * len(ret_matrix.index), index=ret_matrix.index,
              name="regime").to_csv(cfg.regime_path)
    log.info("saved regime labels  -> %s", cfg.regime_path)


def load_returns_and_meta(cfg):
    """Load the returns matrix, regime labels and sector map.

    Returns ``(ret, regime_int, sector_map, tickers, dates)``.
    """
    if not cfg.returns_path.is_file():
        raise FileNotFoundError(
            f"{cfg.returns_path} not found — run scripts/01_download_data.py first."
        )
    ret = pd.read_csv(cfg.returns_path, index_col=0)
    ret.index = pd.to_datetime(ret.index, errors="coerce")
    ret = ret[~ret.index.isna()].sort_index()
    tickers = list(ret.columns)
    dates = ret.index

    regime_raw = pd.read_csv(cfg.regime_path, index_col=0).squeeze("columns")
    regime_raw = regime_raw.reindex(dates).ffill().bfill()
    regime_map = {"low": 0, "medium": 1, "high": 2}
    regime_int = regime_raw.map(regime_map).fillna(1).astype(int)

    _all_tickers, sector_all = build_universe()
    sector_map = {t: sector_all.get(t, "Other") for t in tickers}
    return ret, regime_int, sector_map, tickers, dates
