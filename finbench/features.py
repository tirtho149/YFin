"""Feature engineering — the two feature representations compared by the benchmark.

* ``ohlc_features``    — Branch 1: ~9 pure price/volume features (no indicators).
* ``finance_features`` — Branch 2: 91 technical indicators (financial priors).

Both emit the same three forward-looking targets so results are comparable:
  * ``Direction``       — {-1, 0, 1}: next close down / flat / up (PRIMARY task)
  * ``LogReturn_Next``  — next-day log return
  * ``Volatility_Next`` — forward 5-day std of log returns
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12
TARGET_COLS = ["LogReturn_Next", "Direction", "Volatility_Next"]
DEFAULT_DIRECTION_THRESHOLD = 0.005


# --- small indicator helpers -------------------------------------------------
def _ema(s: pd.Series, n: int) -> pd.Series:
    """Exponential moving average."""
    return s.ewm(span=n, adjust=False).mean()


def _sma(s: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return s.rolling(n).mean()


def _true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's True Range from an OHLC frame."""
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _add_targets(out: pd.DataFrame, logret: pd.Series,
                 direction_threshold: float) -> pd.DataFrame:
    """Attach the three forward-looking targets and drop warm-up / tail NaN rows."""
    out["LogReturn_Next"] = logret.shift(-1)
    out["Volatility_Next"] = logret.shift(-1).rolling(5).std()
    nxt = out["LogReturn_Next"]
    out["Direction"] = np.where(
        nxt > direction_threshold, 1,
        np.where(nxt < -direction_threshold, -1, 0),
    )
    return out.dropna()


# --- Branch 1: pure OHLC -----------------------------------------------------
def ohlc_features(raw: pd.DataFrame,
                  direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD) -> pd.DataFrame:
    """Branch 1 — compact price/volume features only (no technical indicators)."""
    df = raw.copy()
    if "Close" not in df.columns:
        return pd.DataFrame()
    df = df.sort_index()
    c = df["Close"].astype(float)
    logret = np.log(c / c.shift(1))
    out = pd.DataFrame(index=df.index)

    out["Ret1"] = logret.shift(1)
    out["Ret2"] = logret.shift(2)
    out["Ret5"] = logret.shift(5)
    out["Mom10"] = c / c.shift(10) - 1.0
    out["Mom20"] = c / c.shift(20) - 1.0
    out["Vol5"] = logret.rolling(5).std().shift(1)
    out["Vol20"] = logret.rolling(20).std().shift(1)
    out["Range1"] = (
        ((df["High"].astype(float) - df["Low"].astype(float)) / c).shift(1)
        if "High" in df.columns and "Low" in df.columns else np.nan
    )
    out["VolChg"] = out["Vol5"] / (out["Vol20"] + EPS) - 1.0
    return _add_targets(out, logret, direction_threshold)


# --- Branch 2: finance-informed (91 indicators) ------------------------------
def finance_features(raw: pd.DataFrame,
                     direction_threshold: float = DEFAULT_DIRECTION_THRESHOLD) -> pd.DataFrame:
    """Branch 2 — 91 technical indicators across 9 families."""
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

    # F1 — Returns
    frames.append(pd.DataFrame({
        "Ret1": c.pct_change(), "LogRet1": logret,
        "LogRet2": np.log(c / c.shift(2)), "LogRet5": np.log(c / c.shift(5)),
        "LogRet10": np.log(c / c.shift(10)), "LogRet20": np.log(c / c.shift(20)),
        "GapRet": np.log(o / c.shift(1)), "IntradayRet": np.log(c / o),
        "Cum5Ret": c.pct_change(5)}, index=raw.index))

    # F2 — Trend (moving averages used only to derive scale-free distances)
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
    dist["Cross_EMA5_EMA20"] = (ma["EMA5"] - ma["EMA20"]) / c
    dist["Cross_EMA10_EMA50"] = (ma["EMA10"] - ma["EMA50"]) / c
    dist["Cross_SMA50_SMA200"] = (ma["SMA50"] - ma["SMA200"]) / c
    dist["Dist_VWMA20"] = (c - ma["VWMA20"]) / c
    frames.append(pd.DataFrame(dist, index=raw.index))

    # F3 — Momentum
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

    # F4 — Volatility
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

    # F5 — Volume
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

    # F6 — Microstructure
    rng = h - l + 1e-9
    body_top = pd.concat([c, o], axis=1).max(axis=1)
    body_bot = pd.concat([c, o], axis=1).min(axis=1)
    frames.append(pd.DataFrame({
        "HL_Range": (h - l) / c, "Body_Pct": (c - o).abs() / rng,
        "Upper_Shadow": (h - body_top) / rng, "Lower_Shadow": (body_bot - l) / rng,
        "Close_Position": (c - l) / rng, "GapOpen": (o - c.shift(1)) / c.shift(1),
        "IsDoji": ((c - o).abs() / rng < 0.1).astype(float),
        "Spread_Proxy": (h - l) / o}, index=raw.index))

    # F7 — Regime
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

    # F8 — Temporal (cyclical calendar encodings)
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

    # F9 — Interactions
    df["RSI_x_ATR"] = df["RSI14"] * df["ATR14"]
    df["MACD_x_Vol"] = df["MACD_hist"] * df["RolStd20"]
    df["VolSurp_x_Ret"] = df["Vol_ZScore"] * df["LogRet1"]
    df["BB_RSI"] = df["BB_Pct20"] * (df["RSI14"] - 0.5)
    df["ADX_x_MACD"] = df["ADX14"] * np.sign(df["MACD_line"])
    df["Body_x_VolZ"] = df["Body_Pct"] * df["Vol_ZScore"]

    return _add_targets(df, df["LogRet1"], direction_threshold)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the feature column names of an engineered frame (targets excluded)."""
    ignore = set(TARGET_COLS) | {"date", "ticker"}
    return [c for c in df.columns if c not in ignore]


# Registry of the feature sets compared by the benchmark (Branch 1 vs Branch 2).
FEATURE_SETS = {
    "ohlc": ohlc_features,
    "finance": finance_features,
}
