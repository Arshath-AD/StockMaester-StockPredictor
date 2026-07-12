# ---------------------------------------------------------------------------
# core/predictor.py — ML data pipeline and prediction logic
# ---------------------------------------------------------------------------

import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .config import (
    MIN_DATA_ROWS,
    PREDICTORS,
    TRAIN_RATIO,
    RF_N_ESTIMATORS,
    RF_MIN_SAMPLES_SPLIT,
    RF_MAX_DEPTH,
    RF_CLASS_WEIGHT,
    RF_RANDOM_STATE,
)

# Signal labels
SIGNAL_BUY  = "BUY"
SIGNAL_SELL = "SELL"
SIGNAL_HOLD = "HOLD"


def _fetch_data(symbol: str) -> pd.DataFrame:
    """Download historical OHLCV data from Yahoo Finance."""
    data = yf.download(symbol, period="5y", interval="1d", progress=False)

    # Flatten MultiIndex columns produced by some yfinance versions
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — pure pandas, no extra dependencies."""
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add scale-invariant technical indicators and classification target."""
    data  = data.copy()
    close = data["Close"]

    # --- Moving average ratios (scale-invariant trend signal) ---
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    data["MA10_ratio"]   = close / ma10          # >1 → price above 10-day avg
    data["MA50_ratio"]   = close / ma50          # >1 → price above 50-day avg

    # --- Momentum ---
    data["Return_1d"]    = close.pct_change(1)   # yesterday's move
    data["Return_5d"]    = close.pct_change(5)   # 5-day momentum

    # --- RSI (14-period) ---
    data["RSI_14"]       = _rsi(close, 14)

    # --- Bollinger Band %B ---
    std20                = close.rolling(20).std()
    bb_upper             = ma20 + 2 * std20
    bb_lower             = ma20 - 2 * std20
    data["BB_pct"]       = (close - bb_lower) / (bb_upper - bb_lower)

    # --- Volume ratio vs 20-day average ---
    data["Volume_ratio"] = data["Volume"] / data["Volume"].rolling(20).mean()

    # --- Target label (unchanged logic) ---
    data["Tomorrow"]     = close.shift(-1)
    data["Target"]       = 1
    data.loc[data["Tomorrow"] > close * 1.01, "Target"] = 2   # BUY
    data.loc[data["Tomorrow"] < close * 0.99, "Target"] = 0   # SELL

    data.dropna(inplace=True)
    return data


def _build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators      = RF_N_ESTIMATORS,
        min_samples_split = RF_MIN_SAMPLES_SPLIT,
        max_depth         = RF_MAX_DEPTH,
        class_weight      = RF_CLASS_WEIGHT,
        random_state      = RF_RANDOM_STATE,
        n_jobs            = -1,   # parallelise across all CPU cores
    )


def _decode_signal(label: int) -> str:
    return {2: SIGNAL_BUY, 0: SIGNAL_SELL}.get(label, SIGNAL_HOLD)


def get_prediction(stock_symbol: str) -> tuple[str | None, float | str]:
    """
    Run the full ML pipeline for a given NSE ticker.

    Returns
    -------
    (signal, value)
        signal — "BUY" | "SELL" | "HOLD", or None on failure
        value  — accuracy float (0–1) on success, error string on failure
    """
    try:
        data = _fetch_data(stock_symbol)

        if len(data) < MIN_DATA_ROWS:
            return None, "INSUFFICIENT_DATA"

        data = _engineer_features(data)

        model = _build_model()
        split = int(len(data) * TRAIN_RATIO)
        train, test = data.iloc[:split], data.iloc[split:]

        # Evaluate accuracy on held-out test set
        model.fit(train[PREDICTORS], train["Target"])
        preds = model.predict(test[PREDICTORS])
        accuracy = accuracy_score(test["Target"], preds)

        # Retrain on full data for the final prediction
        model.fit(data[PREDICTORS], data["Target"])
        latest = data.iloc[-1:][PREDICTORS]
        signal = _decode_signal(model.predict(latest)[0])

        return signal, accuracy

    except Exception as exc:
        return None, str(exc)
