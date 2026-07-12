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


def _engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and classification target."""
    data = data.copy()
    data["MA10"] = data["Close"].rolling(window=10).mean()
    data["MA50"] = data["Close"].rolling(window=50).mean()

    # Tomorrow's close — used to define the target label
    data["Tomorrow"] = data["Close"].shift(-1)

    # 0 = SELL  (drop >1%)  |  1 = HOLD  |  2 = BUY  (rise >1%)
    data["Target"] = 1
    data.loc[data["Tomorrow"] > data["Close"] * 1.01, "Target"] = 2
    data.loc[data["Tomorrow"] < data["Close"] * 0.99, "Target"] = 0

    data.dropna(inplace=True)
    return data


def _build_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        random_state=RF_RANDOM_STATE,
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
