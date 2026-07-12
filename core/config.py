# ---------------------------------------------------------------------------
# core/config.py — App-wide constants and stock universe
# ---------------------------------------------------------------------------

# Minimum trading days required for the model to run
MIN_DATA_ROWS: int = 50

# Features used by the Random Forest
PREDICTORS: list[str] = [
    "MA10_ratio",    # price relative to 10-day MA  (trend direction)
    "MA50_ratio",    # price relative to 50-day MA  (trend strength)
    "RSI_14",        # momentum oscillator
    "BB_pct",        # Bollinger Band position       (mean-reversion signal)
    "Return_1d",     # yesterday's return            (short-term momentum)
    "Return_5d",     # 5-day momentum
    "Volume_ratio",  # today vs 20-day avg volume   (participation strength)
]

# Train / test split ratio
TRAIN_RATIO: float = 0.8

# Random Forest hyperparameters
RF_N_ESTIMATORS: int = 200          # more trees → lower variance
RF_MIN_SAMPLES_SPLIT: int = 10      # was 100 — trees were only 3–4 levels deep
RF_MAX_DEPTH: int = 15              # prevents overfitting on noisy data
RF_CLASS_WEIGHT: str = "balanced"   # compensates for HOLD majority class
RF_RANDOM_STATE: int = 1

# NSE stocks available for analysis
STOCK_MAP: dict[str, str] = {
    "Reliance Industries": "RELIANCE.NS",
    "Tata Steel":          "TATASTEEL.NS",
    "Infosys":             "INFY.NS",
    "Tata Motors":         "TATAMOTORS.NS",
    "HDFC Bank":           "HDFCBANK.NS",
    "MRF Tyres":           "MRF.NS",
    "Zomato":              "ZOMATO.NS",
    "Wipro":               "WIPRO.NS",
    "ITC Limited":         "ITC.NS",
}
