# ---------------------------------------------------------------------------
# core/config.py — App-wide constants and stock universe
# ---------------------------------------------------------------------------

# Minimum trading days required for the model to run
MIN_DATA_ROWS: int = 50

# Features used by the Random Forest
PREDICTORS: list[str] = ["Close", "MA10", "MA50", "Volume"]

# Train / test split ratio
TRAIN_RATIO: float = 0.8

# Random Forest hyperparameters
RF_N_ESTIMATORS: int = 100
RF_MIN_SAMPLES_SPLIT: int = 100
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
