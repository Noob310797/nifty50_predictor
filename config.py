import os
from datetime import datetime, timedelta

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure directories exist
for directory in [DATA_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Model parameters
LOOKBACK_PERIOD = 126  # 6 months of trading days (approx)
PREDICTION_DAYS = 5  # Next 5 trading sessions
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

# Data parameters
TRAINING_DATA_YEARS = 3  # Use 3 years for robust model training
BACKTEST_PERIOD = 252  # 1 year of trading days for backtesting

# Stock symbols (NSE format for yfinance)
DEFAULT_SYMBOL = "RELIANCE.NS"
INDIA_VIX_SYMBOL = "^INDIAVIX"
NIFTY50_SYMBOL = "^NSEI"

# Model thresholds
MIN_DATA_POINTS = 100  # Minimum data points required
VOLATILITY_WEIGHT = 0.3  # Weight for VIX in prediction (adjustable)

# Backtesting parameters
INITIAL_CAPITAL = 100000  # Starting capital for backtesting
COMMISSION = 0.001  # 0.1% commission per trade