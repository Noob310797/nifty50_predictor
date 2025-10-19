import sys
import argparse
from datetime import datetime
from utils.data_fetcher import DataFetcher
from models.arima_garch_model import ArimaGarchPredictor
from utils.backtester import ModelBacktester
from config import *

def predict_stock(symbol, backtest=False):
    """Main prediction function"""
    print(f"\n{'='*70}")
    print(f"STOCK PRICE PREDICTOR - {symbol}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    # Step 1: Fetch data
    print("Step 1: Fetching data...")
    fetcher = DataFetcher(symbol)
    data = fetcher.combine_data()
    
    if data is None or len(data) < MIN_DATA_POINTS:
        print(f"Error: Insufficient data for {symbol}")
        return
    
    print(f"✓ Fetched {len(data)} trading days of data")
    print(f"  Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"  Current price: ₹{data['Close'].iloc[-1]:.2f}")
    print(f"  India VIX: {data['VIX'].iloc[-1]:.2f}\n")
    
    # Step 2: Run backtest (optional)
    if backtest:
        print("Step 2: Running backtest...")
        backtester = ModelBacktester(symbol)
        results, metrics = backtester.run_backtest(data)
        if metrics is not None:
            backtester.print_results(metrics)
        else:
            print("Backtest skipped due to insufficient data\n")
    
    # Step 3: Train model
    print("Step 3: Training prediction model...")
    recent_data = data.tail(LOOKBACK_PERIOD)
    model = ArimaGarchPredictor(recent_data, symbol)
    
    if not model.fit_arima():
        print("Error: Failed to fit ARIMA model")
        return
    
    if not model.fit_garch():
        print("Warning: GARCH model failed, using simplified volatility")
    
    # Step 4: Generate predictions
    print("\nStep 4: Generating predictions...")
    predictions = model.predict(steps=PREDICTION_DAYS)
    
    if predictions is None:
        print("Error: Prediction failed")
        return
    
    # Display results
    print(model.get_summary())
    print("\nDetailed 5-Day Forecast:")
    print(predictions[['VIX_Adjusted_Price', 'Lower_Bound', 'Upper_Bound', 'Hit_Probability']].to_string())
    
    # Save results
    output_file = f"{OUTPUT_DIR}/{symbol.replace('.NS', '')}_{datetime.now().strftime('%Y%m%d')}.csv"
    predictions.to_csv(output_file)
    print(f"\n✓ Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Stock Price Prediction Model')
    parser.add_argument('--symbol', type=str, default=DEFAULT_SYMBOL,
                       help='Stock symbol (e.g., TCS.NS, RELIANCE.NS)')
    parser.add_argument('--backtest', action='store_true',
                       help='Run backtesting before prediction')
    
    args = parser.parse_args()
    
    predict_stock(args.symbol, args.backtest)

if __name__ == "__main__":
    main()
