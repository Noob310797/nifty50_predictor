import pandas as pd
import numpy as np
from datetime import timedelta
from config import *

class ModelBacktester:
    """
    Walk-forward backtesting for ARIMA-GARCH model
    Tests model performance on historical data
    """
    
    def __init__(self, symbol, test_period_days=BACKTEST_PERIOD):
        self.symbol = symbol
        self.test_period = test_period_days
        self.results = []
        
    def run_backtest(self, data):
        """
        Perform walk-forward backtest
        """
        print(f"Starting backtest for {self.symbol}...")
        print(f"Test period: {self.test_period} days")
        
        total_days = len(data)
        train_size = total_days - self.test_period
        
        if train_size < MIN_DATA_POINTS:
            print("Insufficient data for backtesting")
            return None, None
        
        predictions_list = []
        actuals_list = []
        
        # Walk-forward validation
        for i in range(0, self.test_period - PREDICTION_DAYS, PREDICTION_DAYS):
            train_data = data.iloc[:train_size + i]
            test_data = data.iloc[train_size + i:train_size + i + PREDICTION_DAYS]
            
            if len(test_data) < PREDICTION_DAYS:
                break
            
            print(f"  Testing window {i//PREDICTION_DAYS + 1}...", end=' ')
            
            # Train model on current window
            from models.arima_garch_model import ArimaGarchPredictor
            model = ArimaGarchPredictor(train_data, self.symbol)
            
            if not model.fit_arima() or not model.fit_garch():
                print("Failed")
                continue
            
            # Predict next 5 days
            forecast = model.predict(steps=PREDICTION_DAYS)
            
            if forecast is not None:
                pred_prices = forecast['VIX_Adjusted_Price'].values
                actual_prices = test_data['Close'].values
                
                predictions_list.extend(pred_prices[:len(actual_prices)])
                actuals_list.extend(actual_prices)
                print("✓")
            else:
                print("Failed")
        
        if len(predictions_list) == 0:
            print("Backtest failed - no successful predictions")
            return None, None
        
        # Calculate metrics
        results_df = pd.DataFrame({
            'Predicted': predictions_list,
            'Actual': actuals_list
        })
        
        metrics = self._calculate_metrics(results_df)
        
        return results_df, metrics
    
    def _calculate_metrics(self, results_df):
        """Calculate backtest performance metrics"""
        predictions = results_df['Predicted'].values
        actuals = results_df['Actual'].values
        
        # Mean Absolute Error
        mae = np.mean(np.abs(predictions - actuals))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Direction Accuracy (did we predict up/down correctly?)
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        direction_accuracy = np.mean(pred_direction == actual_direction) * 100
        
        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_accuracy,
            'R2_Score': r2,
            'Sample_Size': len(actuals)
        }
        
        return metrics
    
    def print_results(self, metrics):
        """Print backtest results"""
        print(f"\n{'='*60}")
        print(f"BACKTEST RESULTS FOR {self.symbol}")
        print(f"{'='*60}")
        print(f"Mean Absolute Error (MAE): ₹{metrics['MAE']:.2f}")
        print(f"Root Mean Squared Error (RMSE): ₹{metrics['RMSE']:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%")
        print(f"Direction Accuracy: {metrics['Direction_Accuracy']:.2f}%")
        print(f"R² Score: {metrics['R2_Score']:.4f}")
        print(f"Sample Size: {metrics['Sample_Size']} predictions")
        print(f"{'='*60}\n")
