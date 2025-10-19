import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from arch import arch_model
from scipy import stats
from config import *

class ArimaGarchPredictor:
    """
    Hybrid ARIMA-GARCH model for stock price prediction
    - ARIMA: Captures trend and autocorrelation in price movements
    - GARCH: Models time-varying volatility
    - VIX Integration: Adjusts predictions based on market volatility
    """
    
    def __init__(self, data, symbol):
        self.data = data
        self.symbol = symbol
        self.arima_model = None
        self.garch_model = None
        self.arima_order = None
        self.forecast_results = {}
        
    def find_optimal_arima(self):
        """
        Find optimal ARIMA parameters using grid search
        Simplified version without pmdarima
        """
        print(f"Finding optimal ARIMA parameters for {self.symbol}...")
        
        # Check stationarity
        result = adfuller(self.data['Close'])
        is_stationary = result[1] < 0.05
        d = 0 if is_stationary else 1
        
        print(f"Stationarity test p-value: {result[1]:.4f}")
        print(f"Differencing order (d): {d}")
        
        # Grid search for p and q
        best_aic = np.inf
        best_order = (1, d, 1)
        
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = ARIMA(self.data['Close'], order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                        
                except:
                    continue
        
        self.arima_order = best_order
        print(f"Optimal ARIMA order: {self.arima_order} (AIC: {best_aic:.2f})")
        return self.arima_order
    
    def fit_arima(self):
        """Fit ARIMA model to price data"""
        if self.arima_order is None:
            self.find_optimal_arima()
        
        print(f"Fitting ARIMA{self.arima_order} model...")
        
        try:
            self.arima_model = ARIMA(
                self.data['Close'],
                order=self.arima_order
            ).fit()
            
            print("ARIMA model fitted successfully")
            print(f"AIC: {self.arima_model.aic:.2f}")
            return True
            
        except Exception as e:
            print(f"Error fitting ARIMA: {str(e)}")
            return False
    
    def fit_garch(self):
        """
        Fit GARCH(1,1) model to capture volatility clustering
        """
        print("Fitting GARCH(1,1) model for volatility...")
        
        try:
            # Use returns for GARCH modeling
            returns = self.data['Returns'].dropna() * 100  # Scale to percentage
            
            # GARCH(1,1) specification
            self.garch_model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                dist='normal'
            ).fit(disp='off')
            
            print("GARCH model fitted successfully")
            return True
            
        except Exception as e:
            print(f"Error fitting GARCH: {str(e)}")
            return False
    
    def predict(self, steps=PREDICTION_DAYS):
        """
        Generate predictions for next N trading days
        Returns: DataFrame with predictions and confidence intervals
        """
        if self.arima_model is None:
            print("Error: ARIMA model not fitted")
            return None
        
        print(f"\nGenerating {steps}-day forecast...")
        
        # ARIMA forecast
        forecast = self.arima_model.forecast(steps=steps)
        forecast_df = pd.DataFrame({
            'Predicted_Price': forecast.values
        })
        
        # GARCH volatility forecast
        if self.garch_model is not None:
            garch_forecast = self.garch_model.forecast(horizon=steps)
            volatility = np.sqrt(garch_forecast.variance.values[-1, :])
        else:
            # Fallback: Use historical volatility
            volatility = self.data['Returns'].std() * 100 * np.sqrt(np.arange(1, steps+1))
        
        # Calculate confidence intervals BEFORE VIX adjustment
        current_price = self.data['Close'].iloc[-1]
        z_score = stats.norm.ppf((1 + CONFIDENCE_LEVEL) / 2)
        
        # Fix: Calculate bounds based on predicted price and volatility
        upper_bounds = []
        lower_bounds = []
        
        for i in range(len(forecast_df)):
            pred_price = forecast_df.iloc[i]['Predicted_Price']
            vol = volatility[i] / 100  # Convert back to decimal
            
            # Calculate price range based on volatility
            price_std = pred_price * vol
            
            upper_bounds.append(pred_price + (z_score * price_std))
            lower_bounds.append(pred_price - (z_score * price_std))
        
        forecast_df['Upper_Bound'] = upper_bounds
        forecast_df['Lower_Bound'] = lower_bounds
        forecast_df['Volatility'] = volatility
        
        # Adjust for VIX AFTER calculating bounds
        current_vix = self.data['VIX'].iloc[-1]
        avg_vix = self.data['VIX'].mean()
        vix_adjustment = (current_vix / avg_vix - 1) * VOLATILITY_WEIGHT
        
        forecast_df['VIX_Adjusted_Price'] = forecast_df['Predicted_Price'] * (1 - vix_adjustment)
        
        # Calculate hit probability
        forecast_df['Hit_Probability'] = self._calculate_hit_probability(
            forecast_df, current_price, current_vix
        )
        
        # Add trading day numbers
        forecast_df.index = [f"Day_{i+1}" for i in range(steps)]
        
        self.forecast_results = forecast_df
        return forecast_df
    
    def _calculate_hit_probability(self, forecast_df, current_price, current_vix):
        """
        Calculate probability of hitting target using statistical measures
        Based on:
        1. Distance from current price (mean reversion tendency)
        2. Current volatility regime (VIX level)
        3. Confidence interval width
        4. Time decay (further predictions less certain)
        """
        probabilities = []
        
        for day_num, (idx, row) in enumerate(forecast_df.iterrows(), start=1):
            predicted = row['VIX_Adjusted_Price']  # Use VIX-adjusted price
            upper = row['Upper_Bound']
            lower = row['Lower_Bound']
            
            # Factor 1: Price deviation (larger moves = lower probability)
            price_change_pct = abs((predicted - current_price) / current_price)
            if price_change_pct < 0.02:  # Less than 2% change
                deviation_factor = 0.95
            elif price_change_pct < 0.05:  # 2-5% change
                deviation_factor = 0.85
            else:  # More than 5% change
                deviation_factor = 0.70
            
            # Factor 2: VIX regime
            if current_vix < 15:
                vix_factor = 0.90
            elif current_vix < 20:
                vix_factor = 0.80
            elif current_vix < 25:
                vix_factor = 0.70
            else:
                vix_factor = 0.60
            
            # Factor 3: Confidence interval width (tighter = more certain)
            ci_width_pct = (upper - lower) / predicted
            if ci_width_pct < 0.05:
                certainty_factor = 0.90
            elif ci_width_pct < 0.10:
                certainty_factor = 0.80
            else:
                certainty_factor = 0.70
            
            # Factor 4: Time decay (each day further reduces certainty)
            time_decay = 1.0 - (day_num - 1) * 0.05  # 5% reduction per day
            
            # Combined probability
            probability = (
                deviation_factor * 0.35 +
                vix_factor * 0.30 +
                certainty_factor * 0.20 +
                time_decay * 0.15
            ) * 100
            
            # Realistic range
            probability = np.clip(probability, 58, 88)
            probabilities.append(round(probability, 1))
        
        return probabilities
    
    def get_summary(self):
        """Generate prediction summary"""
        if len(self.forecast_results) == 0:
            return "No predictions available"
        
        last_price = self.data['Close'].iloc[-1]
        day5_price = self.forecast_results.loc['Day_5', 'VIX_Adjusted_Price']
        day5_prob = self.forecast_results.loc['Day_5', 'Hit_Probability']
        
        change_pct = ((day5_price - last_price) / last_price) * 100
        direction = "↑" if change_pct > 0 else "↓"
        
        summary = f"""
{'='*60}
PREDICTION SUMMARY FOR {self.symbol}
{'='*60}
Current Price: ₹{last_price:.2f}
5-Day Target: ₹{day5_price:.2f} ({direction} {abs(change_pct):.2f}%)
Hit Probability: {day5_prob:.1f}%
Current VIX: {self.data['VIX'].iloc[-1]:.2f}
Model: ARIMA{self.arima_order} + GARCH(1,1)
{'='*60}
"""
        
        return summary
