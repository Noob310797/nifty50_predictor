import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import *

class DataFetcher:
    """
    Fetches and preprocesses stock data from Yahoo Finance
    Handles NSE stocks, India VIX, and data validation
    """
    
    def __init__(self, symbol, start_date=None, end_date=None):
        self.symbol = symbol
        self.end_date = end_date or datetime.now()
        self.start_date = start_date or (self.end_date - timedelta(days=365*TRAINING_DATA_YEARS))
        
    def fetch_stock_data(self):
        """Fetch OHLCV data for the stock"""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=self.start_date, end=self.end_date)
            
            if df.empty:
                raise ValueError(f"No data found for {self.symbol}")
            
            # Clean data
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.dropna()
            
            # Calculate returns
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {self.symbol}: {str(e)}")
            return None
    
    def fetch_india_vix(self):
        """Fetch India VIX data"""
        try:
            vix_ticker = yf.Ticker(INDIA_VIX_SYMBOL)
            vix_df = vix_ticker.history(start=self.start_date, end=self.end_date)
            
            if vix_df.empty:
                print("Warning: India VIX data not available, using alternative calculation")
                return None
            
            vix_df = vix_df[['Close']].rename(columns={'Close': 'VIX'})
            vix_df['VIX_Change'] = vix_df['VIX'].pct_change()
            
            return vix_df
            
        except Exception as e:
            print(f"Error fetching India VIX: {str(e)}")
            return None
    
    def combine_data(self):
        """Combine stock data with India VIX"""
        stock_df = self.fetch_stock_data()
        vix_df = self.fetch_india_vix()
        
        if stock_df is None:
            return None
        
        if vix_df is not None:
            # Merge on date index
            combined_df = stock_df.join(vix_df, how='left')
            # Forward fill missing VIX values
            combined_df['VIX'] = combined_df['VIX'].fillna(method='ffill')
            combined_df['VIX_Change'] = combined_df['VIX_Change'].fillna(0)
        else:
            combined_df = stock_df
            # Calculate implied volatility as alternative
            combined_df['VIX'] = combined_df['Returns'].rolling(window=21).std() * np.sqrt(252) * 100
            combined_df['VIX_Change'] = combined_df['VIX'].pct_change()
        
        # Remove any remaining NaN values
        combined_df = combined_df.dropna()
        
        return combined_df
    
    def get_last_n_days(self, n=LOOKBACK_PERIOD):
        """Get last N trading days of data"""
        df = self.combine_data()
        if df is not None and len(df) >= n:
            return df.tail(n)
        return df
