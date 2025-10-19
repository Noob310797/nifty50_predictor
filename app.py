import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from utils.data_fetcher import DataFetcher
from models.arima_garch_model import ArimaGarchPredictor
from config import *

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Indian Stock Price Predictor")
st.markdown("""
This app predicts stock prices for the next 5 trading days using **ARIMA-GARCH** statistical model combined with **India VIX** volatility analysis.

**How to use:** Enter an NSE stock symbol (e.g., RELIANCE, TCS, INFY) and click Predict!
""")

# Sidebar for input
st.sidebar.header("Stock Selection")
stock_symbol = st.sidebar.text_input(
    "Enter NSE Stock Symbol",
    value="RELIANCE",
    help="Enter stock symbol without .NS (e.g., RELIANCE, TCS, INFY, HDFCBANK)"
).upper()

# Add .NS suffix for Yahoo Finance
full_symbol = f"{stock_symbol}.NS"

# Predict button
if st.sidebar.button("🔮 Predict Stock Price", type="primary"):
    
    # Progress indicator
    with st.spinner(f"Fetching data for {stock_symbol}..."):
        try:
            # Fetch data
            fetcher = DataFetcher(full_symbol)
            data = fetcher.combine_data()
            
            if data is None or len(data) < MIN_DATA_POINTS:
                st.error(f"❌ Insufficient data for {stock_symbol}. Please try another stock.")
                st.stop()
            
            # Display data info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"₹{data['Close'].iloc[-1]:.2f}")
            with col2:
                st.metric("Data Points", f"{len(data)} days")
            with col3:
                st.metric("India VIX", f"{data['VIX'].iloc[-1]:.2f}")
            with col4:
                daily_return = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
                st.metric("Today's Return", f"{daily_return:+.2f}%")
            
            st.success("✅ Data fetched successfully!")
            
        except Exception as e:
            st.error(f"❌ Error fetching data: {str(e)}")
            st.stop()
    
    # Train model
    with st.spinner("Training ARIMA-GARCH model..."):
        try:
            recent_data = data.tail(LOOKBACK_PERIOD)
            model = ArimaGarchPredictor(recent_data, full_symbol)
            
            if not model.fit_arima():
                st.error("❌ Failed to fit ARIMA model")
                st.stop()
            
            if not model.fit_garch():
                st.warning("⚠️ GARCH model failed, using simplified volatility")
            
            st.success("✅ Model trained successfully!")
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            st.stop()
    
    # Generate predictions
    with st.spinner("Generating 5-day forecast..."):
        try:
            predictions = model.predict(steps=PREDICTION_DAYS)
            
            if predictions is None:
                st.error("❌ Prediction failed")
                st.stop()
            
            # Display results
            st.markdown("---")
            st.header(f"📊 Prediction Results for {stock_symbol}")
            
            # Summary metrics
            last_price = data['Close'].iloc[-1]
            day5_price = predictions.loc['Day_5', 'VIX_Adjusted_Price']
            day5_prob = predictions.loc['Day_5', 'Hit_Probability']
            change_pct = ((day5_price - last_price) / last_price) * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "5-Day Target Price",
                    f"₹{day5_price:.2f}",
                    f"{change_pct:+.2f}%",
                    delta_color="normal"
                )
            
            with col2:
                st.metric("Hit Probability", f"{day5_prob:.1f}%")
            
            with col3:
                direction = "📈 Bullish" if change_pct > 0 else "📉 Bearish"
                st.metric("Signal", direction)
            
            # Detailed forecast table
            st.subheader("📅 Detailed 5-Day Forecast")
            
            # Format the dataframe for display
            display_df = predictions[['VIX_Adjusted_Price', 'Lower_Bound', 'Upper_Bound', 'Hit_Probability']].copy()
            display_df.columns = ['Target Price (₹)', 'Lower Bound (₹)', 'Upper Bound (₹)', 'Probability (%)']
            
            # Round values
            display_df['Target Price (₹)'] = display_df['Target Price (₹)'].round(2)
            display_df['Lower Bound (₹)'] = display_df['Lower Bound (₹)'].round(2)
            display_df['Upper Bound (₹)'] = display_df['Upper Bound (₹)'].round(2)
            
            # Display as table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=250
            )
            
            # Model info
            st.info(f"""
            **Model Details:**
            - Model Type: ARIMA{model.arima_order} + GARCH(1,1)
            - Training Data: Last {LOOKBACK_PERIOD} trading days
            - Current VIX: {data['VIX'].iloc[-1]:.2f}
            - Forecast Horizon: 5 trading days
            """)
            
            # Download option
            csv = display_df.to_csv()
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv,
                file_name=f"{stock_symbol}_prediction_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
            # Warning disclaimer
            st.warning("""
            ⚠️ **Disclaimer:** This is a statistical model for educational purposes only. 
            Current backtest shows ~43% direction accuracy. Do not use for actual trading decisions. 
            Always consult financial advisors before making investment decisions.
            """)
            
        except Exception as e:
            st.error(f"❌ Error generating predictions: {str(e)}")
            st.stop()

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Popular NSE Stocks:
- RELIANCE
- TCS
- INFY
- HDFCBANK
- TATAMOTORS
- WIPRO
- ICICIBANK
- SBIN
- BHARTIARTL
- OLAELEC

### About:
This predictor uses:
- ARIMA for trend forecasting
- GARCH for volatility modeling
- India VIX for market sentiment
""")

st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Python & Streamlit")
