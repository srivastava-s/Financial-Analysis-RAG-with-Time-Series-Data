"""
Simplified Streamlit web application for Financial Analysis RAG System.
This version works with minimal dependencies.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import List, Optional
import json

# Page configuration
st.set_page_config(
    page_title="Financial Analysis RAG System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .risk-high { color: #d62728; }
    .risk-medium { color: #ff7f0e; }
    .risk-low { color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

def generate_sample_financial_data():
    """Generate sample financial data for demonstration."""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    data = {
        'Date': dates,
        'AAPL_Price': [150 + i * 0.1 + (i % 30) * 0.5 for i in range(len(dates))],
        'MSFT_Price': [300 + i * 0.15 + (i % 25) * 0.8 for i in range(len(dates))],
        'GOOGL_Price': [2800 + i * 0.2 + (i % 35) * 1.2 for i in range(len(dates))]
    }
    return pd.DataFrame(data)

def analyze_financial_performance(symbols, start_date, end_date):
    """Analyze financial performance for given symbols."""
    # Generate sample data
    df = generate_sample_financial_data()
    
    # Filter by date range
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    results = {}
    for symbol in symbols:
        if symbol in ['AAPL', 'MSFT', 'GOOGL']:
            price_col = f'{symbol}_Price'
            if price_col in df.columns:
                prices = df[price_col].dropna()
                if len(prices) > 0:
                    results[symbol] = {
                        'current_price': prices.iloc[-1],
                        'price_change': prices.iloc[-1] - prices.iloc[0],
                        'price_change_pct': ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100,
                        'volatility': prices.std(),
                        'trend': 'Bullish' if prices.iloc[-1] > prices.iloc[0] else 'Bearish'
                    }
    
    return results

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Financial Analysis RAG System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Query input
        query = st.text_area(
            "Enter your financial query:",
            placeholder="e.g., Analyze Apple's financial performance and risk factors",
            height=100
        )
        
        # Symbol input
        symbols_input = st.text_input(
            "Stock Symbols (comma-separated):",
            placeholder="AAPL, MSFT, GOOGL"
        )
        symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()] if symbols_input else []
        
        # Date range
        st.subheader("📅 Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", start_date)
        with col2:
            end_date = st.date_input("End Date", end_date)
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze", type="primary")
    
    # Main content area
    if analyze_button and symbols:
        st.header("📊 Financial Analysis Results")
        
        with st.spinner("Analyzing financial data..."):
            time.sleep(2)  # Simulate processing time
            
            # Perform analysis
            results = analyze_financial_performance(symbols, start_date, end_date)
            
            if results:
                # Display results in columns
                cols = st.columns(len(results))
                
                for i, (symbol, data) in enumerate(results.items()):
                    with cols[i]:
                        st.markdown(f"### {symbol}")
                        
                        # Price metrics
                        st.metric(
                            label="Current Price",
                            value=f"${data['current_price']:.2f}",
                            delta=f"{data['price_change']:.2f} ({data['price_change_pct']:.1f}%)"
                        )
                        
                        # Additional metrics
                        st.markdown(f"**Volatility:** ${data['volatility']:.2f}")
                        st.markdown(f"**Trend:** {data['trend']}")
                        
                        # Risk assessment
                        risk_level = "High" if data['volatility'] > 10 else "Medium" if data['volatility'] > 5 else "Low"
                        risk_color = "risk-high" if risk_level == "High" else "risk-medium" if risk_level == "Medium" else "risk-low"
                        st.markdown(f"**Risk Level:** <span class='{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
                
                # Insights section
                st.header("💡 Investment Insights")
                
                # Generate insights based on analysis
                insights = []
                for symbol, data in results.items():
                    if data['trend'] == 'Bullish':
                        insights.append(f"**{symbol}** shows a bullish trend with {data['price_change_pct']:.1f}% growth. Consider holding or buying on dips.")
                    else:
                        insights.append(f"**{symbol}** shows a bearish trend with {abs(data['price_change_pct']):.1f}% decline. Monitor for reversal signals.")
                    
                    if data['volatility'] > 10:
                        insights.append(f"**{symbol}** has high volatility (${data['volatility']:.2f}). Implement proper risk management.")
                
                for insight in insights:
                    st.markdown(f"- {insight}")
                
                # Sample data visualization
                st.header("📈 Price Trends")
                df = generate_sample_financial_data()
                df_filtered = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
                
                # Create a simple line chart
                chart_data = {}
                for symbol in symbols:
                    if symbol in ['AAPL', 'MSFT', 'GOOGL']:
                        price_col = f'{symbol}_Price'
                        if price_col in df_filtered.columns:
                            chart_data[symbol] = df_filtered[price_col]
                
                if chart_data:
                    chart_df = pd.DataFrame(chart_data, index=df_filtered['Date'])
                    st.line_chart(chart_df)
                
            else:
                st.warning("No data available for the selected symbols and date range.")
                st.info("Try using symbols like AAPL, MSFT, or GOOGL for sample data.")
    
    elif analyze_button:
        st.warning("Please enter at least one stock symbol to analyze.")
    
    # Default view
    else:
        st.header("🎯 Welcome to Financial Analysis RAG System")
        
        st.markdown("""
        This system provides comprehensive financial analysis and insights for investment decision-making.
        
        ### 🚀 Getting Started:
        1. **Enter your financial query** in the sidebar
        2. **Select stock symbols** (e.g., AAPL, MSFT, GOOGL)
        3. **Choose date range** for analysis
        4. **Click Analyze** to get insights
        
        ### 📊 What You'll Get:
        - **Price Analysis**: Current prices, changes, and trends
        - **Risk Assessment**: Volatility and risk level indicators
        - **Investment Insights**: Actionable recommendations
        - **Visual Charts**: Price trend visualizations
        
        ### 💡 Sample Queries:
        - "Analyze Apple's financial performance and risk factors"
        - "Compare Microsoft vs Google stock performance"
        - "What are the key risk factors for Tesla stock?"
        """)
        
        # Sample data preview
        st.header("📋 Sample Data Preview")
        df = generate_sample_financial_data()
        st.dataframe(df.tail(10), use_container_width=True)

if __name__ == "__main__":
    main()
