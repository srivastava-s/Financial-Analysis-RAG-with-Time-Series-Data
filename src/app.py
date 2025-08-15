"""
Streamlit web application for Financial Analysis RAG System.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import List, Optional
import json

from .config import config
from .rag.generation.rag_system import FinancialRAGSystem
from .models import MarketData, RiskMetrics, TemporalContext

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

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with caching."""
    try:
        return FinancialRAGSystem()
    except Exception as e:
        st.error(f"Failed to initialize RAG system: {e}")
        return None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Financial Analysis RAG System</h1>', unsafe_allow_html=True)
    
    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        st.error("System initialization failed. Please check your configuration.")
        return
    
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
            start_date = st.date_input("Start Date", value=start_date)
        with col2:
            end_date = st.date_input("End Date", value=end_date)
        
        # Analysis options
        st.subheader("📊 Analysis Options")
        include_news = st.checkbox("Include News Analysis", value=True)
        include_market_data = st.checkbox("Include Market Data", value=True)
        
        # Analysis button
        analyze_button = st.button("🚀 Analyze", type="primary", use_container_width=True)
    
    # Main content area
    if analyze_button and query:
        with st.spinner("Analyzing your query..."):
            try:
                # Convert dates to datetime
                start_dt = datetime.combine(start_date, datetime.min.time())
                end_dt = datetime.combine(end_date, datetime.min.time())
                
                # Perform analysis
                response = rag_system.analyze_query(
                    query=query,
                    symbols=symbols,
                    start_date=start_dt,
                    end_date=end_dt,
                    include_news=include_news,
                    include_market_data=include_market_data
                )
                
                # Display results
                display_analysis_results(response, rag_system)
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
    
    elif not query:
        # Show welcome page
        display_welcome_page()

def display_welcome_page():
    """Display welcome page with examples."""
    
    st.markdown("""
    ## Welcome to the Financial Analysis RAG System! 🎯
    
    This system combines advanced AI with real-time financial data to provide comprehensive investment insights and risk assessments.
    
    ### 🚀 Key Features:
    - **Multi-source Data Integration**: Real-time market data, news, and financial reports
    - **Temporal Analysis**: Trend identification and pattern recognition
    - **Risk Assessment**: Comprehensive risk metrics and stress testing
    - **AI-Powered Insights**: Natural language analysis and recommendations
    
    ### 📝 Example Queries:
    - "Analyze Apple's financial performance and risk factors"
    - "Compare Tesla and Ford stock performance"
    - "What are the key risks for Microsoft stock?"
    - "Generate a risk assessment for the technology sector"
    - "Analyze market trends for AAPL, MSFT, and GOOGL"
    
    ### 🔧 How to Use:
    1. Enter your financial query in the sidebar
    2. Optionally specify stock symbols
    3. Set your desired date range
    4. Choose analysis options
    5. Click "Analyze" to get insights
    
    ---
    """)
    
    # System status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", "🟢 Online")
    
    with col2:
        st.metric("Data Sources", "📊 Active")
    
    with col3:
        st.metric("Analysis Engine", "🤖 Ready")

def display_analysis_results(response, rag_system):
    """Display analysis results."""
    
    # Response header
    st.header("📊 Analysis Results")
    
    # Confidence and processing time
    col1, col2, col3 = st.columns(3)
    
    with col1:
        confidence_color = "green" if response.confidence_score > 0.7 else "orange" if response.confidence_score > 0.4 else "red"
        st.metric("Confidence Score", f"{response.confidence_score:.1%}", delta=None)
    
    with col2:
        st.metric("Processing Time", f"{response.processing_time:.2f}s")
    
    with col3:
        st.metric("Sources Analyzed", len(response.sources))
    
    # Main response
    st.subheader("💡 Analysis Summary")
    st.markdown(response.answer)
    
    # Temporal context
    if response.temporal_context:
        display_temporal_context(response.temporal_context)
    
    # Risk assessment
    if response.risk_assessment:
        display_risk_assessment(response.risk_assessment)
    
    # Sources
    if response.sources:
        display_sources(response.sources)
    
    # Interactive charts (if we have market data)
    if hasattr(response, 'market_data') and response.market_data:
        display_market_charts(response.market_data)

def display_temporal_context(temporal_context: TemporalContext):
    """Display temporal context information."""
    
    st.subheader("⏰ Temporal Context")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Timeframe", temporal_context.timeframe)
    
    with col2:
        trend_color = "green" if temporal_context.trend_direction == "uptrend" else "red" if temporal_context.trend_direction == "downtrend" else "gray"
        st.metric("Trend", temporal_context.trend_direction.title(), delta=None)
    
    with col3:
        vol_color = "red" if temporal_context.volatility_regime == "high" else "green" if temporal_context.volatility_regime == "low" else "orange"
        st.metric("Volatility", temporal_context.volatility_regime.title(), delta=None)
    
    with col4:
        st.metric("Period", f"{temporal_context.start_date.strftime('%Y-%m-%d')} to {temporal_context.end_date.strftime('%Y-%m-%d')}")

def display_risk_assessment(risk_metrics: RiskMetrics):
    """Display risk assessment metrics."""
    
    st.subheader("⚠️ Risk Assessment")
    
    # Risk level indicator
    risk_color = {
        "low": "green",
        "medium": "orange", 
        "high": "red",
        "very_high": "darkred"
    }.get(risk_metrics.risk_level.value, "gray")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Risk Level", risk_metrics.risk_level.value.title(), delta=None)
    
    with col2:
        st.metric("Volatility", f"{risk_metrics.volatility:.1%}")
    
    with col3:
        st.metric("Beta", f"{risk_metrics.beta:.2f}")
    
    with col4:
        st.metric("VaR (95%)", f"{risk_metrics.var_95:.1%}")
    
    # Additional metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Sharpe Ratio", f"{risk_metrics.sharpe_ratio:.2f}")
    
    with col2:
        st.metric("Max Drawdown", f"{risk_metrics.max_drawdown:.1%}")
    
    with col3:
        st.metric("Market Correlation", f"{risk_metrics.correlation_market:.2f}")
    
    # Risk visualization
    fig = go.Figure()
    
    # Risk radar chart
    categories = ['Volatility', 'VaR', 'Beta', 'Drawdown', 'Correlation']
    values = [
        risk_metrics.volatility * 100,
        risk_metrics.var_95,
        risk_metrics.beta * 20,  # Scale beta
        risk_metrics.max_drawdown,
        risk_metrics.correlation_market * 100
    ]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Profile',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 50]
            )),
        showlegend=False,
        title="Risk Profile Radar Chart"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_sources(sources: List[dict]):
    """Display analysis sources."""
    
    st.subheader("📚 Sources")
    
    # Separate news and analysis sources
    news_sources = [s for s in sources if s.get('type') == 'news']
    analysis_sources = [s for s in sources if s.get('type') == 'analysis']
    
    if news_sources:
        st.write("**News Articles:**")
        for i, source in enumerate(news_sources[:5]):  # Show top 5
            with st.expander(f"{i+1}. {source.get('title', 'Unknown')}"):
                st.write(f"**Source:** {source.get('source', 'Unknown')}")
                st.write(f"**Published:** {source.get('published_at', 'Unknown')}")
                if source.get('url'):
                    st.write(f"**URL:** {source.get('url')}")
    
    if analysis_sources:
        st.write("**Analysis Insights:**")
        for source in analysis_sources:
            st.write(f"• {source.get('symbol', 'Unknown')}: {source.get('insight_type', 'Unknown')} (Confidence: {source.get('confidence_score', 0):.1%})")

def display_market_charts(market_data: dict):
    """Display interactive market charts."""
    
    st.subheader("📈 Market Data Visualization")
    
    for symbol, data in market_data.items():
        if not data:
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'date': md.date,
                'open': md.open_price,
                'high': md.high_price,
                'low': md.low_price,
                'close': md.close_price,
                'volume': md.volume
            }
            for md in data
        ])
        
        # Candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df['date'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name=symbol
        )])
        
        fig.update_layout(
            title=f"{symbol} Stock Price",
            yaxis_title="Price ($)",
            xaxis_title="Date",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Volume chart
        fig_volume = px.bar(df, x='date', y='volume', title=f"{symbol} Trading Volume")
        st.plotly_chart(fig_volume, use_container_width=True)

if __name__ == "__main__":
    main()
