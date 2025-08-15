"""
Basic tests for the Financial Analysis RAG System.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pytest
from datetime import datetime, timedelta

from config import config
from models import MarketData, NewsArticle, RiskMetrics, RiskLevel
from data.collectors.market_data_collector import MarketDataCollector
from data.collectors.news_collector import NewsCollector
from analysis.risk.risk_analyzer import RiskAnalyzer
from analysis.temporal.temporal_analyzer import TemporalAnalyzer

def test_config():
    """Test configuration loading."""
    assert config is not None
    assert hasattr(config, 'OPENAI_API_KEY')
    assert hasattr(config, 'DATA_DIR')

def test_market_data_collector():
    """Test market data collector."""
    collector = MarketDataCollector()
    assert collector is not None
    
    # Test with a known stock symbol
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = collector.get_stock_data('AAPL', start_date, end_date)
    
    # Should return some data (even if empty list)
    assert isinstance(data, list)
    
    if data:
        # Check data structure
        assert isinstance(data[0], MarketData)
        assert data[0].symbol == 'AAPL'
        assert data[0].close_price > 0

def test_news_collector():
    """Test news collector."""
    collector = NewsCollector()
    assert collector is not None
    
    # Test mock news generation
    articles = collector._get_mock_news(['AAPL'], 5)
    
    assert isinstance(articles, list)
    assert len(articles) <= 5
    
    if articles:
        assert isinstance(articles[0], NewsArticle)
        assert 'AAPL' in articles[0].title

def test_risk_analyzer():
    """Test risk analyzer."""
    analyzer = RiskAnalyzer()
    assert analyzer is not None
    
    # Create mock market data
    mock_data = []
    base_price = 100.0
    for i in range(30):
        date = datetime.now() - timedelta(days=30-i)
        price = base_price + (i * 0.5)  # Simple upward trend
        mock_data.append(MarketData(
            symbol='TEST',
            date=date,
            open_price=price,
            high_price=price + 1,
            low_price=price - 1,
            close_price=price,
            volume=1000000,
            adjusted_close=price
        ))
    
    # Calculate risk metrics
    risk_metrics = analyzer.calculate_risk_metrics(mock_data)
    
    assert isinstance(risk_metrics, RiskMetrics)
    assert risk_metrics.symbol == 'TEST'
    assert risk_metrics.volatility >= 0
    assert risk_metrics.beta >= 0
    assert isinstance(risk_metrics.risk_level, RiskLevel)

def test_temporal_analyzer():
    """Test temporal analyzer."""
    analyzer = TemporalAnalyzer()
    assert analyzer is not None
    
    # Create mock market data
    mock_data = []
    base_price = 100.0
    for i in range(20):
        date = datetime.now() - timedelta(days=20-i)
        price = base_price + (i * 0.5)  # Simple upward trend
        mock_data.append(MarketData(
            symbol='TEST',
            date=date,
            open_price=price,
            high_price=price + 1,
            low_price=price - 1,
            close_price=price,
            volume=1000000,
            adjusted_close=price
        ))
    
    # Analyze temporal patterns
    analysis = analyzer.analyze_temporal_patterns(mock_data)
    
    assert isinstance(analysis, dict)
    assert 'basic_stats' in analysis
    assert 'trend_analysis' in analysis
    assert 'volatility_analysis' in analysis
    assert 'temporal_context' in analysis

def test_models():
    """Test data models."""
    # Test MarketData model
    market_data = MarketData(
        symbol='AAPL',
        date=datetime.now(),
        open_price=150.0,
        high_price=155.0,
        low_price=148.0,
        close_price=152.0,
        volume=1000000,
        adjusted_close=152.0
    )
    
    assert market_data.symbol == 'AAPL'
    assert market_data.close_price == 152.0
    
    # Test NewsArticle model
    news_article = NewsArticle(
        id='test_article',
        title='Test Article',
        content='This is a test article about AAPL.',
        source='Test Source',
        url='https://test.com/article',
        published_at=datetime.now(),
        sentiment_score=0.1,
        keywords=['AAPL', 'test']
    )
    
    assert news_article.title == 'Test Article'
    assert 'AAPL' in news_article.keywords

if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...")
    
    try:
        test_config()
        print("✅ Configuration test passed")
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    try:
        test_market_data_collector()
        print("✅ Market data collector test passed")
    except Exception as e:
        print(f"❌ Market data collector test failed: {e}")
    
    try:
        test_news_collector()
        print("✅ News collector test passed")
    except Exception as e:
        print(f"❌ News collector test failed: {e}")
    
    try:
        test_risk_analyzer()
        print("✅ Risk analyzer test passed")
    except Exception as e:
        print(f"❌ Risk analyzer test failed: {e}")
    
    try:
        test_temporal_analyzer()
        print("✅ Temporal analyzer test passed")
    except Exception as e:
        print(f"❌ Temporal analyzer test failed: {e}")
    
    try:
        test_models()
        print("✅ Models test passed")
    except Exception as e:
        print(f"❌ Models test failed: {e}")
    
    print("\n🎉 Basic tests completed!")
