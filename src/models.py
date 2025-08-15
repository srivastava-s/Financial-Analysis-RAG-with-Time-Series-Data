"""
Data models for the Financial Analysis RAG System.
"""

from datetime import datetime, date
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import pandas as pd


class DataSource(str, Enum):
    """Enumeration of data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FRED = "fred"
    NEWS_API = "news_api"
    SEC_FILINGS = "sec_filings"
    MANUAL = "manual"


class DocumentType(str, Enum):
    """Enumeration of document types."""
    FINANCIAL_REPORT = "financial_report"
    NEWS_ARTICLE = "news_article"
    MARKET_DATA = "market_data"
    ECONOMIC_INDICATOR = "economic_indicator"
    RISK_ASSESSMENT = "risk_assessment"
    ANALYSIS_REPORT = "analysis_report"


class RiskLevel(str, Enum):
    """Enumeration of risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class SentimentType(str, Enum):
    """Enumeration of sentiment types."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class MarketData(BaseModel):
    """Model for market data points."""
    symbol: str
    date: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    adjusted_close: float
    source: DataSource = DataSource.YAHOO_FINANCE
    
    @validator('date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class FinancialMetrics(BaseModel):
    """Model for financial metrics."""
    symbol: str
    date: datetime
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    profit_margin: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None


class NewsArticle(BaseModel):
    """Model for news articles."""
    id: str
    title: str
    content: str
    summary: Optional[str] = None
    author: Optional[str] = None
    source: str
    url: str
    published_at: datetime
    sentiment_score: Optional[float] = None
    sentiment_type: Optional[SentimentType] = None
    keywords: List[str] = Field(default_factory=list)
    relevance_score: Optional[float] = None
    
    @validator('published_at', pre=True)
    def parse_published_at(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class FinancialDocument(BaseModel):
    """Model for financial documents (reports, filings, etc.)."""
    id: str
    title: str
    content: str
    document_type: DocumentType
    company_symbol: Optional[str] = None
    filing_date: Optional[datetime] = None
    period_end: Optional[datetime] = None
    source: DataSource
    url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('filing_date', 'period_end', pre=True)
    def parse_dates(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class DocumentChunk(BaseModel):
    """Model for document chunks used in RAG."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RiskMetrics(BaseModel):
    """Model for risk assessment metrics."""
    symbol: str
    date: datetime
    volatility: float
    beta: float
    sharpe_ratio: float
    var_95: float  # Value at Risk at 95% confidence
    max_drawdown: float
    correlation_market: float
    risk_level: RiskLevel
    
    @validator('date', pre=True)
    def parse_date(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class TemporalContext(BaseModel):
    """Model for temporal context in analysis."""
    start_date: datetime
    end_date: datetime
    timeframe: str = "1d"
    seasonality_period: Optional[int] = None
    trend_direction: Optional[str] = None
    volatility_regime: Optional[str] = None
    
    @validator('start_date', 'end_date', pre=True)
    def parse_dates(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class InvestmentInsight(BaseModel):
    """Model for investment insights and recommendations."""
    id: str
    symbol: str
    insight_type: str  # "buy", "sell", "hold", "risk_warning", "opportunity"
    confidence_score: float
    reasoning: str
    supporting_evidence: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    temporal_context: TemporalContext
    generated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisQuery(BaseModel):
    """Model for analysis queries."""
    query: str
    symbols: Optional[List[str]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    analysis_type: Optional[str] = None  # "risk", "performance", "comparison", "trend"
    include_news: bool = True
    include_reports: bool = True
    include_market_data: bool = True
    
    @validator('start_date', 'end_date', pre=True)
    def parse_dates(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v


class AnalysisResult(BaseModel):
    """Model for analysis results."""
    query_id: str
    query: AnalysisQuery
    insights: List[InvestmentInsight]
    market_data: List[MarketData] = Field(default_factory=list)
    news_articles: List[NewsArticle] = Field(default_factory=list)
    risk_metrics: List[RiskMetrics] = Field(default_factory=list)
    temporal_context: TemporalContext
    generated_at: datetime = Field(default_factory=datetime.now)
    processing_time: float
    confidence_score: float


class RetrievalResult(BaseModel):
    """Model for retrieval results."""
    query: str
    documents: List[DocumentChunk]
    scores: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGResponse(BaseModel):
    """Model for RAG system responses."""
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: float
    processing_time: float
    temporal_context: Optional[TemporalContext] = None
    risk_assessment: Optional[RiskMetrics] = None
    generated_at: datetime = Field(default_factory=datetime.now)


class SystemMetrics(BaseModel):
    """Model for system performance metrics."""
    timestamp: datetime = Field(default_factory=datetime.now)
    retrieval_latency: float
    generation_latency: float
    total_latency: float
    retrieval_accuracy: Optional[float] = None
    response_relevance: Optional[float] = None
    system_load: Optional[float] = None
    memory_usage: Optional[float] = None


# Utility functions for data conversion
def market_data_to_dataframe(market_data: List[MarketData]) -> pd.DataFrame:
    """Convert list of MarketData to pandas DataFrame."""
    data = []
    for md in market_data:
        data.append({
            'symbol': md.symbol,
            'date': md.date,
            'open': md.open_price,
            'high': md.high_price,
            'low': md.low_price,
            'close': md.close_price,
            'volume': md.volume,
            'adj_close': md.adjusted_close,
            'source': md.source
        })
    return pd.DataFrame(data)


def dataframe_to_market_data(df: pd.DataFrame) -> List[MarketData]:
    """Convert pandas DataFrame to list of MarketData."""
    market_data = []
    for _, row in df.iterrows():
        market_data.append(MarketData(
            symbol=row['symbol'],
            date=row['date'],
            open_price=row['open'],
            high_price=row['high'],
            low_price=row['low'],
            close_price=row['close'],
            volume=row['volume'],
            adjusted_close=row['adj_close'],
            source=row.get('source', DataSource.YAHOO_FINANCE)
        ))
    return market_data
