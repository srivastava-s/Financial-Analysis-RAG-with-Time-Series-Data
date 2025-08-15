"""
Main RAG system for financial analysis.
"""

import openai
import time
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
import uuid

from ...models import (
    AnalysisQuery, AnalysisResult, RAGResponse, InvestmentInsight,
    MarketData, NewsArticle, RiskMetrics, TemporalContext
)
from ...config import config
from ..embeddings.embedding_manager import EmbeddingManager
from ...data.storage.vector_store import VectorStore
from ...data.collectors.market_data_collector import MarketDataCollector
from ...data.collectors.news_collector import NewsCollector
from ...analysis.temporal.temporal_analyzer import TemporalAnalyzer
from ...analysis.risk.risk_analyzer import RiskAnalyzer

logger = logging.getLogger(__name__)


class FinancialRAGSystem:
    """Main RAG system for financial analysis."""
    
    def __init__(self):
        """Initialize the RAG system."""
        self.openai_client = None
        self.embedding_manager = None
        self.vector_store = None
        self.market_collector = None
        self.news_collector = None
        self.temporal_analyzer = None
        self.risk_analyzer = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Initialize OpenAI client
            if config.OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
                logger.info("Initialized OpenAI client")
            
            # Initialize components
            self.embedding_manager = EmbeddingManager()
            self.vector_store = VectorStore()
            self.market_collector = MarketDataCollector()
            self.news_collector = NewsCollector()
            self.temporal_analyzer = TemporalAnalyzer()
            self.risk_analyzer = RiskAnalyzer()
            
            logger.info("Financial RAG System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def analyze_query(
        self, 
        query: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        include_news: bool = True,
        include_market_data: bool = True
    ) -> RAGResponse:
        """
        Analyze a financial query and generate insights.
        
        Args:
            query: User query
            symbols: List of stock symbols
            start_date: Start date for analysis
            end_date: End date for analysis
            include_news: Whether to include news analysis
            include_market_data: Whether to include market data analysis
            
        Returns:
            RAGResponse with analysis results
        """
        start_time = time.time()
        
        try:
            # Create analysis query
            analysis_query = AnalysisQuery(
                query=query,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                include_news=include_news,
                include_market_data=include_market_data
            )
            
            # Extract symbols from query if not provided
            if not symbols:
                symbols = self._extract_symbols_from_query(query)
            
            # Collect data
            market_data = {}
            news_articles = []
            
            if include_market_data and symbols:
                market_data = self._collect_market_data(symbols, start_date, end_date)
            
            if include_news and symbols:
                news_articles = self._collect_news_data(symbols, start_date, end_date)
            
            # Perform analysis
            insights = []
            risk_assessment = None
            temporal_context = None
            
            if market_data:
                # Temporal analysis
                for symbol, data in market_data.items():
                    if data:
                        temporal_analysis = self.temporal_analyzer.analyze_temporal_patterns(data)
                        temporal_context = temporal_analysis.get('temporal_context')
                        
                        # Risk analysis
                        risk_metrics = self.risk_analyzer.calculate_risk_metrics(data)
                        risk_assessment = risk_metrics
                        
                        # Generate insights
                        symbol_insights = self._generate_insights(
                            query, symbol, data, news_articles, temporal_analysis, risk_metrics
                        )
                        insights.extend(symbol_insights)
            
            # Generate response
            answer = self._generate_response(query, insights, market_data, news_articles)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(insights, market_data, news_articles)
            
            return RAGResponse(
                query=query,
                answer=answer,
                sources=self._extract_sources(insights, news_articles),
                confidence_score=confidence_score,
                processing_time=processing_time,
                temporal_context=temporal_context,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            processing_time = time.time() - start_time
            
            return RAGResponse(
                query=query,
                answer=f"I apologize, but I encountered an error while analyzing your query: {str(e)}. Please try again or rephrase your question.",
                sources=[],
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query text."""
        # Simple symbol extraction - can be enhanced with NLP
        import re
        
        # Common stock symbol patterns
        symbol_pattern = r'\b[A-Z]{1,5}\b'
        symbols = re.findall(symbol_pattern, query.upper())
        
        # Filter out common words that might match symbol pattern
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'MAN', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WAY', 'WHO', 'BOY', 'DID', 'ITS', 'LET', 'PUT', 'SAY', 'SHE', 'TOO', 'USE'}
        
        symbols = [s for s in symbols if s not in common_words and len(s) >= 2]
        
        return symbols[:5]  # Limit to 5 symbols
    
    def _collect_market_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime], 
        end_date: Optional[datetime]
    ) -> Dict[str, List[MarketData]]:
        """Collect market data for symbols."""
        try:
            return self.market_collector.get_multiple_stocks_data(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            logger.error(f"Error collecting market data: {e}")
            return {}
    
    def _collect_news_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime], 
        end_date: Optional[datetime]
    ) -> List[NewsArticle]:
        """Collect news data for symbols."""
        try:
            all_articles = []
            for symbol in symbols:
                articles = self.news_collector.get_company_news(
                    company_name=symbol,
                    symbol=symbol,
                    days_back=30
                )
                all_articles.extend(articles)
            
            # Remove duplicates and sort by relevance
            unique_articles = []
            seen_titles = set()
            for article in all_articles:
                if article.title not in seen_titles:
                    seen_titles.add(article.title)
                    unique_articles.append(article)
            
            # Sort by relevance score
            unique_articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)
            
            return unique_articles[:50]  # Limit to top 50 articles
            
        except Exception as e:
            logger.error(f"Error collecting news data: {e}")
            return []
    
    def _generate_insights(
        self,
        query: str,
        symbol: str,
        market_data: List[MarketData],
        news_articles: List[NewsArticle],
        temporal_analysis: Dict[str, Any],
        risk_metrics: RiskMetrics
    ) -> List[InvestmentInsight]:
        """Generate investment insights."""
        insights = []
        
        try:
            # Analyze market performance
            if market_data:
                current_price = market_data[-1].close_price
                start_price = market_data[0].close_price
                total_return = (current_price - start_price) / start_price * 100
                
                # Performance insight
                if total_return > 10:
                    insight_type = "opportunity"
                    reasoning = f"{symbol} has shown strong performance with {total_return:.1f}% return"
                elif total_return < -10:
                    insight_type = "risk_warning"
                    reasoning = f"{symbol} has declined by {abs(total_return):.1f}%, indicating potential risks"
                else:
                    insight_type = "hold"
                    reasoning = f"{symbol} has shown moderate performance with {total_return:.1f}% return"
                
                # Add supporting evidence
                supporting_evidence = []
                if temporal_analysis.get('trend_analysis', {}).get('trend_direction') == 'uptrend':
                    supporting_evidence.append("Positive trend detected")
                if risk_metrics.risk_level.value in ['low', 'medium']:
                    supporting_evidence.append("Acceptable risk profile")
                
                # Add risk factors
                risk_factors = []
                if risk_metrics.volatility > 0.25:
                    risk_factors.append("High volatility")
                if risk_metrics.var_95 > 3.0:
                    risk_factors.append("High Value at Risk")
                
                insight = InvestmentInsight(
                    id=str(uuid.uuid4()),
                    symbol=symbol,
                    insight_type=insight_type,
                    confidence_score=0.7,
                    reasoning=reasoning,
                    supporting_evidence=supporting_evidence,
                    risk_factors=risk_factors,
                    temporal_context=temporal_analysis.get('temporal_context'),
                    metadata={
                        'total_return': total_return,
                        'current_price': current_price,
                        'volatility': risk_metrics.volatility
                    }
                )
                
                insights.append(insight)
            
            # News sentiment insight
            if news_articles:
                symbol_news = [article for article in news_articles if symbol in article.keywords]
                if symbol_news:
                    avg_sentiment = sum(article.sentiment_score or 0 for article in symbol_news) / len(symbol_news)
                    
                    if avg_sentiment > 0.1:
                        news_insight = InvestmentInsight(
                            id=str(uuid.uuid4()),
                            symbol=symbol,
                            insight_type="opportunity",
                            confidence_score=0.6,
                            reasoning=f"Positive news sentiment detected for {symbol}",
                            supporting_evidence=[f"Average sentiment score: {avg_sentiment:.2f}"],
                            risk_factors=[],
                            temporal_context=temporal_analysis.get('temporal_context')
                        )
                        insights.append(news_insight)
                    elif avg_sentiment < -0.1:
                        news_insight = InvestmentInsight(
                            id=str(uuid.uuid4()),
                            symbol=symbol,
                            insight_type="risk_warning",
                            confidence_score=0.6,
                            reasoning=f"Negative news sentiment detected for {symbol}",
                            supporting_evidence=[f"Average sentiment score: {avg_sentiment:.2f}"],
                            risk_factors=["Negative media coverage"],
                            temporal_context=temporal_analysis.get('temporal_context')
                        )
                        insights.append(news_insight)
            
        except Exception as e:
            logger.error(f"Error generating insights for {symbol}: {e}")
        
        return insights
    
    def _generate_response(
        self,
        query: str,
        insights: List[InvestmentInsight],
        market_data: Dict[str, List[MarketData]],
        news_articles: List[NewsArticle]
    ) -> str:
        """Generate natural language response."""
        try:
            if not self.openai_client:
                return self._generate_simple_response(query, insights, market_data, news_articles)
            
            # Prepare context for LLM
            context = self._prepare_context(insights, market_data, news_articles)
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a financial analyst assistant. Provide clear, accurate, and actionable insights based on the provided data. Always include risk warnings and disclaimer that this is not financial advice."""
                    },
                    {
                        "role": "user",
                        "content": f"Query: {query}\n\nContext: {context}\n\nPlease provide a comprehensive financial analysis response."
                    }
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_simple_response(query, insights, market_data, news_articles)
    
    def _generate_simple_response(
        self,
        query: str,
        insights: List[InvestmentInsight],
        market_data: Dict[str, List[MarketData]],
        news_articles: List[NewsArticle]
    ) -> str:
        """Generate simple response without LLM."""
        response_parts = []
        
        # Add query response
        response_parts.append(f"Analysis for: {query}")
        
        # Add insights
        if insights:
            response_parts.append("\nKey Insights:")
            for insight in insights[:3]:  # Top 3 insights
                response_parts.append(f"• {insight.symbol}: {insight.reasoning}")
        
        # Add market data summary
        if market_data:
            response_parts.append("\nMarket Data Summary:")
            for symbol, data in market_data.items():
                if data:
                    current_price = data[-1].close_price
                    start_price = data[0].close_price
                    return_pct = (current_price - start_price) / start_price * 100
                    response_parts.append(f"• {symbol}: ${current_price:.2f} ({return_pct:+.1f}%)")
        
        # Add news summary
        if news_articles:
            response_parts.append(f"\nRecent News: {len(news_articles)} articles analyzed")
        
        # Add disclaimer
        response_parts.append("\n⚠️ Disclaimer: This analysis is for informational purposes only and does not constitute financial advice.")
        
        return "\n".join(response_parts)
    
    def _prepare_context(
        self,
        insights: List[InvestmentInsight],
        market_data: Dict[str, List[MarketData]],
        news_articles: List[NewsArticle]
    ) -> str:
        """Prepare context for LLM."""
        context_parts = []
        
        # Add insights
        if insights:
            context_parts.append("Investment Insights:")
            for insight in insights:
                context_parts.append(f"- {insight.symbol}: {insight.insight_type} - {insight.reasoning}")
        
        # Add market data
        if market_data:
            context_parts.append("\nMarket Data:")
            for symbol, data in market_data.items():
                if data:
                    current_price = data[-1].close_price
                    start_price = data[0].close_price
                    return_pct = (current_price - start_price) / start_price * 100
                    context_parts.append(f"- {symbol}: Current ${current_price:.2f}, Return {return_pct:+.1f}%")
        
        # Add news summary
        if news_articles:
            context_parts.append(f"\nNews Analysis: {len(news_articles)} articles")
            positive_news = [a for a in news_articles if a.sentiment_score and a.sentiment_score > 0]
            negative_news = [a for a in news_articles if a.sentiment_score and a.sentiment_score < 0]
            context_parts.append(f"- Positive: {len(positive_news)}, Negative: {len(negative_news)}")
        
        return "\n".join(context_parts)
    
    def _extract_sources(
        self,
        insights: List[InvestmentInsight],
        news_articles: List[NewsArticle]
    ) -> List[Dict[str, Any]]:
        """Extract sources for response."""
        sources = []
        
        # Add news sources
        for article in news_articles[:5]:  # Top 5 articles
            sources.append({
                'type': 'news',
                'title': article.title,
                'source': article.source,
                'url': article.url,
                'published_at': article.published_at.isoformat()
            })
        
        # Add insight sources
        for insight in insights:
            sources.append({
                'type': 'analysis',
                'symbol': insight.symbol,
                'insight_type': insight.insight_type,
                'confidence_score': insight.confidence_score
            })
        
        return sources
    
    def _calculate_confidence_score(
        self,
        insights: List[InvestmentInsight],
        market_data: Dict[str, List[MarketData]],
        news_articles: List[NewsArticle]
    ) -> float:
        """Calculate confidence score for the response."""
        score = 0.0
        
        # Base score
        if market_data:
            score += 0.3
        
        if news_articles:
            score += 0.2
        
        if insights:
            score += 0.3
        
        # Quality adjustments
        if insights:
            avg_confidence = sum(insight.confidence_score for insight in insights) / len(insights)
            score += avg_confidence * 0.2
        
        return min(score, 1.0)
