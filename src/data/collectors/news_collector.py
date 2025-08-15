"""
News data collector for financial analysis RAG system.
"""

import requests
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from urllib.parse import urlparse
import time

from ...models import NewsArticle, SentimentType, DataSource
from ...config import config

logger = logging.getLogger(__name__)


class NewsCollector:
    """Collector for financial news from various sources."""
    
    def __init__(self):
        self.news_api_key = config.NEWS_API_KEY
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Financial-Analysis-RAG/1.0'
        })
    
    def get_financial_news(
        self, 
        query: str = "finance",
        symbols: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_articles: int = 100
    ) -> List[NewsArticle]:
        """
        Fetch financial news articles.
        
        Args:
            query: Search query
            symbols: List of stock symbols to include in search
            start_date: Start date for news search
            end_date: End date for news search
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of NewsArticle objects
        """
        try:
            if not self.news_api_key:
                logger.warning("News API key not configured, using mock data")
                return self._get_mock_news(symbols, max_articles)
            
            # Build search query
            search_query = query
            if symbols:
                symbol_query = " OR ".join([f'"{symbol}"' for symbol in symbols])
                search_query = f"({search_query}) AND ({symbol_query})"
            
            # Set default dates
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=7)
            
            logger.info(f"Fetching news for query: {search_query}")
            
            # NewsAPI parameters
            params = {
                'q': search_query,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': min(max_articles, 100),  # NewsAPI max is 100
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            }
            
            # Make API request
            response = self.session.get(
                'https://newsapi.org/v2/everything',
                params=params,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'ok':
                logger.error(f"NewsAPI error: {data.get('message', 'Unknown error')}")
                return []
            
            articles = data.get('articles', [])
            logger.info(f"Retrieved {len(articles)} articles from NewsAPI")
            
            # Convert to NewsArticle objects
            news_articles = []
            for article in articles:
                try:
                    news_article = self._parse_news_article(article, symbols)
                    if news_article:
                        news_articles.append(news_article)
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue
            
            return news_articles[:max_articles]
            
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return self._get_mock_news(symbols, max_articles)
    
    def get_company_news(
        self, 
        company_name: str,
        symbol: Optional[str] = None,
        days_back: int = 30
    ) -> List[NewsArticle]:
        """
        Get news specifically about a company.
        
        Args:
            company_name: Company name
            symbol: Stock symbol
            days_back: Number of days to look back
            
        Returns:
            List of NewsArticle objects
        """
        # Build search terms
        search_terms = [company_name]
        if symbol:
            search_terms.append(symbol)
        
        query = " OR ".join(search_terms)
        start_date = datetime.now() - timedelta(days=days_back)
        
        return self.get_financial_news(
            query=query,
            symbols=[symbol] if symbol else None,
            start_date=start_date,
            max_articles=50
        )
    
    def get_market_news(self, days_back: int = 7) -> List[NewsArticle]:
        """
        Get general market news.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of NewsArticle objects
        """
        market_queries = [
            "stock market",
            "financial markets",
            "trading",
            "investing",
            "economy",
            "Federal Reserve",
            "interest rates"
        ]
        
        all_articles = []
        start_date = datetime.now() - timedelta(days=days_back)
        
        for query in market_queries:
            articles = self.get_financial_news(
                query=query,
                start_date=start_date,
                max_articles=20
            )
            all_articles.extend(articles)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            if article.title not in seen_titles:
                seen_titles.add(article.title)
                unique_articles.append(article)
        
        return unique_articles
    
    def get_sector_news(self, sector: str, days_back: int = 7) -> List[NewsArticle]:
        """
        Get news about a specific sector.
        
        Args:
            sector: Sector name (e.g., "technology", "healthcare")
            days_back: Number of days to look back
            
        Returns:
            List of NewsArticle objects
        """
        start_date = datetime.now() - timedelta(days=days_back)
        
        return self.get_financial_news(
            query=sector,
            start_date=start_date,
            max_articles=50
        )
    
    def _parse_news_article(self, article_data: Dict[str, Any], symbols: Optional[List[str]] = None) -> Optional[NewsArticle]:
        """
        Parse raw article data into NewsArticle object.
        
        Args:
            article_data: Raw article data from API
            symbols: List of symbols to check for relevance
            
        Returns:
            NewsArticle object or None
        """
        try:
            # Extract basic information
            title = article_data.get('title', '').strip()
            content = article_data.get('content', '').strip()
            description = article_data.get('description', '').strip()
            
            # Skip articles without content
            if not title or not content:
                return None
            
            # Parse published date
            published_str = article_data.get('publishedAt', '')
            if published_str:
                published_at = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
            else:
                published_at = datetime.now()
            
            # Generate article ID
            article_id = self._generate_article_id(title, published_at)
            
            # Extract source information
            source_name = article_data.get('source', {}).get('name', 'Unknown')
            url = article_data.get('url', '')
            
            # Clean content
            content = self._clean_content(content)
            if not content:
                content = description
            
            # Analyze sentiment
            sentiment_score, sentiment_type = self._analyze_sentiment(content)
            
            # Extract keywords
            keywords = self._extract_keywords(title, content, symbols)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(title, content, symbols)
            
            # Create NewsArticle object
            news_article = NewsArticle(
                id=article_id,
                title=title,
                content=content,
                summary=description,
                author=article_data.get('author'),
                source=source_name,
                url=url,
                published_at=published_at,
                sentiment_score=sentiment_score,
                sentiment_type=sentiment_type,
                keywords=keywords,
                relevance_score=relevance_score
            )
            
            return news_article
            
        except Exception as e:
            logger.warning(f"Error parsing article: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> tuple[float, SentimentType]:
        """
        Analyze sentiment of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (sentiment_score, sentiment_type)
        """
        try:
            # Use VADER sentiment analysis
            scores = self.sentiment_analyzer.polarity_scores(text)
            compound_score = scores['compound']
            
            # Determine sentiment type
            if compound_score >= 0.05:
                sentiment_type = SentimentType.POSITIVE
            elif compound_score <= -0.05:
                sentiment_type = SentimentType.NEGATIVE
            else:
                sentiment_type = SentimentType.NEUTRAL
            
            return compound_score, sentiment_type
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment: {e}")
            return 0.0, SentimentType.NEUTRAL
    
    def _extract_keywords(self, title: str, content: str, symbols: Optional[List[str]] = None) -> List[str]:
        """
        Extract keywords from article.
        
        Args:
            title: Article title
            content: Article content
            symbols: List of symbols to look for
            
        Returns:
            List of keywords
        """
        keywords = []
        
        # Add symbols if found
        if symbols:
            text = f"{title} {content}".lower()
            for symbol in symbols:
                if symbol.lower() in text:
                    keywords.append(symbol.upper())
        
        # Add common financial terms
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'growth', 'decline',
            'stock', 'shares', 'dividend', 'market', 'trading', 'investment',
            'quarterly', 'annual', 'forecast', 'guidance', 'analyst',
            'upgrade', 'downgrade', 'buy', 'sell', 'hold'
        ]
        
        text = f"{title} {content}".lower()
        for term in financial_terms:
            if term in text:
                keywords.append(term.title())
        
        return list(set(keywords))
    
    def _calculate_relevance_score(self, title: str, content: str, symbols: Optional[List[str]] = None) -> float:
        """
        Calculate relevance score for article.
        
        Args:
            title: Article title
            content: Article content
            symbols: List of symbols to check against
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        text = f"{title} {content}".lower()
        
        # Check for symbol mentions
        if symbols:
            symbol_count = sum(1 for symbol in symbols if symbol.lower() in text)
            score += min(symbol_count * 0.3, 0.6)
        
        # Check for financial terms
        financial_terms = [
            'earnings', 'revenue', 'profit', 'stock', 'market', 'trading',
            'investment', 'financial', 'quarterly', 'annual'
        ]
        
        term_count = sum(1 for term in financial_terms if term in text)
        score += min(term_count * 0.1, 0.4)
        
        return min(score, 1.0)
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and preprocess article content.
        
        Args:
            content: Raw content
            
        Returns:
            Cleaned content
        """
        if not content:
            return ""
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', '', content)
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common news article endings
        content = re.sub(r'\[.*?\]', '', content)
        content = re.sub(r'Read more.*', '', content, flags=re.IGNORECASE)
        
        return content.strip()
    
    def _generate_article_id(self, title: str, published_at: datetime) -> str:
        """
        Generate unique article ID.
        
        Args:
            title: Article title
            published_at: Publication date
            
        Returns:
            Unique article ID
        """
        import hashlib
        
        # Create hash from title and date
        text = f"{title}_{published_at.isoformat()}"
        return hashlib.md5(text.encode()).hexdigest()
    
    def _get_mock_news(self, symbols: Optional[List[str]] = None, max_articles: int = 10) -> List[NewsArticle]:
        """
        Generate mock news articles for testing.
        
        Args:
            symbols: List of symbols
            max_articles: Number of articles to generate
            
        Returns:
            List of mock NewsArticle objects
        """
        mock_articles = []
        
        for i in range(min(max_articles, 10)):
            symbol = symbols[0] if symbols else "AAPL"
            
            article = NewsArticle(
                id=f"mock_article_{i}",
                title=f"Mock Financial News for {symbol} - Article {i+1}",
                content=f"This is a mock news article about {symbol}. It contains financial information and market analysis that would be relevant for investment decisions.",
                summary=f"Mock summary for {symbol} news article {i+1}",
                author="Mock Author",
                source="Mock News Source",
                url=f"https://mock-news.com/article/{i}",
                published_at=datetime.now() - timedelta(hours=i),
                sentiment_score=0.1 if i % 2 == 0 else -0.1,
                sentiment_type=SentimentType.POSITIVE if i % 2 == 0 else SentimentType.NEGATIVE,
                keywords=[symbol, "earnings", "market"],
                relevance_score=0.8
            )
            
            mock_articles.append(article)
        
        return mock_articles
