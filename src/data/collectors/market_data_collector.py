"""
Market data collector for financial analysis RAG system.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ...models import MarketData, DataSource, FinancialMetrics
from ...config import config

logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collector for market data from various sources."""
    
    def __init__(self):
        self.session = None
        self._setup_session()
    
    def _setup_session(self):
        """Setup yfinance session with custom headers."""
        try:
            # Configure yfinance session
            yf.pdr_override()
            logger.info("Market data collector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize market data collector: {e}")
    
    def get_stock_data(
        self, 
        symbol: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> List[MarketData]:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval ('1d', '1h', '5m', etc.)
            
        Returns:
            List of MarketData objects
        """
        try:
            # Set default dates if not provided
            if end_date is None:
                end_date = datetime.now()
            if start_date is None:
                start_date = end_date - timedelta(days=config.MAX_HISTORICAL_DAYS)
            
            logger.info(f"Fetching {interval} data for {symbol} from {start_date} to {end_date}")
            
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return []
            
            # Convert to MarketData objects
            market_data = []
            for index, row in df.iterrows():
                market_data.append(MarketData(
                    symbol=symbol.upper(),
                    date=index.to_pydatetime(),
                    open_price=float(row['Open']),
                    high_price=float(row['High']),
                    low_price=float(row['Low']),
                    close_price=float(row['Close']),
                    volume=int(row['Volume']),
                    adjusted_close=float(row['Close']),  # Already adjusted
                    source=DataSource.YAHOO_FINANCE
                ))
            
            logger.info(f"Successfully fetched {len(market_data)} data points for {symbol}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return []
    
    def get_multiple_stocks_data(
        self, 
        symbols: List[str], 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        interval: str = "1d",
        max_workers: int = 5
    ) -> Dict[str, List[MarketData]]:
        """
        Fetch data for multiple stocks concurrently.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data interval
            max_workers: Maximum number of concurrent workers
            
        Returns:
            Dictionary mapping symbols to their MarketData lists
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(
                    self.get_stock_data, 
                    symbol, 
                    start_date, 
                    end_date, 
                    interval
                ): symbol for symbol in symbols
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data = future.result()
                    results[symbol] = data
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    results[symbol] = []
        
        return results
    
    def get_financial_metrics(self, symbol: str) -> Optional[FinancialMetrics]:
        """
        Fetch financial metrics for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            FinancialMetrics object or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            metrics = FinancialMetrics(
                symbol=symbol.upper(),
                date=datetime.now(),
                market_cap=info.get('marketCap'),
                pe_ratio=info.get('trailingPE'),
                pb_ratio=info.get('priceToBook'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                quick_ratio=info.get('quickRatio'),
                roe=info.get('returnOnEquity'),
                roa=info.get('returnOnAssets'),
                profit_margin=info.get('profitMargins'),
                revenue_growth=info.get('revenueGrowth'),
                earnings_growth=info.get('earningsGrowth')
            )
            
            logger.info(f"Successfully fetched financial metrics for {symbol}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching financial metrics for {symbol}: {e}")
            return None
    
    def get_market_index_data(
        self, 
        index_symbol: str = "^GSPC",  # S&P 500
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[MarketData]:
        """
        Fetch market index data (e.g., S&P 500, NASDAQ).
        
        Args:
            index_symbol: Index symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            List of MarketData objects
        """
        return self.get_stock_data(index_symbol, start_date, end_date)
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time stock price.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price or None
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error fetching real-time price for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract relevant information
            stock_info = {
                'symbol': symbol.upper(),
                'name': info.get('longName'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'volume': info.get('volume'),
                'avg_volume': info.get('averageVolume'),
                'currency': info.get('currency'),
                'exchange': info.get('exchange'),
                'country': info.get('country')
            }
            
            return stock_info
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}")
            return {}
    
    def get_earnings_calendar(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get earnings calendar for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of earnings events
        """
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is None or calendar.empty:
                return []
            
            earnings_events = []
            for index, row in calendar.iterrows():
                earnings_events.append({
                    'date': index.to_pydatetime(),
                    'estimate': row.get('Earnings Average', 0),
                    'actual': row.get('Earnings Actual', 0),
                    'surprise': row.get('Earnings Surprise', 0),
                    'surprise_percent': row.get('Earnings Surprise %', 0)
                })
            
            return earnings_events
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
            return []
    
    def get_analyst_recommendations(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get analyst recommendations for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of analyst recommendations
        """
        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations
            
            if recommendations is None or recommendations.empty:
                return []
            
            recs = []
            for index, row in recommendations.iterrows():
                recs.append({
                    'date': index.to_pydatetime(),
                    'firm': row.get('Firm', ''),
                    'to_grade': row.get('To Grade', ''),
                    'from_grade': row.get('From Grade', ''),
                    'action': row.get('Action', '')
                })
            
            return recs
            
        except Exception as e:
            logger.error(f"Error fetching analyst recommendations for {symbol}: {e}")
            return []
    
    def get_insider_transactions(self, symbol: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a stock.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            List of insider transactions
        """
        try:
            ticker = yf.Ticker(symbol)
            insider = ticker.insider_transactions
            
            if insider is None or insider.empty:
                return []
            
            transactions = []
            for index, row in insider.iterrows():
                transactions.append({
                    'date': index.to_pydatetime(),
                    'insider_name': row.get('Insider Name', ''),
                    'title': row.get('Title', ''),
                    'transaction_type': row.get('Transaction Type', ''),
                    'shares': row.get('Shares', 0),
                    'value': row.get('Value', 0)
                })
            
            return transactions
            
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {e}")
            return []
    
    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol exists and is valid.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('regularMarketPrice') is not None
        except Exception:
            return False
