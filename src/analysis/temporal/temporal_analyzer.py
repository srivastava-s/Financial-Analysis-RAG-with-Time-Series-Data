"""
Temporal analysis module for financial time series data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from scipy import stats

from ...models import MarketData, TemporalContext
from ...config import config

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """Analyzer for temporal patterns in financial data."""
    
    def __init__(self):
        """Initialize the temporal analyzer."""
        self.min_data_points = 10
    
    def analyze_temporal_patterns(
        self, 
        market_data: List[MarketData]
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in market data.
        
        Args:
            market_data: List of market data points
            
        Returns:
            Dictionary with analysis results
        """
        try:
            if len(market_data) < self.min_data_points:
                return self._get_basic_analysis(market_data)
            
            # Convert to DataFrame
            df = self._market_data_to_dataframe(market_data)
            
            return {
                'basic_stats': self._calculate_basic_stats(df),
                'trend_analysis': self._analyze_trends(df),
                'volatility_analysis': self._analyze_volatility(df),
                'temporal_context': self._create_temporal_context(df)
            }
                
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
            return {'error': str(e)}
    
    def _market_data_to_dataframe(self, market_data: List[MarketData]) -> pd.DataFrame:
        """Convert MarketData list to pandas DataFrame."""
        data = []
        for md in market_data:
            data.append({
                'date': md.date,
                'open': md.open_price,
                'high': md.high_price,
                'low': md.low_price,
                'close': md.close_price,
                'volume': md.volume,
                'adj_close': md.adjusted_close
            })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _calculate_basic_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic statistical measures."""
        returns = df['close'].pct_change().dropna()
        
        return {
            'total_days': len(df),
            'start_date': df.index[0].isoformat(),
            'end_date': df.index[-1].isoformat(),
            'price_range': {
                'min': float(df['low'].min()),
                'max': float(df['high'].max()),
                'current': float(df['close'].iloc[-1])
            },
            'returns': {
                'mean': float(returns.mean()),
                'std': float(returns.std()),
                'total_return': float((df['close'].iloc[-1] / df['close'].iloc[0]) - 1)
            }
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze price trends."""
        close_prices = df['close']
        
        # Linear trend
        x = np.arange(len(close_prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, close_prices)
        
        # Moving averages
        ma_20 = close_prices.rolling(window=min(20, len(close_prices))).mean()
        
        # Determine trend direction
        if slope > 0:
            trend_direction = "uptrend"
        elif slope < 0:
            trend_direction = "downtrend"
        else:
            trend_direction = "sideways"
        
        current_price = close_prices.iloc[-1]
        above_ma20 = current_price > ma_20.iloc[-1] if not pd.isna(ma_20.iloc[-1]) else False
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': float(abs(r_value)),
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'moving_average_20': float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else None,
            'above_ma20': above_ma20
        }
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility patterns."""
        returns = df['close'].pct_change().dropna()
        
        # Rolling volatility
        volatility_window = min(config.VOLATILITY_WINDOW, len(returns))
        rolling_vol = returns.rolling(window=volatility_window).std()
        
        vol_mean = rolling_vol.mean()
        current_vol = rolling_vol.iloc[-1] if not pd.isna(rolling_vol.iloc[-1]) else vol_mean
        
        if current_vol > vol_mean * 1.2:
            vol_regime = "high"
        elif current_vol < vol_mean * 0.8:
            vol_regime = "low"
        else:
            vol_regime = "normal"
        
        return {
            'current_volatility': float(current_vol),
            'volatility_regime': vol_regime,
            'volatility_mean': float(vol_mean)
        }
    
    def _create_temporal_context(self, df: pd.DataFrame) -> TemporalContext:
        """Create temporal context from data."""
        start_date = df.index[0]
        end_date = df.index[-1]
        
        # Determine timeframe
        date_diff = end_date - start_date
        if date_diff.days <= 7:
            timeframe = "1h"
        elif date_diff.days <= 30:
            timeframe = "1d"
        else:
            timeframe = "1w"
        
        # Determine trend direction
        slope = self._analyze_trends(df)['slope']
        if slope > 0.01:
            trend_direction = "uptrend"
        elif slope < -0.01:
            trend_direction = "downtrend"
        else:
            trend_direction = "sideways"
        
        # Determine volatility regime
        vol_analysis = self._analyze_volatility(df)
        volatility_regime = vol_analysis['volatility_regime']
        
        return TemporalContext(
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            trend_direction=trend_direction,
            volatility_regime=volatility_regime
        )
    
    def _get_basic_analysis(self, market_data: List[MarketData]) -> Dict[str, Any]:
        """Get basic analysis for insufficient data."""
        if not market_data:
            return {'error': 'No market data provided'}
        
        df = self._market_data_to_dataframe(market_data)
        
        return {
            'basic_stats': self._calculate_basic_stats(df),
            'temporal_context': self._create_temporal_context(df),
            'message': f'Limited analysis due to insufficient data points ({len(market_data)})'
        }
