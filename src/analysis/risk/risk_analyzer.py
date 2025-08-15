"""
Risk analysis module for financial data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime, timedelta
from scipy import stats
from scipy.stats import norm

from ...models import MarketData, RiskMetrics, RiskLevel
from ...config import config

logger = logging.getLogger(__name__)


class RiskAnalyzer:
    """Analyzer for financial risk metrics."""
    
    def __init__(self):
        """Initialize the risk analyzer."""
        self.risk_free_rate = config.RISK_FREE_RATE
        self.var_confidence_level = config.VAR_CONFIDENCE_LEVEL
    
    def calculate_risk_metrics(
        self, 
        market_data: List[MarketData],
        market_index_data: Optional[List[MarketData]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            market_data: Stock market data
            market_index_data: Market index data for beta calculation
            
        Returns:
            RiskMetrics object
        """
        try:
            if len(market_data) < 30:
                logger.warning("Insufficient data for risk analysis")
                return self._get_default_risk_metrics(market_data)
            
            # Convert to DataFrame
            df = self._market_data_to_dataframe(market_data)
            
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            # Basic risk metrics
            volatility = self._calculate_volatility(returns)
            var_95 = self._calculate_var(returns)
            max_drawdown = self._calculate_max_drawdown(df['close'])
            
            # Beta calculation
            beta = self._calculate_beta(returns, market_index_data)
            
            # Sharpe ratio
            sharpe_ratio = self._calculate_sharpe_ratio(returns)
            
            # Correlation with market
            correlation_market = self._calculate_market_correlation(returns, market_index_data)
            
            # Determine risk level
            risk_level = self._determine_risk_level(volatility, var_95, max_drawdown)
            
            return RiskMetrics(
                symbol=market_data[0].symbol,
                date=datetime.now(),
                volatility=volatility,
                beta=beta,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95,
                max_drawdown=max_drawdown,
                correlation_market=correlation_market,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics(market_data)
    
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
    
    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        try:
            # Daily volatility
            daily_vol = returns.std()
            
            # Annualize (assuming 252 trading days)
            annual_vol = daily_vol * np.sqrt(252)
            
            return float(annual_vol)
        except Exception as e:
            logger.warning(f"Error calculating volatility: {e}")
            return 0.0
    
    def _calculate_var(self, returns: pd.Series) -> float:
        """Calculate Value at Risk."""
        try:
            # Historical VaR
            var_percentile = (1 - self.var_confidence_level) * 100
            var = np.percentile(returns, var_percentile)
            
            # Convert to percentage
            var_percentage = abs(var) * 100
            
            return float(var_percentage)
        except Exception as e:
            logger.warning(f"Error calculating VaR: {e}")
            return 0.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            # Calculate cumulative maximum
            cumulative_max = prices.expanding().max()
            
            # Calculate drawdown
            drawdown = (prices - cumulative_max) / cumulative_max
            
            # Get maximum drawdown
            max_drawdown = drawdown.min()
            
            return float(abs(max_drawdown) * 100)  # Convert to percentage
        except Exception as e:
            logger.warning(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _calculate_beta(
        self, 
        stock_returns: pd.Series, 
        market_index_data: Optional[List[MarketData]] = None
    ) -> float:
        """Calculate beta relative to market index."""
        try:
            if market_index_data is None or len(market_index_data) < 30:
                # Use default beta if no market data
                return 1.0
            
            # Convert market data to DataFrame
            market_df = self._market_data_to_dataframe(market_index_data)
            market_returns = market_df['close'].pct_change().dropna()
            
            # Align returns
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 30:
                return 1.0
            
            stock_ret = aligned_data.iloc[:, 0]
            market_ret = aligned_data.iloc[:, 1]
            
            # Calculate beta
            covariance = np.cov(stock_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            
            return float(beta)
            
        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return 1.0
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            # Calculate excess returns
            excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
            
            # Calculate Sharpe ratio
            if returns.std() == 0:
                return 0.0
            
            sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(252)
            
            return float(sharpe_ratio)
            
        except Exception as e:
            logger.warning(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _calculate_market_correlation(
        self, 
        stock_returns: pd.Series, 
        market_index_data: Optional[List[MarketData]] = None
    ) -> float:
        """Calculate correlation with market index."""
        try:
            if market_index_data is None or len(market_index_data) < 30:
                return 0.5  # Default correlation
            
            # Convert market data to DataFrame
            market_df = self._market_data_to_dataframe(market_index_data)
            market_returns = market_df['close'].pct_change().dropna()
            
            # Align returns
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 30:
                return 0.5
            
            stock_ret = aligned_data.iloc[:, 0]
            market_ret = aligned_data.iloc[:, 1]
            
            # Calculate correlation
            correlation = stock_ret.corr(market_ret)
            
            return float(correlation) if not pd.isna(correlation) else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating market correlation: {e}")
            return 0.5
    
    def _determine_risk_level(
        self, 
        volatility: float, 
        var_95: float, 
        max_drawdown: float
    ) -> RiskLevel:
        """Determine overall risk level."""
        try:
            # Score based on different risk metrics
            vol_score = 0
            if volatility < 0.15:  # 15% annualized volatility
                vol_score = 1
            elif volatility < 0.25:
                vol_score = 2
            elif volatility < 0.35:
                vol_score = 3
            else:
                vol_score = 4
            
            var_score = 0
            if var_95 < 2.0:  # 2% daily VaR
                var_score = 1
            elif var_95 < 3.5:
                var_score = 2
            elif var_95 < 5.0:
                var_score = 3
            else:
                var_score = 4
            
            drawdown_score = 0
            if max_drawdown < 10:  # 10% max drawdown
                drawdown_score = 1
            elif max_drawdown < 20:
                drawdown_score = 2
            elif max_drawdown < 30:
                drawdown_score = 3
            else:
                drawdown_score = 4
            
            # Average score
            avg_score = (vol_score + var_score + drawdown_score) / 3
            
            # Determine risk level
            if avg_score <= 1.5:
                return RiskLevel.LOW
            elif avg_score <= 2.5:
                return RiskLevel.MEDIUM
            elif avg_score <= 3.5:
                return RiskLevel.HIGH
            else:
                return RiskLevel.VERY_HIGH
                
        except Exception as e:
            logger.warning(f"Error determining risk level: {e}")
            return RiskLevel.MEDIUM
    
    def _get_default_risk_metrics(self, market_data: List[MarketData]) -> RiskMetrics:
        """Get default risk metrics when insufficient data."""
        symbol = market_data[0].symbol if market_data else "UNKNOWN"
        
        return RiskMetrics(
            symbol=symbol,
            date=datetime.now(),
            volatility=0.20,  # 20% default volatility
            beta=1.0,
            sharpe_ratio=0.0,
            var_95=3.0,  # 3% default VaR
            max_drawdown=15.0,  # 15% default max drawdown
            correlation_market=0.5,
            risk_level=RiskLevel.MEDIUM
        )
    
    def calculate_portfolio_risk(
        self, 
        portfolio_data: Dict[str, List[MarketData]],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Calculate portfolio-level risk metrics.
        
        Args:
            portfolio_data: Dictionary mapping symbols to market data
            weights: Portfolio weights (optional)
            
        Returns:
            Dictionary with portfolio risk metrics
        """
        try:
            if not portfolio_data:
                return {'error': 'No portfolio data provided'}
            
            # Calculate individual asset returns
            asset_returns = {}
            for symbol, market_data in portfolio_data.items():
                df = self._market_data_to_dataframe(market_data)
                returns = df['close'].pct_change().dropna()
                asset_returns[symbol] = returns
            
            # Align all returns
            aligned_returns = pd.DataFrame(asset_returns).dropna()
            
            if len(aligned_returns) < 30:
                return {'error': 'Insufficient data for portfolio analysis'}
            
            # Set equal weights if not provided
            if weights is None:
                n_assets = len(asset_returns)
                weights = {symbol: 1.0/n_assets for symbol in asset_returns.keys()}
            
            # Calculate portfolio returns
            portfolio_returns = pd.Series(0.0, index=aligned_returns.index)
            for symbol, weight in weights.items():
                if symbol in aligned_returns.columns:
                    portfolio_returns += weight * aligned_returns[symbol]
            
            # Calculate portfolio risk metrics
            portfolio_volatility = self._calculate_volatility(portfolio_returns)
            portfolio_var = self._calculate_var(portfolio_returns)
            portfolio_sharpe = self._calculate_sharpe_ratio(portfolio_returns)
            
            # Calculate correlation matrix
            correlation_matrix = aligned_returns.corr()
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'portfolio_var_95': portfolio_var,
                'portfolio_sharpe_ratio': portfolio_sharpe,
                'correlation_matrix': correlation_matrix.to_dict(),
                'weights': weights,
                'num_assets': len(asset_returns),
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {e}")
            return {'error': str(e)}
    
    def stress_test(
        self, 
        market_data: List[MarketData],
        scenarios: List[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform stress testing on portfolio.
        
        Args:
            market_data: Market data for stress testing
            scenarios: List of stress scenarios
            
        Returns:
            Dictionary with stress test results
        """
        try:
            if not scenarios:
                # Default scenarios
                scenarios = [
                    {'name': 'Market Crash', 'shock': -0.20},
                    {'name': 'Moderate Decline', 'shock': -0.10},
                    {'name': 'Volatility Spike', 'shock': 0.50},
                    {'name': 'Recovery', 'shock': 0.15}
                ]
            
            df = self._market_data_to_dataframe(market_data)
            current_price = df['close'].iloc[-1]
            
            stress_results = {}
            
            for scenario in scenarios:
                scenario_name = scenario['name']
                shock = scenario['shock']
                
                # Calculate stressed price
                if 'shock' in scenario:
                    stressed_price = current_price * (1 + shock)
                    
                    # Calculate potential loss
                    potential_loss = (current_price - stressed_price) / current_price * 100
                    
                    stress_results[scenario_name] = {
                        'current_price': float(current_price),
                        'stressed_price': float(stressed_price),
                        'potential_loss_percent': float(potential_loss),
                        'shock_magnitude': float(shock * 100)
                    }
            
            return {
                'stress_scenarios': stress_results,
                'analysis_date': datetime.now().isoformat(),
                'base_price': float(current_price)
            }
            
        except Exception as e:
            logger.error(f"Error in stress testing: {e}")
            return {'error': str(e)}
