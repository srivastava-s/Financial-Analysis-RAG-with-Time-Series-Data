"""
Configuration management for the Financial Analysis RAG System.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Central configuration class for the Financial Analysis RAG System."""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EMBEDDINGS_DIR = DATA_DIR / "embeddings"
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
    
    # Financial Data APIs
    YAHOO_FINANCE_API_KEY: str = os.getenv("YAHOO_FINANCE_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    
    # News and Media APIs
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")
    
    # Database Configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(DATA_DIR / "chroma_db"))
    SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(DATA_DIR / "financial_analysis.db"))
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Application Configuration
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Risk Analysis Configuration
    VAR_CONFIDENCE_LEVEL: float = float(os.getenv("VAR_CONFIDENCE_LEVEL", "0.95"))
    MAX_LOOKBACK_DAYS: int = int(os.getenv("MAX_LOOKBACK_DAYS", "252"))
    VOLATILITY_WINDOW: int = int(os.getenv("VOLATILITY_WINDOW", "30"))
    CORRELATION_THRESHOLD: float = float(os.getenv("CORRELATION_THRESHOLD", "0.7"))
    
    # Data Collection Configuration
    MAX_NEWS_ARTICLES: int = int(os.getenv("MAX_NEWS_ARTICLES", "100"))
    MAX_HISTORICAL_DAYS: int = int(os.getenv("MAX_HISTORICAL_DAYS", "365"))
    UPDATE_FREQUENCY_HOURS: int = int(os.getenv("UPDATE_FREQUENCY_HOURS", "24"))
    
    # Security Configuration
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-this")
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-jwt-secret-key-change-this")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Monitoring Configuration
    PROMETHEUS_PORT: int = int(os.getenv("PROMETHEUS_PORT", "9090"))
    METRICS_ENABLED: bool = os.getenv("METRICS_ENABLED", "True").lower() == "true"
    
    # Model Configuration
    EMBEDDING_DIMENSION: int = 1536  # OpenAI ada-002 dimension
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Financial Analysis Settings
    DEFAULT_CURRENCY: str = "USD"
    DEFAULT_MARKET: str = "US"
    RISK_FREE_RATE: float = 0.02  # 2% risk-free rate
    
    # Time Series Analysis
    DEFAULT_TIMEFRAME: str = "1d"
    FORECAST_PERIODS: int = 30
    SEASONALITY_PERIODS: int = 252  # Trading days in a year
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that all required configuration is present."""
        required_keys = [
            "OPENAI_API_KEY",
            "NEWS_API_KEY"
        ]
        
        missing_keys = []
        for key in required_keys:
            if not getattr(cls, key):
                missing_keys.append(key)
        
        if missing_keys:
            print(f"Missing required configuration keys: {missing_keys}")
            return False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EMBEDDINGS_DIR,
            Path(cls.CHROMA_DB_PATH).parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()
