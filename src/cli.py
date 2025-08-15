"""
Command-line interface for Financial Analysis RAG System.
"""

import click
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent))

from rag.generation.rag_system import FinancialRAGSystem
from config import config

@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Financial Analysis RAG System CLI"""
    pass

@cli.command()
@click.option('--query', '-q', required=True, help='Financial analysis query')
@click.option('--symbols', '-s', help='Comma-separated stock symbols')
@click.option('--start-date', help='Start date (YYYY-MM-DD)')
@click.option('--end-date', help='End date (YYYY-MM-DD)')
@click.option('--include-news/--no-news', default=True, help='Include news analysis')
@click.option('--include-market-data/--no-market-data', default=True, help='Include market data analysis')
@click.option('--output', '-o', help='Output file for results (JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def analyze(query, symbols, start_date, end_date, include_news, include_market_data, output, verbose):
    """Analyze a financial query"""
    
    try:
        # Parse symbols
        symbol_list = None
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        # Parse dates
        start_dt = None
        end_dt = None
        
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Initialize RAG system
        if verbose:
            click.echo("Initializing RAG system...")
        
        rag_system = FinancialRAGSystem()
        
        # Perform analysis
        if verbose:
            click.echo(f"Analyzing query: {query}")
            if symbol_list:
                click.echo(f"Symbols: {', '.join(symbol_list)}")
        
        response = rag_system.analyze_query(
            query=query,
            symbols=symbol_list,
            start_date=start_dt,
            end_date=end_dt,
            include_news=include_news,
            include_market_data=include_market_data
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("📊 ANALYSIS RESULTS")
        click.echo("="*60)
        
        click.echo(f"Query: {response.query}")
        click.echo(f"Confidence Score: {response.confidence_score:.1%}")
        click.echo(f"Processing Time: {response.processing_time:.2f}s")
        click.echo(f"Sources Analyzed: {len(response.sources)}")
        
        click.echo("\n💡 Analysis Summary:")
        click.echo("-" * 40)
        click.echo(response.answer)
        
        # Display temporal context
        if response.temporal_context:
            click.echo("\n⏰ Temporal Context:")
            click.echo("-" * 40)
            click.echo(f"Timeframe: {response.temporal_context.timeframe}")
            click.echo(f"Trend: {response.temporal_context.trend_direction}")
            click.echo(f"Volatility Regime: {response.temporal_context.volatility_regime}")
        
        # Display risk assessment
        if response.risk_assessment:
            click.echo("\n⚠️ Risk Assessment:")
            click.echo("-" * 40)
            click.echo(f"Risk Level: {response.risk_assessment.risk_level.value.title()}")
            click.echo(f"Volatility: {response.risk_assessment.volatility:.1%}")
            click.echo(f"Beta: {response.risk_assessment.beta:.2f}")
            click.echo(f"VaR (95%): {response.risk_assessment.var_95:.1%}")
            click.echo(f"Sharpe Ratio: {response.risk_assessment.sharpe_ratio:.2f}")
            click.echo(f"Max Drawdown: {response.risk_assessment.max_drawdown:.1%}")
        
        # Display sources
        if response.sources:
            click.echo("\n📚 Sources:")
            click.echo("-" * 40)
            for i, source in enumerate(response.sources[:5], 1):
                if source.get('type') == 'news':
                    click.echo(f"{i}. News: {source.get('title', 'Unknown')}")
                else:
                    click.echo(f"{i}. Analysis: {source.get('symbol', 'Unknown')} - {source.get('insight_type', 'Unknown')}")
        
        # Save to file if requested
        if output:
            result_data = {
                'query': response.query,
                'answer': response.answer,
                'confidence_score': response.confidence_score,
                'processing_time': response.processing_time,
                'sources': response.sources,
                'temporal_context': response.temporal_context.dict() if response.temporal_context else None,
                'risk_assessment': response.risk_assessment.dict() if response.risk_assessment else None,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            click.echo(f"\n✅ Results saved to {output}")
        
        click.echo("\n" + "="*60)
        
    except Exception as e:
        click.echo(f"❌ Analysis failed: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--symbol', '-s', required=True, help='Stock symbol')
@click.option('--days', '-d', default=30, help='Number of days of data')
@click.option('--output', '-o', help='Output file (CSV)')
def market_data(symbol, days, output):
    """Get market data for a symbol"""
    
    try:
        from data.collectors.market_data_collector import MarketDataCollector
        
        click.echo(f"Fetching market data for {symbol}...")
        
        collector = MarketDataCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = collector.get_stock_data(symbol, start_date, end_date)
        
        if not data:
            click.echo(f"❌ No data found for {symbol}")
            return
        
        click.echo(f"✅ Retrieved {len(data)} data points")
        
        # Display summary
        first_price = data[0].close_price
        last_price = data[-1].close_price
        change = ((last_price - first_price) / first_price) * 100
        
        click.echo(f"Price: ${last_price:.2f} ({change:+.2f}%)")
        click.echo(f"Period: {data[0].date.strftime('%Y-%m-%d')} to {data[-1].date.strftime('%Y-%m-%d')}")
        
        # Save to CSV if requested
        if output:
            import pandas as pd
            
            df_data = []
            for md in data:
                df_data.append({
                    'date': md.date,
                    'open': md.open_price,
                    'high': md.high_price,
                    'low': md.low_price,
                    'close': md.close_price,
                    'volume': md.volume,
                    'adj_close': md.adjusted_close
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output, index=False)
            click.echo(f"✅ Data saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Failed to get market data: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--symbol', '-s', help='Stock symbol')
@click.option('--query', '-q', help='News search query')
@click.option('--days', '-d', default=7, help='Number of days to look back')
@click.option('--output', '-o', help='Output file (JSON)')
def news(symbol, query, days, output):
    """Get financial news"""
    
    try:
        from data.collectors.news_collector import NewsCollector
        
        collector = NewsCollector()
        
        if symbol:
            click.echo(f"Fetching news for {symbol}...")
            articles = collector.get_company_news(symbol, symbol, days)
        elif query:
            click.echo(f"Searching news for: {query}")
            articles = collector.get_financial_news(query, max_articles=50)
        else:
            click.echo("Fetching general market news...")
            articles = collector.get_market_news(days)
        
        if not articles:
            click.echo("❌ No news articles found")
            return
        
        click.echo(f"✅ Found {len(articles)} articles")
        
        # Display articles
        for i, article in enumerate(articles[:10], 1):
            click.echo(f"\n{i}. {article.title}")
            click.echo(f"   Source: {article.source}")
            click.echo(f"   Date: {article.published_at.strftime('%Y-%m-%d %H:%M')}")
            if article.sentiment_score:
                sentiment = "Positive" if article.sentiment_score > 0 else "Negative" if article.sentiment_score < 0 else "Neutral"
                click.echo(f"   Sentiment: {sentiment} ({article.sentiment_score:.2f})")
        
        # Save to JSON if requested
        if output:
            articles_data = []
            for article in articles:
                articles_data.append({
                    'title': article.title,
                    'content': article.content,
                    'source': article.source,
                    'url': article.url,
                    'published_at': article.published_at.isoformat(),
                    'sentiment_score': article.sentiment_score,
                    'sentiment_type': article.sentiment_type.value if article.sentiment_type else None,
                    'keywords': article.keywords,
                    'relevance_score': article.relevance_score
                })
            
            with open(output, 'w') as f:
                json.dump(articles_data, f, indent=2)
            
            click.echo(f"\n✅ Articles saved to {output}")
        
    except Exception as e:
        click.echo(f"❌ Failed to get news: {e}", err=True)
        sys.exit(1)

@cli.command()
@click.option('--symbol', '-s', required=True, help='Stock symbol')
@click.option('--days', '-d', default=365, help='Number of days of data')
def risk(symbol, days):
    """Calculate risk metrics for a symbol"""
    
    try:
        from data.collectors.market_data_collector import MarketDataCollector
        from analysis.risk.risk_analyzer import RiskAnalyzer
        
        click.echo(f"Calculating risk metrics for {symbol}...")
        
        # Get market data
        collector = MarketDataCollector()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = collector.get_stock_data(symbol, start_date, end_date)
        
        if not data:
            click.echo(f"❌ No data found for {symbol}")
            return
        
        # Calculate risk metrics
        risk_analyzer = RiskAnalyzer()
        risk_metrics = risk_analyzer.calculate_risk_metrics(data)
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo(f"📊 RISK METRICS FOR {symbol}")
        click.echo("="*50)
        
        click.echo(f"Risk Level: {risk_metrics.risk_level.value.title()}")
        click.echo(f"Volatility: {risk_metrics.volatility:.1%}")
        click.echo(f"Beta: {risk_metrics.beta:.2f}")
        click.echo(f"Sharpe Ratio: {risk_metrics.sharpe_ratio:.2f}")
        click.echo(f"VaR (95%): {risk_metrics.var_95:.1%}")
        click.echo(f"Max Drawdown: {risk_metrics.max_drawdown:.1%}")
        click.echo(f"Market Correlation: {risk_metrics.correlation_market:.2f}")
        
        # Risk interpretation
        click.echo("\n📋 Risk Interpretation:")
        if risk_metrics.risk_level.value in ['low', 'medium']:
            click.echo("✅ Acceptable risk profile")
        else:
            click.echo("⚠️ High risk - consider carefully")
        
        if risk_metrics.volatility > 0.25:
            click.echo("⚠️ High volatility detected")
        
        if risk_metrics.var_95 > 3.0:
            click.echo("⚠️ High Value at Risk")
        
        click.echo("="*50)
        
    except Exception as e:
        click.echo(f"❌ Risk calculation failed: {e}", err=True)
        sys.exit(1)

@cli.command()
def status():
    """Show system status"""
    
    try:
        click.echo("🏥 System Status Check")
        click.echo("="*40)
        
        # Check configuration
        config_valid = config.validate_config()
        click.echo(f"Configuration: {'✅ Valid' if config_valid else '❌ Invalid'}")
        
        # Check directories
        dirs_exist = all([
            config.DATA_DIR.exists(),
            config.RAW_DATA_DIR.exists(),
            config.PROCESSED_DATA_DIR.exists(),
            config.EMBEDDINGS_DIR.exists()
        ])
        click.echo(f"Directories: {'✅ Exist' if dirs_exist else '❌ Missing'}")
        
        # Check components
        try:
            from rag.embeddings.embedding_manager import EmbeddingManager
            embedding_manager = EmbeddingManager()
            cache_stats = embedding_manager.get_cache_stats()
            click.echo(f"Embedding Manager: ✅ Ready (Cache: {cache_stats.get('cache_files', 0)} files)")
        except Exception as e:
            click.echo(f"Embedding Manager: ❌ Error - {e}")
        
        try:
            from data.storage.vector_store import VectorStore
            vector_store = VectorStore()
            stats = vector_store.get_collection_stats()
            click.echo(f"Vector Store: ✅ Ready ({stats.get('total_documents', 0)} documents)")
        except Exception as e:
            click.echo(f"Vector Store: ❌ Error - {e}")
        
        try:
            rag_system = FinancialRAGSystem()
            click.echo("RAG System: ✅ Ready")
        except Exception as e:
            click.echo(f"RAG System: ❌ Error - {e}")
        
        click.echo("="*40)
        
    except Exception as e:
        click.echo(f"❌ Status check failed: {e}", err=True)
        sys.exit(1)

@cli.command()
def examples():
    """Show example queries"""
    
    click.echo("📝 Example Queries")
    click.echo("="*40)
    
    examples = [
        "Analyze Apple's financial performance and risk factors",
        "Compare Tesla and Ford stock performance",
        "What are the key risks for Microsoft stock?",
        "Generate a risk assessment for the technology sector",
        "Analyze market trends for AAPL, MSFT, and GOOGL",
        "What's the current sentiment around Bitcoin?",
        "Compare the volatility of tech stocks vs utility stocks",
        "Analyze the impact of recent news on Tesla stock"
    ]
    
    for i, example in enumerate(examples, 1):
        click.echo(f"{i}. {example}")
    
    click.echo("\n💡 Usage Examples:")
    click.echo("  python src/cli.py analyze -q 'Analyze AAPL performance' -s AAPL")
    click.echo("  python src/cli.py market-data -s AAPL -d 30")
    click.echo("  python src/cli.py news -s AAPL -d 7")
    click.echo("  python src/cli.py risk -s AAPL -d 365")
    click.echo("="*40)

if __name__ == '__main__':
    cli()
