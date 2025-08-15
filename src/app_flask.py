"""
Flask-based web application for Financial Analysis RAG System.
This version works with existing Flask installation.
"""

from flask import Flask, render_template_string, request, jsonify
import json
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Analysis RAG System</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5rem;
            font-weight: 300;
        }
        .content {
            padding: 30px;
        }
        .form-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }
        input, textarea, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            box-sizing: border-box;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .stock-card {
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px 15px;
            background: #e3f2fd;
            border-radius: 5px;
            min-width: 120px;
        }
        .metric-value {
            font-size: 1.2em;
            font-weight: bold;
            color: #1976d2;
        }
        .risk-high { color: #d32f2f; }
        .risk-medium { color: #f57c00; }
        .risk-low { color: #388e3c; }
        .insights {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .chart-placeholder {
            background: #f0f0f0;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            margin: 20px 0;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Financial Analysis RAG System</h1>
            <p>Comprehensive financial analysis and investment insights</p>
        </div>
        
        <div class="content">
            <form method="POST" action="/analyze">
                <div class="form-section">
                    <h3>🔧 Analysis Configuration</h3>
                    
                    <div class="form-group">
                        <label for="query">Financial Query:</label>
                        <textarea name="query" id="query" rows="3" placeholder="e.g., Analyze Apple's financial performance and risk factors"></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label for="symbols">Stock Symbols (comma-separated):</label>
                        <input type="text" name="symbols" id="symbols" placeholder="AAPL, MSFT, GOOGL" value="{{ symbols or '' }}">
                    </div>
                    
                    <div class="form-group">
                        <label for="start_date">Start Date:</label>
                        <input type="date" name="start_date" id="start_date" value="{{ start_date or '' }}">
                    </div>
                    
                    <div class="form-group">
                        <label for="end_date">End Date:</label>
                        <input type="date" name="end_date" id="end_date" value="{{ end_date or '' }}">
                    </div>
                    
                    <button type="submit">🚀 Analyze</button>
                </div>
            </form>
            
            {% if results %}
            <div class="results">
                <h2>📊 Financial Analysis Results</h2>
                
                {% for symbol, data in results.items() %}
                <div class="stock-card">
                    <h3>{{ symbol }}</h3>
                    
                    <div class="metric">
                        <div>Current Price</div>
                        <div class="metric-value">${{ "%.2f"|format(data.current_price) }}</div>
                    </div>
                    
                    <div class="metric">
                        <div>Price Change</div>
                        <div class="metric-value">{{ "%.2f"|format(data.price_change) }} ({{ "%.1f"|format(data.price_change_pct) }}%)</div>
                    </div>
                    
                    <div class="metric">
                        <div>Volatility</div>
                        <div class="metric-value">${{ "%.2f"|format(data.volatility) }}</div>
                    </div>
                    
                    <div class="metric">
                        <div>Trend</div>
                        <div class="metric-value">{{ data.trend }}</div>
                    </div>
                    
                    <div class="metric">
                        <div>Risk Level</div>
                        <div class="metric-value {{ 'risk-high' if data.risk_level == 'High' else 'risk-medium' if data.risk_level == 'Medium' else 'risk-low' }}">
                            {{ data.risk_level }}
                        </div>
                    </div>
                </div>
                {% endfor %}
                
                <div class="insights">
                    <h3>💡 Investment Insights</h3>
                    {% for insight in insights %}
                    <p>• {{ insight }}</p>
                    {% endfor %}
                </div>
                
                <div class="chart-placeholder">
                    📈 Price Trend Chart (Interactive charts would be displayed here)
                </div>
            </div>
            {% endif %}
            
            {% if not results %}
            <div class="results">
                <h2>🎯 Welcome to Financial Analysis RAG System</h2>
                <p>This system provides comprehensive financial analysis and insights for investment decision-making.</p>
                
                <h3>🚀 Getting Started:</h3>
                <ol>
                    <li><strong>Enter your financial query</strong> in the form above</li>
                    <li><strong>Select stock symbols</strong> (e.g., AAPL, MSFT, GOOGL)</li>
                    <li><strong>Choose date range</strong> for analysis</li>
                    <li><strong>Click Analyze</strong> to get insights</li>
                </ol>
                
                <h3>📊 What You'll Get:</h3>
                <ul>
                    <li><strong>Price Analysis:</strong> Current prices, changes, and trends</li>
                    <li><strong>Risk Assessment:</strong> Volatility and risk level indicators</li>
                    <li><strong>Investment Insights:</strong> Actionable recommendations</li>
                    <li><strong>Visual Charts:</strong> Price trend visualizations</li>
                </ul>
                
                <h3>💡 Sample Queries:</h3>
                <ul>
                    <li>"Analyze Apple's financial performance and risk factors"</li>
                    <li>"Compare Microsoft vs Google stock performance"</li>
                    <li>"What are the key risk factors for Tesla stock?"</li>
                </ul>
            </div>
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

def generate_sample_financial_data():
    """Generate sample financial data for demonstration."""
    base_prices = {
        'AAPL': 150,
        'MSFT': 300,
        'GOOGL': 2800,
        'TSLA': 250,
        'AMZN': 3300
    }
    
    data = {}
    for symbol, base_price in base_prices.items():
        # Generate realistic price movements
        current_price = base_price + random.uniform(-20, 30)
        price_change = random.uniform(-15, 25)
        price_change_pct = (price_change / base_price) * 100
        volatility = random.uniform(5, 15)
        
        data[symbol] = {
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volatility': volatility,
            'trend': 'Bullish' if price_change > 0 else 'Bearish',
            'risk_level': 'High' if volatility > 10 else 'Medium' if volatility > 5 else 'Low'
        }
    
    return data

def analyze_financial_performance(symbols, start_date, end_date):
    """Analyze financial performance for given symbols."""
    if not symbols:
        return {}
    
    # Generate sample data
    all_data = generate_sample_financial_data()
    
    # Filter by requested symbols
    results = {}
    for symbol in symbols:
        symbol = symbol.strip().upper()
        if symbol in all_data:
            results[symbol] = all_data[symbol]
    
    return results

def generate_insights(results):
    """Generate investment insights based on analysis."""
    insights = []
    
    for symbol, data in results.items():
        if data['trend'] == 'Bullish':
            insights.append(f"{symbol} shows a bullish trend with {data['price_change_pct']:.1f}% growth. Consider holding or buying on dips.")
        else:
            insights.append(f"{symbol} shows a bearish trend with {abs(data['price_change_pct']):.1f}% decline. Monitor for reversal signals.")
        
        if data['risk_level'] == 'High':
            insights.append(f"{symbol} has high volatility (${data['volatility']:.2f}). Implement proper risk management.")
        elif data['risk_level'] == 'Medium':
            insights.append(f"{symbol} has moderate volatility. Suitable for balanced portfolios.")
        else:
            insights.append(f"{symbol} has low volatility, making it suitable for conservative investors.")
    
    return insights

@app.route('/')
def index():
    """Main page."""
    return render_template_string(HTML_TEMPLATE, results=None, insights=None, symbols=None, start_date=None, end_date=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze financial data."""
    query = request.form.get('query', '')
    symbols_input = request.form.get('symbols', '')
    start_date = request.form.get('start_date', '')
    end_date = request.form.get('end_date', '')
    
    # Parse symbols
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()] if symbols_input else []
    
    # Set default dates if not provided
    if not start_date:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Perform analysis
    results = analyze_financial_performance(symbols, start_date, end_date)
    insights = generate_insights(results) if results else []
    
    return render_template_string(
        HTML_TEMPLATE, 
        results=results, 
        insights=insights, 
        symbols=symbols_input,
        start_date=start_date,
        end_date=end_date
    )

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis."""
    data = request.get_json()
    symbols = data.get('symbols', [])
    start_date = data.get('start_date', '')
    end_date = data.get('end_date', '')
    
    results = analyze_financial_performance(symbols, start_date, end_date)
    insights = generate_insights(results) if results else []
    
    return jsonify({
        'results': results,
        'insights': insights,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Financial Analysis RAG System...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🔧 Press Ctrl+C to stop the server")
    app.run(debug=True, host='0.0.0.0', port=5000)
