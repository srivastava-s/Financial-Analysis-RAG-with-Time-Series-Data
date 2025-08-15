# Financial Analysis RAG System - Deployment Guide

This guide provides step-by-step instructions for deploying the Financial Analysis RAG System.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9 or higher
- Git
- Internet connection for API access
- At least 4GB RAM (8GB recommended)
- 2GB free disk space

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd financial-analysis-rag

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your API keys
# Required keys:
# - OPENAI_API_KEY
# - NEWS_API_KEY
```

### 4. Initialize System

```bash
# Initialize the system
python src/initialize_system.py

# Check system health
python src/initialize_system.py --action health
```

### 5. Run the Application

#### Option A: Streamlit Web Interface
```bash
streamlit run src/app.py
```
Open http://localhost:8501 in your browser.

#### Option B: Command Line Interface
```bash
# Analyze a query
python src/cli.py analyze -q "Analyze Apple's financial performance" -s AAPL

# Get market data
python src/cli.py market-data -s AAPL -d 30

# Get news
python src/cli.py news -s AAPL -d 7

# Calculate risk metrics
python src/cli.py risk -s AAPL -d 365
```

## 🌐 Production Deployment

### Docker Deployment

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and Run**
```bash
# Build image
docker build -t financial-rag .

# Run container
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=your_key \
  -e NEWS_API_KEY=your_key \
  financial-rag
```

### Cloud Deployment

#### Heroku
1. Create `Procfile`:
```
web: streamlit run src/app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Deploy:
```bash
heroku create your-app-name
heroku config:set OPENAI_API_KEY=your_key
heroku config:set NEWS_API_KEY=your_key
git push heroku main
```

#### AWS EC2
1. Launch EC2 instance (t3.medium or larger)
2. Install dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv
```

3. Clone and setup:
```bash
git clone <repository-url>
cd financial-analysis-rag
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. Configure environment variables and run:
```bash
export OPENAI_API_KEY=your_key
export NEWS_API_KEY=your_key
streamlit run src/app.py --server.port=8501 --server.address=0.0.0.0
```

#### Google Cloud Platform
1. Create Compute Engine instance
2. Follow AWS EC2 steps above
3. Configure firewall rules to allow port 8501

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM | Yes | - |
| `NEWS_API_KEY` | NewsAPI key for news data | Yes | - |
| `YAHOO_FINANCE_API_KEY` | Yahoo Finance API key | No | - |
| `ALPHA_VANTAGE_API_KEY` | Alpha Vantage API key | No | - |
| `FRED_API_KEY` | FRED API key | No | - |
| `DEBUG` | Enable debug mode | No | True |
| `LOG_LEVEL` | Logging level | No | INFO |
| `MAX_TOKENS` | Max tokens for LLM | No | 4000 |
| `TEMPERATURE` | LLM temperature | No | 0.7 |

### System Configuration

The system can be configured through the `src/config.py` file:

- **Data Collection**: Adjust time ranges, API limits
- **Analysis**: Modify risk thresholds, temporal windows
- **Caching**: Configure embedding and data caching
- **Performance**: Tune batch sizes, concurrency limits

## 📊 Monitoring and Maintenance

### Health Checks
```bash
# Check system health
python src/cli.py status

# Run tests
python tests/test_basic.py
```

### Logs
- Application logs: Check console output
- Error logs: Look for exceptions in stderr
- Performance logs: Monitor processing times

### Data Management
```bash
# Clear system data
python src/initialize_system.py --action clear

# Backup vector store
python -c "from src.data.storage.vector_store import VectorStore; VectorStore().backup_collection('backup.json')"
```

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Environment Variables**: Use secure environment variable management
3. **Network Security**: Restrict access to production instances
4. **Data Privacy**: Ensure compliance with data protection regulations
5. **Rate Limiting**: Implement API rate limiting to avoid costs

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure you're in the correct directory
   cd financial-analysis-rag
   
   # Check Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **API Key Issues**
   ```bash
   # Verify API keys are set
   echo $OPENAI_API_KEY
   echo $NEWS_API_KEY
   ```

3. **Memory Issues**
   - Reduce batch sizes in config
   - Use smaller embedding models
   - Increase system RAM

4. **Performance Issues**
   - Enable caching
   - Use GPU acceleration if available
   - Optimize database queries

### Getting Help

1. Check the logs for error messages
2. Run the test suite: `python tests/test_basic.py`
3. Verify system health: `python src/cli.py status`
4. Check configuration: `python src/initialize_system.py --action health`

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers for multiple instances
- Implement Redis for session management
- Use cloud databases for vector storage

### Vertical Scaling
- Increase CPU/RAM for better performance
- Use GPU instances for embedding generation
- Optimize database indexes

### Cost Optimization
- Use caching to reduce API calls
- Implement request batching
- Monitor API usage and costs

## 🔄 Updates and Maintenance

### Regular Maintenance
1. Update dependencies monthly
2. Monitor API usage and costs
3. Backup data regularly
4. Check system health weekly

### Updating the System
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinitialize if needed
python src/initialize_system.py --action health
```

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs and error messages
3. Run the test suite to identify problems
4. Consult the documentation and examples

---

**Note**: This system is for educational and research purposes. Always consult with qualified financial professionals before making investment decisions.
