# Financial Data ETL Pipeline

## Design Architecture

**Modular Components:**
- **ETL Pipeline**: Advanced data cleaning with multi-method outlier detection
- **REST API**: FastAPI with auto-documentation and type safety
- **Risk Analyzer**: Portfolio risk assessment with severity classification  
- **AI Predictor**: Time series forecasting with interactive visualizations

## Key Trade-offs

**Performance vs Accuracy:** Multi-method outlier detection (~2s processing) for robust financial data cleaning

**Complexity vs Maintainability:** 4 separate modules for clearer responsibilities vs single monolithic file

**Memory vs Speed:** In-memory caching for sub-100ms API responses vs lower memory usage

**AI Accuracy vs Simplicity:** Advanced ML forecasting vs simple linear models - chose sophisticated AI with linear fallback

## Core Assumptions

**Financial Standards:** 4% risk-free rate, 252 trading days/year, 3σ outlier threshold

**Data Constraints:** 30-day historical window limits AI prediction accuracy - longer datasets would improve results

**AI Limitations:** Models analyze patterns/trends but cannot predict external market disruptions or events not in training data

**Usage Patterns:** Users need both individual fund metrics and portfolio-level analytics

## AI Forecasting Approach

**Method:** Machine learning algorithms detect trends, seasonality, and volatility cycles in NAV data

**Limitations:** 
- Only 30 days of training data available
- Cannot capture long-term market cycles  
- Predictions are estimates with uncertainty bounds
- Best for funds with consistent patterns

**Recommendations:** Use 1-3 years of data and external market indicators for production deployment

## Quick Start

```bash
pip install -r core_requirements.txt
python core_demo.py  # Demo all features
python core_api.py   # Start API server
```

**API Documentation:** http://localhost:8000/docs

**Generate HTML charts:** 
```bash
curl "http://localhost:8000/predict/F003/html" -o html/F003_prediction.html
```