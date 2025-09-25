
A comprehensive ETL pipeline for fund NAV (Net Asset Value) data with analytics, risk analysis, and NAV prediction.

1. Data Cleaning
- Date Parsing & Normalization: String to datetime conversion with error handling
- Missing NAV Handling: Forward-fill → Backward-fill → Linear interpolation  
- Duplicate NAV Removal: By fund_id and date combinations
- Fund-level Data Consistency: Name validation and cleaning
- Advanced Outlier Detection: IQR + Z-score + percentage change methods

2. Analytics
- CAGR: Compounded Annual Growth Rate calculation
- Volatility: Standard deviation of daily returns 
- Sharpe Ratio: Risk-adjusted returns (4% risk-free rate)
- Additional Metrics: Average daily returns, total returns
- Summary Table: fund_id → metrics output format

3. Interface
- GET /funds/summary: Returns complete analytics as JSON
- GET /funds/{fund_id}: Returns metrics for specific fund
- FastAPI Framework: Auto-generated documentation at /docs
- Type Safety: Pydantic models for all endpoints

4. Portfolio-level Metrics
- Weighted CAGR: Portfolio-level returns calculation
- Portfolio Volatility: Risk aggregation (simplified correlation model)
- Portfolio Sharpe Ratio: Portfolio risk-adjusted performance
- Custom Weights: Calculate metrics with user-provided allocations

Portfolio Risk Warnings
- Volatility Risk Detection: Identifies funds with excessive volatility (>15%)
- Performance Risk Analysis: Flags negative returns and poor Sharpe ratios
- Efficiency Risk Assessment: Detects poor risk-adjusted returns
- Portfolio-level Risk Scoring: Overall portfolio risk assessment

NAV Prediction (Prophet-based)
- 15-Day Forecasting: Prophet-based time series prediction
- Predicted Metrics: CAGR, Volatility, Sharpe Ratio for forecast period
- Interactive HTML Visualization: Plotly charts with current vs predicted NAV
- Confidence Intervals: Model uncertainty quantification
- Portfolio Predictions: Batch prediction for multiple funds

Quick Start

Install Dependencies
```bash
pip install -r core_requirements.txt
```

Run Demo
```bash
python core_demo.py
```

Start API Server
```bash
python core_api.py
```

Access API
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health
- System Status: http://localhost:8000/status

API Usage Examples

Core Endpoints

Get All Fund Metrics
```bash
curl http://localhost:8000/funds/summary
```

Get Specific Fund Details
```bash
curl http://localhost:8000/funds/F001
```

Response:
```json
{
  "fund_id": "F001",
  "fund_name": "Axis Bluechip Fund", 
  "cagr": -20.48,
  "volatility": 7.2,
  "sharpe_ratio": -2.716,
  "avg_daily_return": -0.0892,
  "total_return": -20.48,
  "data_points": 31,
  "date_range": {
    "start": "2022-01-01",
    "end": "2022-01-31"
  }
}
```

#### Portfolio Metrics with Custom Weights
```bash
curl -X POST http://localhost:8000/portfolio/calculate \
  -H "Content-Type: application/json" \
  -d '{"F001": 0.3, "F003": 0.4, "F006": 0.3}'
```

Response:
```json
{
  "weighted_cagr": 39.01,
  "portfolio_volatility": 4.44,
  "portfolio_sharpe_ratio": 7.8869,
  "num_funds": 3,
  "valid_funds": 3,
  "weights": {"F001": 0.3, "F003": 0.4, "F006": 0.3}
}
```

### Risk Analysis Endpoints

#### Portfolio Risk Warnings
```bash
curl -X POST http://localhost:8000/portfolio/risk-warnings \
  -H "Content-Type: application/json" \
  -d '["F001", "F003", "F008"]'
```

Response:
```json
{
  "fund_warnings": {
    "F001": {
      "fund_name": "Axis Bluechip Fund",
      "warnings": [
        {
          "type": "NEGATIVE_RETURNS",
          "severity": "HIGH", 
          "message": "Fund has negative returns (-20.5% CAGR)",
          "suggestion": "Review fund strategy and consider alternatives"
        }
      ],
      "warning_count": 2,
      "risk_level": "HIGH"
    }
  },
  "portfolio_summary": {
    "total_warnings": 3,
    "high_severity_count": 2,
    "portfolio_risk_level": "HIGH",
    "recommendations": [
      "Immediate portfolio review recommended - multiple high-risk funds detected"
    ]
  }
}
```

### NAV Prediction Endpoints

#### Get NAV Prediction (JSON)
```bash
curl http://localhost:8000/predict/F003?periods=15
```

Response:
```json
{
  "fund_id": "F003",
  "fund_name": "SBI Small Cap Fund",
  "current_nav": 110.55,
  "predicted_navs": [111.23, 111.87, 112.45],
  "prediction_dates": ["2024-01-16", "2024-01-17", "2024-01-18"],
  "predicted_metrics": {
    "predicted_cagr": 77.82,
    "predicted_volatility": 0.02,
    "predicted_sharpe_ratio": 2095.118,
    "predicted_total_return": 2.39,
    "prediction_confidence": 0.90
  },
  "model_accuracy": 0.60,
  "prediction_period": 15
}
```

#### Get NAV Prediction (HTML Visualization)
```bash
http://localhost:8000/predict/F003/html
```

Features:
- Interactive Plotly chart showing historical + predicted NAV
- Current vs predicted metrics comparison
- Confidence intervals visualization
- Professional styling and layout
- Saved in html/ folder for offline viewing

#### Portfolio NAV Predictions
```bash
curl -X POST http://localhost:8000/predict/portfolio \
  -H "Content-Type: application/json" \
  -d '["F003", "F006", "F007"]'
```

Response:
```json
{
  "predictions": {
    "F003": {
      "fund_name": "SBI Small Cap Fund",
      "current_nav": 110.55,
      "predicted_final_nav": 113.19,
      "predicted_metrics": {},
      "model_accuracy": 0.60
    }
  },
  "summary": {
    "total_funds": 3,
    "successful_predictions": 3,
    "average_confidence": 0.85,
    "prediction_period": 15
  }
}
```

#### System Operations
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
curl -X POST http://localhost:8000/refresh
```

## File Structure

```
Financial-Data-ETL/
├── etl_pipeline.py              # Core ETL & analytics engine (Requirements 1-4)
├── fund_navs_10funds.csv        # Sample NAV data (10 funds)
├── core_api.py                  # FastAPI server with all endpoints
├── core_demo.py                 # Interactive demo
├── core_requirements.txt        # Dependencies (including Prophet)
├── risk_analyzer.py             # Portfolio risk analysis module
├── nav_predictor.py             # NAV prediction with Prophet
├── html/                        # HTML prediction visualizations
│   └── nav_prediction_*.html    # Generated prediction charts
└── README.md                    # This documentation
```

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | System health check |
| GET | /funds/summary | Complete fund analytics |
| GET | /funds/{fund_id} | Individual fund metrics |
| POST | /portfolio/calculate | Portfolio metrics with custom weights |
| POST | /portfolio/risk-warnings | Risk analysis for portfolio funds |
| GET | /predict/{fund_id} | NAV prediction (JSON) |
| GET | /predict/{fund_id}/html | NAV prediction (HTML visualization) |
| POST | /predict/portfolio | Portfolio NAV predictions |
| GET | /status | System status and features |
| POST | /refresh | Refresh analytics data |

## Sample Data

The pipeline processes fund NAV data with the following structure:
```csv
fund_id,fund_name,date,nav
F001,Axis Bluechip Fund,2022-01-01,38.61
F002,HDFC Top 100 Fund,2022-01-01,515.18
F003,SBI Small Cap Fund,2022-01-01,142.83
```

Dataset includes:
- 10 mutual funds
- 31 days of NAV data (January 2022)
- Various fund types: Large cap, small cap, balanced, index, debt

## Configuration & Parameters

### Default Settings
- Risk-free rate: 4% annual (used in Sharpe ratio calculation)
- Trading days: 252 per year (for annualization)
- Prediction period: 15 days (configurable)
- Prophet model: Facebook's time series forecasting

### Risk Warning Thresholds
- High volatility: >15% (MEDIUM), >25% (HIGH)
- Negative returns: 0% to -10% (MEDIUM), <-10% (HIGH)
- Poor Sharpe ratio: <0.5 with volatility >10%
- Severe underperformance: Sharpe ratio <-1

### NAV Prediction Features
- Prophet Integration: Advanced time series forecasting
- Confidence Intervals: 95% prediction bounds
- HTML Visualizations: Interactive Plotly charts saved in html/ folder
- Synthetic Historical Data: Realistic data generation based on fund metrics
- Fallback Model: Linear trend if Prophet unavailable

## Key Features Summary

### Professional ETL Pipeline
- Advanced Data Cleaning: Multi-method outlier detection and missing value handling
- Financial Metrics: Industry-standard CAGR, volatility, and Sharpe ratio calculations
- Portfolio Analytics: Weighted portfolio metrics with custom allocations
- Data Quality: Comprehensive validation and error handling

### Risk Analysis System
- Fund-Level Risk Detection: 5 types of risk warnings
- Portfolio Risk Assessment: Overall risk scoring and recommendations
- Severity Classification: HIGH/MEDIUM/LOW risk categorization
- Actionable Insights: Specific suggestions for risk mitigation

### NAV Prediction Engine
- Prophet Forecasting: Facebook's time series prediction model
- Interactive Visualizations: Professional HTML charts with Plotly
- Predicted Metrics: CAGR, volatility, Sharpe ratio for forecast period
- Portfolio Predictions: Batch processing for multiple funds
- Confidence Analysis: Model accuracy and prediction reliability

### Production-Ready API
- FastAPI Framework: High-performance, async web framework
- Auto-Generated Docs: Interactive API documentation at /docs
- Type Safety: Pydantic models ensure data validation
- Error Handling: Comprehensive HTTP error responses
- Health Monitoring: System status and health check endpoints

## Testing the System

### 1. Run the Demo
```bash
python core_demo.py
```
Validates: ETL pipeline, analytics, risk analysis, NAV prediction, and HTML generation

### 2. Start the API
```bash
python core_api.py
```
Provides: All REST endpoints for programmatic access

### 3. Interactive Testing
- Visit http://localhost:8000/docs for interactive API testing
- Check http://localhost:8000/status for system overview
- View generated HTML charts in the html/ folder

### 4. Test NAV Predictions
```bash
curl http://localhost:8000/predict/F003
http://localhost:8000/predict/F003/html
curl -X POST http://localhost:8000/predict/portfolio -d '["F003", "F006", "F007"]'
```

## Dependencies

Core ETL and API:
- fastapi: Web framework
- uvicorn: ASGI server
- pandas: Data manipulation
- numpy: Numerical computing
- pydantic: Data validation
- scikit-learn: Basic ML utilities

NAV Prediction:
- prophet: Facebook's time series forecasting
- plotly: Interactive visualizations
- jinja2: HTML templating

## Performance Characteristics

- Data Processing: Handles 310 records across 10 funds in ~2 seconds
- API Response Time: Sub-100ms for individual fund queries
- NAV Prediction: 1-3 seconds per fund (depending on Prophet model)
- HTML Generation: Real-time chart creation with Plotly
- Memory Usage: Minimal footprint with efficient caching

Built with FastAPI, Pandas, Prophet, Plotly, and professional financial analytics principles