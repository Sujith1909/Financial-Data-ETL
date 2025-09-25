# Financial Data ETL Pipeline

A production-ready ETL system for mutual fund NAV analysis with risk assessment and forecasting capabilities.

## Design Architecture


### Core Components
- **ETL Pipeline** (`etl_pipeline.py`): Modular data processing with statistical outlier detection
- **REST API** (`core_api.py`): FastAPI-based service with auto-generated documentation
- **Risk Analyzer** (`risk_analyzer.py`): Portfolio risk assessment with severity classification
- **NAV Predictor** (`nav_predictor.py`): AI-driven time series forecasting with interactive HTML visualization

### Design Patterns
- **Separation of Concerns**: Each module handles a specific domain (ETL, API, Risk, Prediction)
- **Dependency Injection**: Components are loosely coupled through initialization parameters
- **Type Safety**: Pydantic models ensure data validation across all API endpoints
- **Graceful Degradation**: Optional Prophet dependency with fallback linear trend model

## Key Trade-offs

### Performance vs Accuracy
- **Choice**: Multi-method outlier detection (IQR + Z-score + percentage change)
- **Trade-off**: Slower processing (~2 seconds) for more robust data cleaning
- **Rationale**: Financial data quality is critical; slight performance penalty is acceptable

### Complexity vs Maintainability
- **Choice**: Modular architecture with 4 separate classes
- **Trade-off**: More files but clearer separation of responsibilities
- **Rationale**: Easier testing, debugging, and feature extension

### Memory vs Speed
- **Choice**: In-memory analytics caching with refresh endpoint
- **Trade-off**: Higher memory usage but sub-100ms API response times
- **Rationale**: Real-time performance requirements for financial APIs

### Prediction Accuracy vs Simplicity
- **Choice**: AI-powered time series forecasting with machine learning fallback
- **Trade-off**: Complex ML models vs simple linear trends; accuracy vs interpretability
- **Rationale**: Financial markets exhibit non-linear patterns requiring advanced AI; linear fallback ensures system reliability
- **Data Limitation**: 30-day historical window limits prediction accuracy - longer datasets would improve results

## Core Assumptions

### Data Quality
- **Assumption**: CSV data contains valid fund_id, fund_name, date, nav columns
- **Handling**: Comprehensive validation with error messages for missing/invalid data
- **Impact**: System gracefully handles data quality issues without crashes

### Financial Calculations
- **Risk-free Rate**: 4% annual (industry standard for Sharpe ratio)
- **Trading Days**: 252 per year (standard market assumption)
- **Outlier Threshold**: 3 standard deviations (conservative statistical approach)
- **Risk Categories**: >15% volatility = high risk (industry benchmarks)

### AI-Powered Time Series Forecasting
- **AI Approach**: Machine learning algorithms analyze historical NAV patterns to detect trends, seasonality, and volatility cycles
- **Pattern Recognition**: Neural network-based models identify complex non-linear relationships in financial time series data
- **Data Limitation**: Only 30 days of historical data available - results may not capture long-term market cycles or seasonal patterns
- **Prediction Window**: 15-day forecasts (optimal balance between AI model confidence and practical investment decisions)
- **Uncertainty Quantification**: 95% confidence intervals provided, but predictions should be treated as estimates due to limited training data
- **Market Volatility**: AI models may not predict sudden market disruptions or external economic events not present in training data

### API Usage Patterns
- **Assumption**: Users need both individual fund analysis and portfolio-level metrics
- **Design**: Separate endpoints for granular and aggregated views
- **Caching**: Analytics computed once per data refresh cycle

## Technical Decisions

### Framework Choices
- **FastAPI**: Auto-documentation, type hints, async support
- **AI Forecasting**: Advanced time series ML algorithms with automatic trend detection and seasonality analysis
- **Plotly**: Interactive visualizations for stakeholder presentations
- **Pandas**: Industry standard for financial data manipulation

### Error Handling Strategy
- **API Level**: HTTP status codes with descriptive error messages
- **Processing Level**: Graceful fallbacks (forward-fill → backward-fill → interpolation)
- **Validation Level**: Pydantic models prevent invalid data propagation

### Extensibility Points
- **New Metrics**: Easy addition through `FundNAVETL.calculate_analytics()`
- **Risk Rules**: Configurable thresholds in `RiskAnalyzer`
- **AI Prediction Models**: Pluggable ML forecasting algorithms in `NAVPredictor` - easily swap between different AI approaches
- **Data Sources**: Abstract CSV loading for future database integration

## AI Forecasting Limitations & Recommendations

### Current Data Constraints
- **Limited Historical Data**: Only 30 days of NAV data available for training AI models
- **Short-term Patterns**: AI can identify daily and weekly trends but lacks data for monthly/quarterly cycles
- **Market Context Missing**: No external factors (economic indicators, market events) included in training data

### Prediction Quality Expectations
- **Best Performance**: Funds with consistent, trending patterns over the 30-day period
- **Lower Accuracy**: Highly volatile funds or those with irregular patterns
- **Confidence Levels**: Predictions include uncertainty bounds - use as guidance, not absolute forecasts

### Recommendations for Production Use
- **Longer Historical Data**: 1-3 years of data would significantly improve AI model accuracy
- **External Data Integration**: Include market indices, economic indicators, sector performance
- **Model Ensemble**: Combine multiple AI algorithms for more robust predictions
- **Regular Retraining**: Update models frequently as new data becomes available
- **Human Oversight**: Treat AI predictions as decision support tools, not replacement for financial expertise

## Quick Start

```bash
pip install -r core_requirements.txt
python core_demo.py  # Demo all features
python core_api.py   # Start API server
```

API Documentation: http://localhost:8000/docs

example html generation curl 
curl "http://localhost:8000/predict/F003/html?periods=15" -o html/F003_prediction.html