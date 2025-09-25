# Financial Data ETL Pipeline

A production-ready ETL system for mutual fund NAV analysis with risk assessment and forecasting capabilities.

## Design Architecture


### Core Components
- **ETL Pipeline** (`etl_pipeline.py`): Modular data processing with statistical outlier detection
- **REST API** (`core_api.py`): FastAPI-based service with auto-generated documentation
- **Risk Analyzer** (`risk_analyzer.py`): Portfolio risk assessment with severity classification
- **NAV Predictor** (`nav_predictor.py`): Prophet-based time series forecasting with HTML visualization

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
- **Choice**: Prophet for time series forecasting with fallback to linear trend
- **Trade-off**: Heavy dependency (Prophet) but industry-standard forecasting
- **Rationale**: Financial forecasting requires sophisticated models; fallback ensures reliability

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

### Time Series Forecasting
- **Assumption**: Historical patterns indicate future trends (Prophet's core assumption)
- **Limitation**: 15-day prediction window (balance between accuracy and usefulness)
- **Confidence**: 95% intervals provided to quantify uncertainty

### API Usage Patterns
- **Assumption**: Users need both individual fund analysis and portfolio-level metrics
- **Design**: Separate endpoints for granular and aggregated views
- **Caching**: Analytics computed once per data refresh cycle

## Technical Decisions

### Framework Choices
- **FastAPI**: Auto-documentation, type hints, async support
- **Prophet**: Facebook's proven time series library
- **Plotly**: Interactive visualizations for stakeholder presentations
- **Pandas**: Industry standard for financial data manipulation

### Error Handling Strategy
- **API Level**: HTTP status codes with descriptive error messages
- **Processing Level**: Graceful fallbacks (forward-fill → backward-fill → interpolation)
- **Validation Level**: Pydantic models prevent invalid data propagation

### Extensibility Points
- **New Metrics**: Easy addition through `FundNAVETL.calculate_analytics()`
- **Risk Rules**: Configurable thresholds in `RiskAnalyzer`
- **Prediction Models**: Pluggable forecasting algorithms in `NAVPredictor`
- **Data Sources**: Abstract CSV loading for future database integration

## Quick Start

```bash
pip install -r core_requirements.txt
python core_demo.py  # Demo all features
python core_api.py   # Start API server
```

API Documentation: http://localhost:8000/docs