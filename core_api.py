"""
Core API - Only Core Requirements (1, 2, 3, 4)
Clean API with just the essential ETL + Analytics endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
from datetime import datetime
import os

from etl_pipeline import FundNAVETL
from risk_analyzer import RiskAnalyzer
from nav_predictor import NAVPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Fund Analytics API - Core Only",
    description="Core ETL Pipeline + Analytics (Requirements 1-4)",
    version="1.0.0",
    docs_url="/docs"
)

# Global variables
etl_pipeline: Optional[FundNAVETL] = None
analytics_cache: Optional[Dict] = None
last_updated: Optional[datetime] = None
risk_analyzer: Optional[RiskAnalyzer] = None
nav_predictor: Optional[NAVPredictor] = None

# =====================================================
# PYDANTIC MODELS
# =====================================================

class FundMetrics(BaseModel):
    fund_id: str
    fund_name: str
    cagr: Optional[float]
    volatility: Optional[float]
    sharpe_ratio: Optional[float]
    avg_daily_return: Optional[float]
    total_return: Optional[float]
    data_points: int
    date_range: Dict[str, str]

class PortfolioMetrics(BaseModel):
    weighted_cagr: float
    portfolio_volatility: Optional[float]
    portfolio_sharpe_ratio: Optional[float]
    num_funds: int
    valid_funds: int
    weights: Dict[str, float]

class SummaryResponse(BaseModel):
    funds: List[FundMetrics]
    portfolio_metrics: PortfolioMetrics
    data_summary: Dict
    last_updated: str

class HealthResponse(BaseModel):
    status: str
    message: str
    pipeline_loaded: bool
    total_funds: Optional[int]
    last_updated: Optional[str]

class RiskWarningResponse(BaseModel):
    fund_warnings: Dict
    portfolio_summary: Dict

class NAVPredictionResponse(BaseModel):
    fund_id: str
    fund_name: str
    current_nav: float
    predicted_navs: List[float]
    prediction_dates: List[str]
    predicted_metrics: Dict
    confidence_intervals: Dict
    model_accuracy: float
    prediction_period: int

class MultiplePredictionsResponse(BaseModel):
    predictions: Dict
    summary: Dict


# =====================================================
# INITIALIZATION
# =====================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    global etl_pipeline, analytics_cache, last_updated, risk_analyzer, nav_predictor
    
    try:
        csv_path = "fund_navs_10funds.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file {csv_path} not found")
        
        print("🚀 Initializing Core ETL Pipeline...")
        etl_pipeline = FundNAVETL(csv_path)
        analytics_cache = etl_pipeline.run_full_pipeline()
        last_updated = datetime.now()
        print("✅ Core ETL Pipeline ready!")
        
        print("🔍 Initializing Risk Analyzer...")
        risk_analyzer = RiskAnalyzer(analytics_cache['fund_analytics'])
        print("✅ Risk Analyzer ready!")
        
        print("📈 Initializing NAV Predictor...")
        nav_predictor = NAVPredictor(etl_pipeline)
        print("✅ NAV Predictor ready!")
        
        print("🎉 Core system with risk analysis and NAV prediction fully initialized!")
        
    except Exception as e:
        print(f"Failed to initialize: {str(e)}")

# =====================================================
# CORE REQUIREMENTS ENDPOINTS
# =====================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline_loaded = analytics_cache is not None
    total_funds = None
    last_updated_str = None
    
    if analytics_cache:
        total_funds = analytics_cache.get('data_summary', {}).get('total_funds', 0)
        last_updated_str = last_updated.isoformat() if last_updated else None
    
    return HealthResponse(
        status="healthy" if pipeline_loaded else "initializing",
        message="Core ETL pipeline is ready" if pipeline_loaded else "ETL pipeline is loading",
        pipeline_loaded=pipeline_loaded,
        total_funds=total_funds,
        last_updated=last_updated_str
    )

@app.get("/funds/summary", response_model=SummaryResponse)
async def get_funds_summary():
    """
    Core Requirement 3: GET /funds/summary → returns summary analytics as JSON
    
    Returns complete analytics summary for all funds including:
    - Individual fund metrics (CAGR, Volatility, Sharpe Ratio)
    - Portfolio-level weighted metrics
    """
    if analytics_cache is None:
        raise HTTPException(
            status_code=503, 
            detail="ETL pipeline not ready. Please try again in a moment."
        )
    
    try:
        # Convert analytics data to response format
        funds = []
        for fund_id, data in analytics_cache['fund_analytics'].items():
            metrics = data['metrics']
            
            fund_metrics = FundMetrics(
                fund_id=fund_id,
                fund_name=data['fund_name'],
                cagr=metrics.get('cagr'),
                volatility=metrics.get('volatility'),
                sharpe_ratio=metrics.get('sharpe_ratio'),
                avg_daily_return=metrics.get('avg_daily_return'),
                total_return=metrics.get('total_return'),
                data_points=data['data_points'],
                date_range=data['date_range']
            )
            funds.append(fund_metrics)
        
        # Sort funds by Sharpe ratio (descending)
        funds.sort(key=lambda x: x.sharpe_ratio if x.sharpe_ratio is not None else -999, reverse=True)
        
        portfolio_metrics = PortfolioMetrics(**analytics_cache['portfolio_metrics'])
        
        return SummaryResponse(
            funds=funds,
            portfolio_metrics=portfolio_metrics,
            data_summary=analytics_cache['data_summary'],
            last_updated=last_updated.isoformat() if last_updated else ""
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.get("/funds/{fund_id}", response_model=FundMetrics)
async def get_fund_metrics(fund_id: str):
    """
    Core Requirement 3: GET /funds/{fund_id} → returns metrics for a specific fund
    
    Returns detailed metrics for a specific fund including:
    - CAGR (Compounded Annual Growth Rate)
    - Volatility (standard deviation of daily returns)
    - Sharpe Ratio (risk-free rate = 4%)
    """
    if analytics_cache is None:
        raise HTTPException(
            status_code=503, 
            detail="ETL pipeline not ready. Please try again in a moment."
        )
    
    if fund_id not in analytics_cache['fund_analytics']:
        raise HTTPException(
            status_code=404, 
            detail=f"Fund {fund_id} not found"
        )
    
    try:
        data = analytics_cache['fund_analytics'][fund_id]
        metrics = data['metrics']
        
        return FundMetrics(
            fund_id=fund_id,
            fund_name=data['fund_name'],
            cagr=metrics.get('cagr'),
            volatility=metrics.get('volatility'),
            sharpe_ratio=metrics.get('sharpe_ratio'),
            avg_daily_return=metrics.get('avg_daily_return'),
            total_return=metrics.get('total_return'),
            data_points=data['data_points'],
            date_range=data['date_range']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving fund data: {str(e)}")

@app.post("/portfolio/calculate")
async def calculate_portfolio_metrics(weights: Optional[Dict[str, float]] = None):
    """
    Bonus Requirement 4: Portfolio-level metrics (weighted CAGR)
    
    Calculate portfolio metrics with custom weights. Does NOT optimize - just calculates 
    metrics for the weights you provide.
    """
    if analytics_cache is None or etl_pipeline is None:
        raise HTTPException(
            status_code=503, 
            detail="ETL pipeline not ready. Please try again in a moment."
        )
    
    try:
        portfolio_metrics = etl_pipeline.get_portfolio_metrics(weights)
        return portfolio_metrics
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating portfolio metrics: {str(e)}")

# =====================================================
# RISK ANALYSIS ENDPOINTS
# =====================================================

@app.post("/portfolio/risk-warnings", response_model=RiskWarningResponse)
async def get_portfolio_risk_warnings(fund_ids: List[str]):
    """
    Feature 1: Portfolio Risk Warnings
    
    Analyzes a list of fund IDs in your portfolio and identifies specific risks:
    - Volatility risks (high volatility funds)
    - Performance risks (negative returns, poor Sharpe ratios)
    - Efficiency risks (poor risk-adjusted returns)
    - Portfolio-level risk assessment
    """
    if risk_analyzer is None:
        raise HTTPException(status_code=503, detail="Risk analyzer not ready")
    
    if not fund_ids:
        raise HTTPException(status_code=400, detail="Please provide fund_ids list")
    
    try:
        risk_warnings = risk_analyzer.generate_portfolio_risk_warnings(fund_ids)
        return RiskWarningResponse(
            fund_warnings=risk_warnings['fund_warnings'],
            portfolio_summary=risk_warnings['portfolio_summary']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis error: {str(e)}")

# =====================================================
# NAV PREDICTION ENDPOINTS
# =====================================================

@app.get("/predict/{fund_id}", response_model=NAVPredictionResponse)
async def predict_fund_nav(fund_id: str, periods: int = 15):
    """
    Predict NAV for a specific fund for the next N days using Prophet.
    
    Returns:
    - Predicted NAV values for next 15 days (default)
    - Predicted metrics (CAGR, Volatility, Sharpe Ratio)
    - Model accuracy and confidence intervals
    """
    if nav_predictor is None:
        raise HTTPException(status_code=503, detail="NAV predictor not ready")
    
    if fund_id not in analytics_cache['fund_analytics']:
        raise HTTPException(status_code=404, detail=f"Fund {fund_id} not found")
    
    try:
        prediction = nav_predictor.predict_nav(
            fund_id, 
            analytics_cache['fund_analytics'], 
            periods
        )
        
        return NAVPredictionResponse(
            fund_id=prediction.fund_id,
            fund_name=prediction.fund_name,
            current_nav=prediction.current_nav,
            predicted_navs=prediction.predicted_navs,
            prediction_dates=prediction.prediction_dates,
            predicted_metrics=prediction.predicted_metrics,
            confidence_intervals=prediction.confidence_intervals,
            model_accuracy=prediction.model_accuracy,
            prediction_period=prediction.prediction_period
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NAV prediction error: {str(e)}")

@app.get("/predict/{fund_id}/html")
async def get_fund_prediction_html(fund_id: str, periods: int = 15):
    """
    Get HTML visualization showing current vs predicted NAV for a fund.
    
    Returns interactive HTML page with:
    - Chart showing historical and predicted NAV
    - Current vs predicted metrics comparison
    - Confidence intervals and model accuracy
    """
    if nav_predictor is None:
        raise HTTPException(status_code=503, detail="NAV predictor not ready")
    
    if fund_id not in analytics_cache['fund_analytics']:
        raise HTTPException(status_code=404, detail=f"Fund {fund_id} not found")
    
    try:
        prediction = nav_predictor.predict_nav(
            fund_id, 
            analytics_cache['fund_analytics'], 
            periods
        )
        
        html_content = nav_predictor.generate_prediction_html(prediction)
        
        # Return HTML content
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HTML generation error: {str(e)}")

@app.post("/predict/portfolio", response_model=MultiplePredictionsResponse)
async def predict_portfolio_navs(fund_ids: List[str], periods: int = 15):
    """
    Predict NAV for multiple funds in a portfolio.
    
    Returns predictions and summary for all funds in the portfolio.
    """
    if nav_predictor is None:
        raise HTTPException(status_code=503, detail="NAV predictor not ready")
    
    if not fund_ids:
        raise HTTPException(status_code=400, detail="Please provide fund_ids list")
    
    try:
        predictions = nav_predictor.predict_multiple_funds(
            fund_ids, 
            analytics_cache['fund_analytics'], 
            periods
        )
        
        return MultiplePredictionsResponse(
            predictions=predictions['predictions'],
            summary=predictions['summary']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio prediction error: {str(e)}")

# =====================================================
# SYSTEM STATUS
# =====================================================

@app.get("/status")
async def get_system_status():
    """Get status of the core system."""
    return {
        "system": "Fund Analytics API - Core Only",
        "features": [
            "✅ Data Cleaning & ETL Pipeline (Requirement 1)",
            "✅ Fund Analytics - CAGR, Volatility, Sharpe Ratio (Requirement 2)", 
            "✅ REST API - /funds/summary, /funds/{id} (Requirement 3)",
            "✅ Portfolio-level Metrics - Weighted CAGR (Requirement 4)",
            "🔍 Portfolio Risk Warnings - Identify fund risks",
            "📈 NAV Prediction - Prophet-based forecasting with HTML visualization"
        ],
        "endpoints": {
            "health": "GET /health",
            "fund_summary": "GET /funds/summary", 
            "fund_details": "GET /funds/{fund_id}",
            "portfolio_calculate": "POST /portfolio/calculate",
            "risk_warnings": "POST /portfolio/risk-warnings",
            "nav_prediction": "GET /predict/{fund_id}",
            "nav_prediction_html": "GET /predict/{fund_id}/html",
            "portfolio_prediction": "POST /predict/portfolio"
        },
        "pipeline_status": "ready" if analytics_cache else "not_ready",
        "last_updated": last_updated.isoformat() if last_updated else None
    }

@app.post("/refresh")
async def refresh_data():
    """Refresh the analytics data by re-running the ETL pipeline."""
    global analytics_cache, last_updated
    
    try:
        if etl_pipeline is None:
            raise HTTPException(status_code=503, detail="ETL pipeline not initialized")
        
        print("🔄 Refreshing ETL pipeline...")
        analytics_cache = etl_pipeline.run_full_pipeline()
        last_updated = datetime.now()
        print("✅ Data refreshed successfully!")
        
        return {
            "message": "Data refreshed successfully",
            "last_updated": last_updated.isoformat(),
            "total_funds": analytics_cache.get('data_summary', {}).get('total_funds', 0)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing data: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "core_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
