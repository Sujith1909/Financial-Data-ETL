#!/usr/bin/env python3
"""
Core Demo - Only Core Requirements (1, 2, 3, 4)
Showcases: ETL Pipeline + Analytics + Portfolio Metrics
"""

from etl_pipeline import FundNAVETL
from risk_analyzer import RiskAnalyzer
from nav_predictor import NAVPredictor

def main():
    """Run core demo showing only the essential requirements."""
    print("🎯 Financial Data ETL Pipeline - Core Demo")
    print("=" * 50)
    print("ETL + Analytics + API + Portfolio Metrics")
    print("=" * 50)
    
    # =====================================================
    # CORE REQUIREMENTS (1, 2, 3, 4)
    # =====================================================
    
    print("\n📊  DATA CLEANING")
    print("-" * 40)
    print("   ✅ Date parsing and normalization")
    print("   ✅ Missing NAV handling (forward-fill → backward-fill → interpolation)")
    print("   ✅ Duplicate NAV removal")
    print("   ✅ Fund-level data consistency")
    print("   ✅ Outlier detection (IQR + Z-score + percentage change)")
    
    print("\n📈  ANALYTICS COMPUTATION")
    print("-" * 40)
    
    # Run ETL pipeline
    etl = FundNAVETL("fund_navs_10funds.csv")
    results = etl.run_full_pipeline()
    
    # Show fund analytics summary table
    print(f"\n{'Fund ID':<8} {'Fund Name':<30} {'CAGR':<8} {'Vol':<8} {'Sharpe':<8}")
    print("-" * 70)
    
    for fund_id, data in results['fund_analytics'].items():
        name = data['fund_name'][:28] + ".." if len(data['fund_name']) > 30 else data['fund_name']
        metrics = data['metrics']
        cagr = f"{metrics.get('cagr', 'N/A')}%" if metrics.get('cagr') is not None else "N/A"
        vol = f"{metrics.get('volatility', 'N/A')}%" if metrics.get('volatility') is not None else "N/A"
        sharpe = f"{metrics.get('sharpe_ratio', 'N/A')}" if metrics.get('sharpe_ratio') is not None else "N/A"
        
        print(f"{fund_id:<8} {name:<30} {cagr:<8} {vol:<8} {sharpe:<8}")
    
    print("\n💼  PORTFOLIO-LEVEL METRICS")
    print("-" * 40)
    portfolio = results['portfolio_metrics']
    print(f"   Weighted CAGR: {portfolio['weighted_cagr']}%")
    print(f"   Portfolio Volatility: {portfolio['portfolio_volatility']}%")
    print(f"   Portfolio Sharpe Ratio: {portfolio['portfolio_sharpe_ratio']}")
    print(f"   Total Funds: {portfolio['num_funds']}")
    print(f"   Valid Funds: {portfolio['valid_funds']}")
    
    print("\n🌐  API INTERFACE")
    print("-" * 40)
    print("   ✅ FastAPI server with auto-generated documentation")
    print("   ✅ GET /funds/summary → Complete analytics as JSON")
    print("   ✅ GET /funds/{fund_id} → Individual fund metrics")
    print("   ✅ POST /portfolio/calculate → Portfolio-level metrics with custom weights")
    
    print("\n📊 DATA SUMMARY")
    print("-" * 40)
    data_summary = results['data_summary']
    print(f"   Total Funds Processed: {data_summary['total_funds']}")
    print(f"   Total Records: {data_summary['total_records']}")
    print(f"   Date Range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}")
    
    # =====================================================
    # NEW RISK ANALYSIS FEATURES
    # =====================================================
    
    print("\n🔍 NEW RISK ANALYSIS FEATURES")
    print("-" * 40)
    
    # Initialize risk analyzer
    risk_analyzer = RiskAnalyzer(results['fund_analytics'])
    
    # Test with sample portfolio
    sample_portfolio = [ 'F007', 'F001', 'F004']
    
    print(f"\nAnalyzing portfolio: {', '.join(sample_portfolio)}")
    
    # Feature 1: Risk Warnings
    print("\n🚨 Feature 1: Portfolio Risk Warnings")
    risk_warnings = risk_analyzer.generate_portfolio_risk_warnings(sample_portfolio)
    
    portfolio_summary = risk_warnings['portfolio_summary']
    print(f"   Total Warnings: {portfolio_summary['total_warnings']}")
    print(f"   Portfolio Risk Level: {portfolio_summary['portfolio_risk_level']}")
    print(f"   High Severity Issues: {portfolio_summary['high_severity_count']}")
    
    # Show warnings for each fund
    for fund_id, fund_info in risk_warnings['fund_warnings'].items():
        if fund_info['warnings']:
            print(f"\n   ⚠️ {fund_info['fund_name']} ({fund_id}):")
            for warning in fund_info['warnings'][:2]:  # Show first 2 warnings
                print(f"      {warning['severity']}: {warning['message']}")
    
    # =====================================================
    # NAV PREDICTION FEATURE
    # =====================================================
    
    print("\n📈 NEW NAV PREDICTION FEATURE")
    print("-" * 40)
    
    # Initialize NAV predictor
    nav_predictor = NAVPredictor(etl)
    
    # Test NAV prediction for a sample fund
    sample_fund = 'F003'  # SBI Small Cap Fund (good performer)
    
    print(f"\nPredicting NAV for {sample_fund} (next 15 days)...")
    
    try:
        prediction = nav_predictor.predict_nav(sample_fund, results['fund_analytics'], 15)
        
        print(f"\n📊 NAV Prediction Results:")
        print(f"   Fund: {prediction.fund_name}")
        print(f"   Current NAV: ₹{prediction.current_nav:.2f}")
        print(f"   Predicted NAV (15 days): ₹{prediction.predicted_navs[-1]:.2f}")
        print(f"   Expected Change: {((prediction.predicted_navs[-1] / prediction.current_nav - 1) * 100):+.2f}%")
        print(f"   Model Accuracy: {prediction.model_accuracy:.1%}")
        
        print(f"\n🔮 Predicted Metrics:")
        metrics = prediction.predicted_metrics
        print(f"   Predicted CAGR: {metrics['predicted_cagr']:.2f}%")
        print(f"   Predicted Volatility: {metrics['predicted_volatility']:.2f}%")
        print(f"   Predicted Sharpe Ratio: {metrics['predicted_sharpe_ratio']:.3f}")
        print(f"   Prediction Confidence: {metrics['prediction_confidence']:.1%}")
        
        # Generate HTML visualization
        html_content = nav_predictor.generate_prediction_html(prediction)
        
        # Create html folder if it doesn't exist
        import os
        os.makedirs("html", exist_ok=True)
        
        html_filename = f"html/nav_prediction_{sample_fund}.html"
        with open(html_filename, "w", encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\n📄 HTML Visualization generated: {html_filename}")
        print(f"   Open this file in your browser to see the interactive chart!")
        
    except Exception as e:
        print(f"   ⚠️ NAV prediction demo failed: {e}")
    
    print("\n✅ CORE DEMO + RISK ANALYSIS + NAV PREDICTION COMPLETED!")
    print("=" * 50)
    print("🚀 Start Core API: python core_api.py")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("\n📋 Core API Endpoints:")
    print("   • GET  /health                    - System health check")
    print("   • GET  /funds/summary             - Complete fund analytics")
    print("   • GET  /funds/{fund_id}           - Individual fund metrics")
    print("   • POST /portfolio/calculate       - Portfolio metrics with weights")
    print("   • POST /portfolio/risk-warnings   - Risk analysis for portfolio")
    print("   • GET  /predict/{fund_id}         - NAV prediction (JSON)")
    print("   • GET  /predict/{fund_id}/html    - NAV prediction (HTML)")
    print("   • POST /predict/portfolio         - Portfolio NAV predictions")
    print("   • GET  /status                    - System status")
    print("   • POST /refresh                   - Refresh data")
    
    print("\n🧪 Test the API:")
    print("   curl http://localhost:8000/funds/summary")
    print("   curl http://localhost:8000/funds/F001")
    print("   curl -X POST http://localhost:8000/portfolio/calculate \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"F001\": 0.4, \"F003\": 0.6}'")
    print("   curl -X POST http://localhost:8000/portfolio/risk-warnings \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '[\"F001\", \"F003\", \"F008\"]'")
    print("   curl http://localhost:8000/predict/F003")
    print("   curl http://localhost:8000/predict/F003/html")
    print("   curl -X POST http://localhost:8000/predict/portfolio \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '[\"F003\", \"F006\", \"F007\"]'")

if __name__ == "__main__":
    main()
