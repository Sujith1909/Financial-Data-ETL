"""
NAV Prediction Module using Prophet
Predicts future NAV values and calculates predicted metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings

# Prophet for time series forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Plotly for HTML visualization
try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

warnings.filterwarnings('ignore')

@dataclass
class NAVPrediction:
    """Structure for NAV prediction results."""
    fund_id: str
    fund_name: str
    current_nav: float
    predicted_navs: List[float]
    prediction_dates: List[str]
    predicted_metrics: Dict
    confidence_intervals: Dict
    model_accuracy: float
    prediction_period: int

@dataclass
class PredictedMetrics:
    """Predicted financial metrics for the forecast period."""
    predicted_cagr: float
    predicted_volatility: float
    predicted_sharpe_ratio: float
    predicted_total_return: float
    prediction_confidence: float

class NAVPredictor:
    """NAV prediction engine using Prophet for time series forecasting."""
    
    def __init__(self, etl_pipeline=None):
        self.etl_pipeline = etl_pipeline
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
        
    def predict_nav(self, fund_id: str, fund_data: Dict, periods: int = 15) -> NAVPrediction:
        """
        Predict NAV for the next N periods using Prophet.
        
        Args:
            fund_id: Fund identifier
            fund_data: Fund analytics data from ETL pipeline
            periods: Number of days to predict (default 15)
        
        Returns:
            NAVPrediction with forecasted values and metrics
        """
        
        if fund_id not in fund_data:
            raise ValueError(f"Fund {fund_id} not found in data")
        
        fund_info = fund_data[fund_id]
        
        # Generate synthetic historical NAV data for Prophet
        # In real implementation, this would use actual daily NAV time series
        historical_navs = self._generate_synthetic_nav_series(fund_info, periods=60)
        
        if PROPHET_AVAILABLE:
            predicted_navs, confidence_intervals, accuracy = self._prophet_forecast(
                historical_navs, periods
            )
        else:
            predicted_navs, confidence_intervals, accuracy = self._simple_forecast(
                historical_navs, periods
            )
        
        # Calculate predicted metrics
        predicted_metrics = self._calculate_predicted_metrics(
            historical_navs['y'].iloc[-1],  # Current NAV
            predicted_navs,
            periods
        )
        
        # Generate prediction dates
        last_date = datetime.now()
        prediction_dates = [
            (last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') 
            for i in range(periods)
        ]
        
        return NAVPrediction(
            fund_id=fund_id,
            fund_name=fund_info['fund_name'],
            current_nav=float(historical_navs['y'].iloc[-1]),
            predicted_navs=[float(nav) for nav in predicted_navs],
            prediction_dates=prediction_dates,
            predicted_metrics={
                'predicted_cagr': predicted_metrics.predicted_cagr,
                'predicted_volatility': predicted_metrics.predicted_volatility,
                'predicted_sharpe_ratio': predicted_metrics.predicted_sharpe_ratio,
                'predicted_total_return': predicted_metrics.predicted_total_return,
                'prediction_confidence': predicted_metrics.prediction_confidence
            },
            confidence_intervals=confidence_intervals,
            model_accuracy=accuracy,
            prediction_period=periods
        )
    
    def _generate_synthetic_nav_series(self, fund_info: Dict, periods: int = 60) -> pd.DataFrame:
        """Generate synthetic historical NAV data based on fund metrics."""
        metrics = fund_info['metrics']
        
        # Extract fund characteristics
        annual_return = metrics.get('cagr', 0) / 100  # Convert to decimal
        annual_volatility = metrics.get('volatility', 10) / 100
        
        # Generate synthetic daily data
        np.random.seed(hash(fund_info['fund_name']) % 2147483647)  # Consistent seed per fund
        
        # Daily parameters
        daily_return = annual_return / 252
        daily_volatility = annual_volatility / np.sqrt(252)
        
        # Generate NAV series
        initial_nav = 100  # Starting NAV
        nav_series = [initial_nav]
        
        start_date = datetime.now() - timedelta(days=periods)
        dates = [start_date + timedelta(days=i) for i in range(periods)]
        
        for i in range(periods - 1):
            # Add some realistic market patterns
            trend_factor = 1 + (i / periods) * 0.1  # Slight upward trend
            daily_shock = np.random.normal(daily_return * trend_factor, daily_volatility)
            new_nav = nav_series[-1] * (1 + daily_shock)
            nav_series.append(max(new_nav, nav_series[-1] * 0.95))  # Prevent unrealistic drops
        
        # Create Prophet-compatible dataframe
        df = pd.DataFrame({
            'ds': dates,
            'y': nav_series
        })
        
        return df
    
    def _prophet_forecast(self, historical_data: pd.DataFrame, periods: int) -> Tuple[List[float], Dict, float]:
        """Use Prophet to forecast NAV values."""
        try:
            # Initialize Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05  # Conservative changepoint detection
            )
            
            # Fit the model
            model.fit(historical_data)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=periods)
            
            # Generate forecast
            forecast = model.predict(future)
            
            # Extract predictions and confidence intervals
            predicted_navs = forecast['yhat'].tail(periods).tolist()
            
            confidence_intervals = {
                'lower': forecast['yhat_lower'].tail(periods).tolist(),
                'upper': forecast['yhat_upper'].tail(periods).tolist()
            }
            
            # Calculate simple accuracy (R-squared like measure)
            historical_predictions = forecast['yhat'].head(len(historical_data))
            actual_values = historical_data['y']
            
            residuals = actual_values - historical_predictions
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
            accuracy = max(0, 1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.5
            
            return predicted_navs, confidence_intervals, accuracy
            
        except Exception as e:
            print(f"Prophet forecasting failed: {e}")
            return self._simple_forecast(historical_data, periods)
    
    def _simple_forecast(self, historical_data: pd.DataFrame, periods: int) -> Tuple[List[float], Dict, float]:
        """Simple linear trend forecast as fallback."""
        navs = historical_data['y'].values
        
        # Calculate simple trend
        x = np.arange(len(navs))
        coeffs = np.polyfit(x, navs, 1)  # Linear trend
        trend_slope = coeffs[0]
        
        # Generate predictions
        last_nav = navs[-1]
        predicted_navs = []
        
        for i in range(1, periods + 1):
            # Add trend with some noise
            predicted_nav = last_nav + (trend_slope * i)
            predicted_navs.append(predicted_nav)
        
        # Simple confidence intervals (±10%)
        confidence_intervals = {
            'lower': [nav * 0.9 for nav in predicted_navs],
            'upper': [nav * 1.1 for nav in predicted_navs]
        }
        
        # Simple accuracy estimate
        accuracy = 0.6  # Conservative estimate for simple model
        
        return predicted_navs, confidence_intervals, accuracy
    
    def _calculate_predicted_metrics(self, current_nav: float, predicted_navs: List[float], 
                                   periods: int) -> PredictedMetrics:
        """Calculate financial metrics for the predicted period."""
        
        if not predicted_navs or periods == 0:
            return PredictedMetrics(0, 0, 0, 0, 0)
        
        # Calculate predicted total return
        final_nav = predicted_navs[-1]
        predicted_total_return = ((final_nav / current_nav) - 1) * 100
        
        # Calculate predicted CAGR (annualized)
        days_in_period = periods
        years = days_in_period / 365.25
        predicted_cagr = (((final_nav / current_nav) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # Calculate predicted volatility from daily returns
        all_navs = [current_nav] + predicted_navs
        daily_returns = []
        for i in range(1, len(all_navs)):
            daily_return = (all_navs[i] / all_navs[i-1]) - 1
            daily_returns.append(daily_return)
        
        if daily_returns:
            predicted_volatility = np.std(daily_returns) * np.sqrt(252) * 100  # Annualized
        else:
            predicted_volatility = 0
        
        # Calculate predicted Sharpe ratio
        if predicted_volatility > 0:
            avg_daily_return = np.mean(daily_returns) if daily_returns else 0
            excess_return = (avg_daily_return * 252) - self.risk_free_rate
            predicted_sharpe_ratio = excess_return / (predicted_volatility / 100)
        else:
            predicted_sharpe_ratio = 0
        
        # Prediction confidence (based on trend consistency)
        nav_changes = np.diff(predicted_navs)
        trend_consistency = 1 - (np.std(nav_changes) / (np.mean(np.abs(nav_changes)) + 1e-6))
        prediction_confidence = max(0.3, min(0.9, trend_consistency))
        
        return PredictedMetrics(
            predicted_cagr=round(predicted_cagr, 2),
            predicted_volatility=round(predicted_volatility, 2),
            predicted_sharpe_ratio=round(predicted_sharpe_ratio, 4),
            predicted_total_return=round(predicted_total_return, 2),
            prediction_confidence=round(prediction_confidence, 2)
        )
    
    def generate_prediction_html(self, prediction: NAVPrediction) -> str:
        """Generate HTML visualization showing current vs predicted NAV."""
        
        if not PLOTLY_AVAILABLE:
            return self._generate_simple_html(prediction)
        
        # Prepare data for plotting
        historical_dates = [(datetime.now() - timedelta(days=i)) for i in range(30, 0, -1)]
        historical_navs = self._generate_historical_navs_for_plot(prediction, 30)
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=[d.strftime('%Y-%m-%d') for d in historical_dates],
            y=historical_navs,
            mode='lines+markers',
            name='Historical NAV',
            line=dict(color='blue', width=2),
            marker=dict(size=4)
        ))
        
        # Predicted data
        fig.add_trace(go.Scatter(
            x=prediction.prediction_dates,
            y=prediction.predicted_navs,
            mode='lines+markers',
            name='Predicted NAV',
            line=dict(color='red', dash='dash', width=2),
            marker=dict(size=4)
        ))
        
        # Confidence intervals (if available)
        if prediction.confidence_intervals.get('upper') and prediction.confidence_intervals.get('lower'):
            fig.add_trace(go.Scatter(
                x=prediction.prediction_dates + prediction.prediction_dates[::-1],
                y=prediction.confidence_intervals['upper'] + prediction.confidence_intervals['lower'][::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=True,
                name='Confidence Interval'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{prediction.fund_name} - NAV Prediction (Next {prediction.prediction_period} Days)",
            xaxis_title="Date",
            yaxis_title="NAV Value",
            template="plotly_white",
            height=500,
            showlegend=True
        )
        
        # Generate HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NAV Prediction - {prediction.fund_name}</title>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .current {{ background-color: #e8f5e8; }}
                .predicted {{ background-color: #fff4e6; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>NAV Prediction Analysis</h1>
                <h2>{prediction.fund_name} ({prediction.fund_id})</h2>
                <p><strong>Prediction Period:</strong> Next {prediction.prediction_period} days</p>
                <p><strong>Model Accuracy:</strong> {prediction.model_accuracy:.1%}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card current">
                    <h3>Current Metrics</h3>
                    <p><strong>Current NAV:</strong> Rs.{prediction.current_nav:.2f}</p>
                </div>
                <div class="metric-card predicted">
                    <h3>Predicted Metrics</h3>
                    <p><strong>Predicted CAGR:</strong> {prediction.predicted_metrics['predicted_cagr']:.2f}%</p>
                    <p><strong>Predicted Volatility:</strong> {prediction.predicted_metrics['predicted_volatility']:.2f}%</p>
                    <p><strong>Predicted Sharpe Ratio:</strong> {prediction.predicted_metrics['predicted_sharpe_ratio']:.3f}</p>
                    <p><strong>Total Return (15 days):</strong> {prediction.predicted_metrics['predicted_total_return']:.2f}%</p>
                    <p><strong>Confidence:</strong> {prediction.predicted_metrics['prediction_confidence']:.1%}</p>
                </div>
            </div>
            
            <div id="chart">
                {pyo.plot(fig, include_plotlyjs=True, output_type='div')}
            </div>
            
            <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
                <h3>Prediction Summary</h3>
                <p><strong>Final Predicted NAV:</strong> Rs.{prediction.predicted_navs[-1]:.2f}</p>
                <p><strong>Expected Change:</strong> {((prediction.predicted_navs[-1] / prediction.current_nav - 1) * 100):+.2f}%</p>
                <p><strong>Model Used:</strong> {"Prophet (Facebook)" if PROPHET_AVAILABLE else "Linear Trend"}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_historical_navs_for_plot(self, prediction: NAVPrediction, days: int) -> List[float]:
        """Generate historical NAV values for plotting."""
        # Generate synthetic historical data leading to current NAV
        current_nav = prediction.current_nav
        
        # Work backwards to create realistic historical data
        historical_navs = []
        nav = current_nav
        
        # Use fund's volatility to generate realistic historical data
        np.random.seed(hash(prediction.fund_name) % 2147483647)
        
        for i in range(days):
            # Small daily variations based on fund characteristics
            daily_change = np.random.normal(0, 0.01)  # 1% daily std
            nav = nav * (1 + daily_change)
            historical_navs.append(nav)
        
        # Reverse to get chronological order
        historical_navs.reverse()
        
        return historical_navs
    
    def _generate_simple_html(self, prediction: NAVPrediction) -> str:
        """Generate simple HTML without Plotly if not available."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>NAV Prediction - {prediction.fund_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric-card {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
                .prediction-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                .prediction-table th, .prediction-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .prediction-table th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📈 NAV Prediction Analysis</h1>
                <h2>{prediction.fund_name} ({prediction.fund_id})</h2>
                <p><strong>Prediction Period:</strong> Next {prediction.prediction_period} days</p>
                <p><strong>Model Accuracy:</strong> {prediction.model_accuracy:.1%}</p>
            </div>
            
            <div class="metrics">
                <div class="metric-card">
                    <h3>Current NAV</h3>
                    <h2>Rs.{prediction.current_nav:.2f}</h2>
                </div>
                <div class="metric-card">
                    <h3>Predicted Final NAV</h3>
                    <h2>Rs.{prediction.predicted_navs[-1]:.2f}</h2>
                    <p>({((prediction.predicted_navs[-1] / prediction.current_nav - 1) * 100):+.2f}%)</p>
                </div>
            </div>
            
            <table class="prediction-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Predicted NAV</th>
                        <th>Daily Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add prediction rows
        for i, (date, nav) in enumerate(zip(prediction.prediction_dates, prediction.predicted_navs)):
            if i == 0:
                daily_change = ((nav / prediction.current_nav) - 1) * 100
            else:
                daily_change = ((nav / prediction.predicted_navs[i-1]) - 1) * 100
            
            color = "color: green;" if daily_change >= 0 else "color: red;"
            html += f"""
                    <tr>
                        <td>{date}</td>
                        <td>Rs.{nav:.2f}</td>
                        <td style="{color}">{daily_change:+.2f}%</td>
                    </tr>"""
        
        html += f"""
                </tbody>
            </table>
            
            <div style="margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
                <h3>Predicted Metrics Summary</h3>
                <p><strong>Predicted CAGR:</strong> {prediction.predicted_metrics['predicted_cagr']:.2f}%</p>
                <p><strong>Predicted Volatility:</strong> {prediction.predicted_metrics['predicted_volatility']:.2f}%</p>
                <p><strong>Predicted Sharpe Ratio:</strong> {prediction.predicted_metrics['predicted_sharpe_ratio']:.3f}</p>
                <p><strong>Total Return (15 days):</strong> {prediction.predicted_metrics['predicted_total_return']:.2f}%</p>
                <p><strong>Prediction Confidence:</strong> {prediction.predicted_metrics['prediction_confidence']:.1%}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def predict_multiple_funds(self, fund_ids: List[str], fund_data: Dict, periods: int = 15) -> Dict:
        """Predict NAV for multiple funds and return summary."""
        predictions = {}
        summary = {
            'total_funds': len(fund_ids),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_confidence': 0,
            'prediction_period': periods
        }
        
        confidences = []
        
        for fund_id in fund_ids:
            try:
                prediction = self.predict_nav(fund_id, fund_data, periods)
                predictions[fund_id] = {
                    'fund_name': prediction.fund_name,
                    'current_nav': prediction.current_nav,
                    'predicted_final_nav': prediction.predicted_navs[-1],
                    'predicted_metrics': prediction.predicted_metrics,
                    'model_accuracy': prediction.model_accuracy,
                    'prediction_success': True
                }
                confidences.append(prediction.predicted_metrics['prediction_confidence'])
                summary['successful_predictions'] += 1
                
            except Exception as e:
                predictions[fund_id] = {
                    'error': str(e),
                    'prediction_success': False
                }
                summary['failed_predictions'] += 1
        
        if confidences:
            summary['average_confidence'] = sum(confidences) / len(confidences)
        
        return {
            'predictions': predictions,
            'summary': summary
        }

# Demo function
def demo_nav_predictor():
    """Demo the NAV predictor."""
    # Sample fund data
    sample_fund_data = {
        'F001': {
            'fund_name': 'Axis Bluechip Fund',
            'metrics': {'cagr': -20.48, 'volatility': 7.2, 'sharpe_ratio': -2.716}
        },
        'F003': {
            'fund_name': 'SBI Small Cap Fund', 
            'metrics': {'cagr': 39.39, 'volatility': 7.78, 'sharpe_ratio': 2.4713}
        }
    }
    
    predictor = NAVPredictor()
    
    # Test single fund prediction
    prediction = predictor.predict_nav('F003', sample_fund_data, 15)
    
    print(f"📈 NAV Prediction for {prediction.fund_name}:")
    print(f"   Current NAV: ₹{prediction.current_nav:.2f}")
    print(f"   Predicted NAV (15 days): ₹{prediction.predicted_navs[-1]:.2f}")
    print(f"   Expected Change: {((prediction.predicted_navs[-1] / prediction.current_nav - 1) * 100):+.2f}%")
    print(f"   Predicted CAGR: {prediction.predicted_metrics['predicted_cagr']:.2f}%")
    print(f"   Model Accuracy: {prediction.model_accuracy:.1%}")
    
    # Generate HTML (save to file for viewing)
    html_content = predictor.generate_prediction_html(prediction)
    
    # Create html folder if it doesn't exist
    import os
    os.makedirs("html", exist_ok=True)
    
    html_filename = f"html/nav_prediction_{prediction.fund_id}.html"
    with open(html_filename, "w", encoding='utf-8') as f:
        f.write(html_content)
    print(f"   HTML visualization saved: {html_filename}")

if __name__ == "__main__":
    demo_nav_predictor()
