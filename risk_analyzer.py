"""
Risk Analysis Module for Portfolio Fund Monitoring
Provides risk warnings for fund portfolios
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RiskWarning:
    """Structure for risk warnings."""
    type: str
    severity: str
    message: str
    suggestion: str
    context: Optional[str] = None


class RiskAnalyzer:
    """Advanced risk analysis for portfolio funds."""
    
    def __init__(self, fund_data: Dict):
        self.fund_data = fund_data
        
    def generate_risk_warnings(self, fund_id: str) -> List[RiskWarning]:
        """Generate specific risk warnings based on fund characteristics"""
        
        if fund_id not in self.fund_data:
            return [RiskWarning(
                type="DATA_ERROR",
                severity="HIGH",
                message=f"Fund {fund_id} not found in portfolio",
                suggestion="Verify fund ID is correct"
            )]
        
        fund_info = self.fund_data[fund_id]
        metrics = fund_info['metrics']
        warnings = []
        
        # 1. Volatility Risk
        volatility = metrics.get('volatility', 0) / 100  # Convert to decimal
        if volatility > 0.15:  # 15% volatility threshold
            severity = "HIGH" if volatility > 0.25 else "MEDIUM"
            warnings.append(RiskWarning(
                type="VOLATILITY_RISK",
                severity=severity,
                message=f"Fund shows {volatility:.1%} volatility - above comfortable range",
                suggestion="Consider if this matches your risk tolerance"
            ))
        
        # 2. Negative Returns Risk
        cagr = metrics.get('cagr', 0) / 100  # Convert to decimal
        if cagr < 0:
            severity = "HIGH" if cagr < -0.10 else "MEDIUM"
            warnings.append(RiskWarning(
                type="NEGATIVE_RETURNS",
                severity=severity,
                message=f"Fund has negative returns ({cagr:.1%} CAGR)",
                suggestion="Review fund strategy and consider alternatives"
            ))
        
        # 3. Poor Risk-Adjusted Returns
        sharpe_ratio = metrics.get('sharpe_ratio', 0)
        if sharpe_ratio is not None and sharpe_ratio < 0.5 and volatility > 0.10:
            warnings.append(RiskWarning(
                type="EFFICIENCY_RISK",
                severity="MEDIUM",
                message="Low risk-adjusted returns for the volatility level",
                suggestion="Other funds may offer better risk-reward ratio",
                context=f"Sharpe ratio: {sharpe_ratio:.2f}"
            ))
        
        # 4. Extreme Underperformance 
        if sharpe_ratio is not None and sharpe_ratio < -1:
            warnings.append(RiskWarning(
                type="SEVERE_UNDERPERFORMANCE",
                severity="HIGH",
                message="Fund showing severe underperformance with negative risk-adjusted returns",
                suggestion="Consider immediate review and potential replacement"
            ))
        
        # 5. High Volatility with Low Returns
        if volatility > 0.20 and cagr < 0.05:  # High vol, low return
            warnings.append(RiskWarning(
                type="RISK_RETURN_MISMATCH",
                severity="MEDIUM",
                message="High volatility fund with low returns - poor risk compensation",
                suggestion="Consider lower-risk alternatives with similar returns"
            ))
        
        return warnings
    
    def generate_portfolio_risk_warnings(self, fund_ids: List[str]) -> Dict:
        """Generate risk warnings for a portfolio of funds"""
        portfolio_warnings = {}
        all_warnings = []
        
        for fund_id in fund_ids:
            fund_warnings = self.generate_risk_warnings(fund_id)
            portfolio_warnings[fund_id] = {
                'fund_name': self.fund_data.get(fund_id, {}).get('fund_name', 'Unknown'),
                'warnings': [
                    {
                        'type': w.type,
                        'severity': w.severity,
                        'message': w.message,
                        'suggestion': w.suggestion,
                        'context': w.context
                    } for w in fund_warnings
                ],
                'warning_count': len(fund_warnings),
                'risk_level': self._assess_fund_risk_level(fund_warnings)
            }
            all_warnings.extend(fund_warnings)
        
        # Portfolio-level risk assessment
        portfolio_risk = self._assess_portfolio_risk(fund_ids, all_warnings)
        
        return {
            'fund_warnings': portfolio_warnings,
            'portfolio_summary': {
                'total_warnings': len(all_warnings),
                'high_severity_count': len([w for w in all_warnings if w.severity == "HIGH"]),
                'medium_severity_count': len([w for w in all_warnings if w.severity == "MEDIUM"]),
                'portfolio_risk_level': portfolio_risk,
                'recommendations': self._generate_portfolio_recommendations(portfolio_risk, all_warnings)
            }
        }
    
    def _assess_fund_risk_level(self, warnings: List[RiskWarning]) -> str:
        """Assess overall risk level for a fund based on warnings"""
        if not warnings:
            return "LOW"
        
        high_count = len([w for w in warnings if w.severity == "HIGH"])
        medium_count = len([w for w in warnings if w.severity == "MEDIUM"])
        
        if high_count >= 2:
            return "VERY_HIGH"
        elif high_count >= 1:
            return "HIGH"
        elif medium_count >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _assess_portfolio_risk(self, fund_ids: List[str], warnings: List[RiskWarning]) -> str:
        """Assess overall portfolio risk level"""
        if not warnings:
            return "LOW"
        
        total_funds = len(fund_ids)
        high_severity_count = len([w for w in warnings if w.severity == "HIGH"])
        
        high_risk_ratio = high_severity_count / total_funds if total_funds > 0 else 0
        
        if high_risk_ratio > 0.5:
            return "HIGH"
        elif high_risk_ratio > 0.25 or len(warnings) > total_funds * 1.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_portfolio_recommendations(self, portfolio_risk: str, warnings: List[RiskWarning]) -> List[str]:
        """Generate portfolio-level recommendations based on risk assessment"""
        recommendations = []
        
        if portfolio_risk == "HIGH":
            recommendations.append("🚨 Immediate portfolio review recommended - multiple high-risk funds detected")
            recommendations.append("Consider diversifying into lower-risk assets")
        elif portfolio_risk == "MEDIUM":
            recommendations.append("⚠️ Monitor portfolio closely - some concerning patterns detected")
            recommendations.append("Review underperforming funds for potential replacement")
        
        # Specific recommendations based on warning types
        warning_types = [w.type for w in warnings]
        
        if warning_types.count("VOLATILITY_RISK") >= 2:
            recommendations.append("Multiple funds show high volatility - consider adding stable funds")
        
        if warning_types.count("NEGATIVE_RETURNS") >= 1:
            recommendations.append("Some funds have negative returns - review fund selection criteria")
        
        if not recommendations:
            recommendations.append("✅ Portfolio risk appears manageable - continue regular monitoring")
        
        return recommendations[:4]  # Limit to top 4 recommendations


# Demo function
def demo_risk_analyzer():
    """Demo the risk analyzer with sample data"""
    # Sample fund data (would come from ETL pipeline)
    sample_fund_data = {
        'F001': {
            'fund_name': 'Axis Bluechip Fund',
            'metrics': {'cagr': -20.48, 'volatility': 7.2, 'sharpe_ratio': -2.716}
        },
        'F003': {
            'fund_name': 'SBI Small Cap Fund', 
            'metrics': {'cagr': 39.39, 'volatility': 7.78, 'sharpe_ratio': 2.4713}
        },
        'F008': {
            'fund_name': 'UTI Bond Fund',
            'metrics': {'cagr': -2.93, 'volatility': 6.99, 'sharpe_ratio': -0.8324}
        }
    }
    
    analyzer = RiskAnalyzer(sample_fund_data)
    
    # Test portfolio risk warnings
    portfolio_funds = ['F001', 'F003', 'F008']
    risk_warnings = analyzer.generate_portfolio_risk_warnings(portfolio_funds)
    
    print("🔍 Portfolio Risk Analysis:")
    print(f"Total Warnings: {risk_warnings['portfolio_summary']['total_warnings']}")
    print(f"Portfolio Risk Level: {risk_warnings['portfolio_summary']['portfolio_risk_level']}")
    
    print(f"\n✅ Risk analysis completed for {len(portfolio_funds)} funds")

if __name__ == "__main__":
    demo_risk_analyzer()
