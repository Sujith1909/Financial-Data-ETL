"""
ETL Pipeline for Fund NAV Data Processing and Analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

warnings.filterwarnings('ignore')


class FundNAVETL:
    """ETL Pipeline for processing fund NAV data with cleaning and analytics."""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.raw_data = None
        self.cleaned_data = None
        self.analytics_data = None
        self.risk_free_rate = 0.04  # 4% annual risk-free rate
        
    def extract(self) -> pd.DataFrame:
        """Extract data from CSV file."""
        try:
            self.raw_data = pd.read_csv(self.csv_path)
            print(f"Extracted {len(self.raw_data)} records from {self.csv_path}")
            return self.raw_data
        except Exception as e:
            raise Exception(f"Failed to extract data: {str(e)}")
    
    def transform(self) -> pd.DataFrame:
        """Transform and clean the data."""
        if self.raw_data is None:
            raise ValueError("No data to transform. Run extract() first.")
        
        df = self.raw_data.copy()
        
        # 1. Parse and normalize dates
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. Handle missing NAVs
        df['nav'] = pd.to_numeric(df['nav'], errors='coerce')
        
        # 3. Detect and handle outliers using statistical methods
        df = self._detect_and_handle_outliers(df)
        
        # 4. Handle missing values after outlier removal
        df = self._handle_missing_values(df)
        
        # 5. Remove duplicates (if any)
        df = df.drop_duplicates(subset=['fund_id', 'date'], keep='last')
        
        # 6. Sort by fund and date
        df = df.sort_values(['fund_id', 'date']).reset_index(drop=True)

        
        self.cleaned_data = df
        print(f"Transformed data: {len(df)} records after cleaning")
        return df
    
    def _detect_and_handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers using multiple methods."""
        df_clean = df.copy()
        outliers_removed = 0
        
        for fund_id in df['fund_id'].unique():
            fund_mask = df_clean['fund_id'] == fund_id
            fund_data = df_clean[fund_mask].copy()
            
            if len(fund_data) < 5:  # Skip if too few data points
                continue
                
            navs = fund_data['nav'].dropna()
            if len(navs) < 3:
                continue
            
            # Method 1: IQR-based outlier detection
            Q1 = navs.quantile(0.25)
            Q3 = navs.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Method 2: Z-score based (for extreme outliers)
            z_scores = np.abs((navs - navs.mean()) / navs.std())
            z_threshold = 3
            
            # Method 3: Percentage change based detection
            fund_data_sorted = fund_data.sort_values('date')
            fund_data_sorted['pct_change'] = fund_data_sorted['nav'].pct_change()
            extreme_changes = np.abs(fund_data_sorted['pct_change']) > 0.5  # 50% change
            
            # Combine outlier detection methods
            outlier_mask = (
                (fund_data['nav'] < lower_bound) | 
                (fund_data['nav'] > upper_bound) |
                (fund_data.index.isin(fund_data_sorted[extreme_changes].index))
            )
            
            # Additional check for z-score
            for idx in fund_data.index:
                if not pd.isna(fund_data.loc[idx, 'nav']):
                    nav_val = fund_data.loc[idx, 'nav']
                    z_score = abs((nav_val - navs.mean()) / navs.std()) if navs.std() > 0 else 0
                    if z_score > z_threshold:
                        outlier_mask.loc[idx] = True
            
            # Mark outliers as NaN for interpolation
            outlier_indices = fund_data[outlier_mask].index
            df_clean.loc[outlier_indices, 'nav'] = np.nan
            outliers_removed += len(outlier_indices)
            
            if len(outlier_indices) > 0:
                print(f"Fund {fund_id}: Removed {len(outlier_indices)} outliers")
        
        print(f"Total outliers detected and marked for interpolation: {outliers_removed}")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing NAV values using forward fill and interpolation."""
        df_filled = df.copy()
        
        for fund_id in df['fund_id'].unique():
            fund_mask = df_filled['fund_id'] == fund_id
            fund_data = df_filled[fund_mask].copy()
            
            # Sort by date
            fund_data = fund_data.sort_values('date')
            
            # Forward fill first, then backward fill, then interpolate
            fund_data['nav'] = fund_data['nav'].ffill()
            fund_data['nav'] = fund_data['nav'].bfill()
            fund_data['nav'] = fund_data['nav'].interpolate(method='linear')
            
            # Update the main dataframe
            df_filled.loc[fund_mask, 'nav'] = fund_data['nav'].values
        
        # Remove any remaining rows with missing NAVs
        before_count = len(df_filled)
        df_filled = df_filled.dropna(subset=['nav'])
        after_count = len(df_filled)
        
        if before_count > after_count:
            print(f"Removed {before_count - after_count} rows with unrecoverable missing NAVs")
        
        return df_filled
    
    def load_and_compute_analytics(self) -> Dict:
        """Compute analytics metrics for each fund."""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Run transform() first.")
        
        analytics_results = {}
        df = self.cleaned_data.copy()
        
        for fund_id in df['fund_id'].unique():
            fund_data = df[df['fund_id'] == fund_id].sort_values('date')
            fund_name = fund_data['fund_name'].iloc[0]
            
            metrics = self._calculate_fund_metrics(fund_data)
            
            analytics_results[fund_id] = {
                'fund_name': fund_name,
                'metrics': metrics,
                'data_points': len(fund_data),
                'date_range': {
                    'start': fund_data['date'].min().strftime('%Y-%m-%d'),
                    'end': fund_data['date'].max().strftime('%Y-%m-%d')
                }
            }
        
        self.analytics_data = analytics_results
        return analytics_results
    
    def _calculate_fund_metrics(self, fund_data: pd.DataFrame) -> Dict:
        """Calculate CAGR, Volatility, and Sharpe Ratio for a fund."""
        if len(fund_data) < 2:
            return {
                'cagr': None,
                'volatility': None,
                'sharpe_ratio': None,
                'error': 'Insufficient data points'
            }
        
        # Calculate daily returns
        fund_data = fund_data.sort_values('date')
        fund_data['daily_return'] = fund_data['nav'].pct_change()
        daily_returns = fund_data['daily_return'].dropna()
        
        if len(daily_returns) == 0:
            return {
                'cagr': None,
                'volatility': None,
                'sharpe_ratio': None,
                'error': 'No valid returns calculated'
            }
        
        # Calculate CAGR
        start_nav = fund_data['nav'].iloc[0]
        end_nav = fund_data['nav'].iloc[-1]
        start_date = fund_data['date'].iloc[0]
        end_date = fund_data['date'].iloc[-1]
        
        days_diff = (end_date - start_date).days
        years = days_diff / 365.25
        
        if years > 0 and start_nav > 0:
            cagr = ((end_nav / start_nav) ** (1 / years)) - 1
        else:
            cagr = None
        
        # Calculate Volatility (annualized)
        volatility = daily_returns.std() * np.sqrt(252)  # 252 trading days in a year
        
        # Calculate Sharpe Ratio
        avg_daily_return = daily_returns.mean()
        excess_return = avg_daily_return - (self.risk_free_rate / 252)  # Daily risk-free rate
        
        if volatility > 0:
            sharpe_ratio = (excess_return * 252) / volatility  # Annualized Sharpe ratio
        else:
            sharpe_ratio = None
        
        return {
            'cagr': round(cagr * 100, 2) if cagr is not None else None,  # Convert to percentage
            'volatility': round(volatility * 100, 2) if volatility is not None else None,  # Convert to percentage
            'sharpe_ratio': round(sharpe_ratio, 4) if sharpe_ratio is not None else None,
            'avg_daily_return': round(avg_daily_return * 100, 4) if avg_daily_return is not None else None,
            'total_return': round(((end_nav / start_nav) - 1) * 100, 2) if start_nav > 0 else None
        }
    
    def get_portfolio_metrics(self, weights: Optional[Dict[str, float]] = None) -> Dict:
        """Calculate portfolio-level metrics with optional weights."""
        if self.analytics_data is None:
            raise ValueError("No analytics data available. Run load_and_compute_analytics() first.")
        
        # Default equal weights if not provided
        if weights is None:
            fund_ids = list(self.analytics_data.keys())
            weights = {fund_id: 1.0 / len(fund_ids) for fund_id in fund_ids}
        
        # Validate weights
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            # Normalize weights
            weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate weighted metrics
        weighted_cagr = 0
        weighted_volatility = 0
        valid_funds = 0
        
        for fund_id, weight in weights.items():
            if fund_id in self.analytics_data:
                metrics = self.analytics_data[fund_id]['metrics']
                if metrics.get('cagr') is not None:
                    weighted_cagr += metrics['cagr'] * weight
                    valid_funds += 1
                if metrics.get('volatility') is not None:
                    weighted_volatility += (metrics['volatility'] ** 2) * (weight ** 2)
        
        # Portfolio volatility (simplified - assumes zero correlation)
        portfolio_volatility = np.sqrt(weighted_volatility) if weighted_volatility > 0 else None
        
        # Portfolio Sharpe ratio
        portfolio_excess_return = (weighted_cagr / 100) - self.risk_free_rate
        portfolio_sharpe = portfolio_excess_return / (portfolio_volatility / 100) if portfolio_volatility and portfolio_volatility > 0 else None
        
        return {
            'weighted_cagr': round(weighted_cagr, 2),
            'portfolio_volatility': round(portfolio_volatility, 2) if portfolio_volatility else None,
            'portfolio_sharpe_ratio': round(portfolio_sharpe, 4) if portfolio_sharpe else None,
            'num_funds': len(weights),
            'valid_funds': valid_funds,
            'weights': weights
        }
    
    def run_full_pipeline(self) -> Dict:
        """Run the complete ETL pipeline."""
        print("Starting ETL Pipeline...")
        
        # Extract
        self.extract()
        
        # Transform
        self.transform()
        
        # Load and compute analytics
        analytics = self.load_and_compute_analytics()
        
        # Calculate portfolio metrics
        portfolio_metrics = self.get_portfolio_metrics()
        
        print("ETL Pipeline completed successfully!")
        
        return {
            'fund_analytics': analytics,
            'portfolio_metrics': portfolio_metrics,
            'data_summary': {
                'total_funds': len(analytics),
                'total_records': len(self.cleaned_data) if self.cleaned_data is not None else 0,
                'date_range': self._get_overall_date_range()
            }
        }
    
    def _get_overall_date_range(self) -> Dict:
        """Get the overall date range of the dataset."""
        if self.cleaned_data is None:
            return {}
        
        return {
            'start': self.cleaned_data['date'].min().strftime('%Y-%m-%d'),
            'end': self.cleaned_data['date'].max().strftime('%Y-%m-%d')
        }


if __name__ == "__main__":
    # Example usage
    etl = FundNAVETL("fund_navs_10funds.csv")
    results = etl.run_full_pipeline()
    
    print("\n=== FUND ANALYTICS SUMMARY ===")
    for fund_id, data in results['fund_analytics'].items():
        print(f"\n{fund_id} - {data['fund_name']}")
        metrics = data['metrics']
        print(f"  CAGR: {metrics.get('cagr', 'N/A')}%")
        print(f"  Volatility: {metrics.get('volatility', 'N/A')}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
    
    print("\n=== PORTFOLIO METRICS ===")
    portfolio = results['portfolio_metrics']
    print(f"Weighted CAGR: {portfolio['weighted_cagr']}%")
    print(f"Portfolio Volatility: {portfolio['portfolio_volatility']}%")
    print(f"Portfolio Sharpe Ratio: {portfolio['portfolio_sharpe_ratio']}")
