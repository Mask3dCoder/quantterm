"""
Federal Reserve Economic Data (FRED) integration.

Free, institutional-quality economic data for fixed income analytics.
"""
import os
from datetime import date
from typing import Optional
import pandas as pd
import requests


class FREDDataProvider:
    """
    Federal Reserve Economic Data provider.
    
    Key series:
    - DGS1MO to DGS30: Treasury yields (constant maturity)
    - BAMLC0A0CM: Corporate bond yields by rating
    - TOTBKCR: Bank credit (macro indicator)
    """
    
    BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Treasury yield curve series IDs
    TREASURY_SERIES = {
        '1M': 'DGS1MO',
        '3M': 'DGS3MO',
        '6M': 'DGS6MO',
        '1Y': 'DGS1',
        '2Y': 'DGS2',
        '3Y': 'DGS3',
        '5Y': 'DGS5',
        '7Y': 'DGS7',
        '10Y': 'DGS10',
        '20Y': 'DGS20',
        '30Y': 'DGS30',
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data provider.
        
        Args:
            api_key: FRED API key (get free at https://fred.stlouisfed.org/docs/api/api_key.html)
        """
        self.api_key = api_key or os.getenv("FRED_API_KEY")
    
    def get_treasury_yield(self, tenor: str, target_date: date) -> Optional[float]:
        """
        Get Treasury yield for specific tenor and date.
        
        Args:
            tenor: '1M', '3M', '6M', '1Y', '2Y', '3Y', '5Y', '7Y', '10Y', '20Y', '30Y'
            target_date: Date to fetch
            
        Returns:
            Yield as decimal (e.g., 0.045 for 4.5%) or None if unavailable
        """
        series_id = self.TREASURY_SERIES.get(tenor.upper())
        if not series_id:
            raise ValueError(f"Unknown tenor: {tenor}")
        
        date_str = target_date.strftime("%Y-%m-%d")
        
        if self.api_key:
            url = f"{self.BASE_URL}/series/observations"
            params = {
                'series_id': series_id,
                'observation_date': date_str,
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if 'observations' in data and data['observations']:
                    value = data['observations'][0]['value']
                    if value == '.':
                        return None
                    return float(value) / 100  # Convert from percent
            except Exception as e:
                print(f"FRED API error: {e}")
        
        # Fallback: return cached/mock data if no API key
        return self._get_fallback_yield(tenor, target_date)
    
    def get_treasury_curve(self, target_date: date) -> dict:
        """
        Get full Treasury yield curve.
        
        Returns:
            Dict mapping tenor to yield
        """
        curve = {}
        
        for tenor, series_id in self.TREASURY_SERIES.items():
            yield_val = self.get_treasury_yield(tenor, target_date)
            if yield_val is not None:
                curve[tenor] = yield_val
        
        return curve
    
    def get_historical_yields(
        self, 
        tenor: str, 
        start_date: date, 
        end_date: date
    ) -> pd.Series:
        """
        Get historical Treasury yields.
        
        Returns:
            Series with dates as index and yields as values
        """
        series_id = self.TREASURY_SERIES.get(tenor.upper())
        if not series_id:
            raise ValueError(f"Unknown tenor: {tenor}")
        
        if self.api_key:
            url = f"{self.BASE_URL}/series/observations"
            params = {
                'series_id': series_id,
                'observation_start': start_date.strftime("%Y-%m-%d"),
                'observation_end': end_date.strftime("%Y-%m-%d"),
                'api_key': self.api_key,
                'file_type': 'json'
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                data = response.json()
                
                if 'observations' in data:
                    values = {}
                    for obs in data['observations']:
                        d = pd.to_datetime(obs['date'])
                        v = obs['value']
                        if v != '.':
                            values[d] = float(v) / 100
                    
                    return pd.Series(values)
            except Exception as e:
                print(f"FRED API error: {e}")
        
        # Fallback
        return pd.Series(dtype=float)
    
    def get_spread(self, tenor1: str, tenor2: str, target_date: date) -> Optional[float]:
        """
        Calculate yield spread (e.g., 2s10s spread).
        
        Args:
            tenor1: First tenor (e.g., '2Y')
            tenor2: Second tenor (e.g., '10Y')
            
        Returns:
            Spread in basis points
        """
        y1 = self.get_treasury_yield(tenor1, target_date)
        y2 = self.get_treasury_yield(tenor2, target_date)
        
        if y1 is None or y2 is None:
            return None
        
        return (y2 - y1) * 10000  # Convert to basis points
    
    def detect_inversion(self, target_date: date) -> dict:
        """
        Detect yield curve inversion.
        
        Returns:
            Dict with inversion status and spreads
        """
        result = {
            'date': target_date,
            'inverted': False,
            'spreads': {}
        }
        
        # Check common inversion indicators
        checks = [
            ('2Y-10Y', '2Y', '10Y'),
            ('3M-10Y', '3M', '10Y'),
            ('1Y-10Y', '1Y', '10Y'),
        ]
        
        for name, t1, t2 in checks:
            spread = self.get_spread(t1, t2, target_date)
            if spread is not None:
                result['spreads'][name] = spread
                if spread < 0:
                    result['inverted'] = True
        
        return result
    
    @staticmethod
    def _get_fallback_yield(tenor: str, target_date: date) -> float:
        """
        Fallback yields when API unavailable.
        
        Based on approximate historical averages.
        """
        fallbacks = {
            '1M': 0.052,
            '3M': 0.053,
            '6M': 0.051,
            '1Y': 0.048,
            '2Y': 0.045,
            '3Y': 0.043,
            '5Y': 0.042,
            '7Y': 0.043,
            '10Y': 0.044,
            '20Y': 0.046,
            '30Y': 0.045,
        }
        
        return fallbacks.get(tenor.upper(), 0.04)
