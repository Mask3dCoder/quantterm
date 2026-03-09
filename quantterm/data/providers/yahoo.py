"""Yahoo Finance data provider for QuantTerm."""
import warnings
from datetime import datetime, timedelta
from typing import Optional, Union
import numpy as np
import pandas as pd

# Suppress pandas warnings from yfinance (must be before import)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Timestamp.utcnow.*')
warnings.filterwarnings('ignore', message='.*Pandas.*')

import yfinance as yf


def get_quote(ticker: str) -> dict:
    """Get real-time quote from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with quote data
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', message='.*utcnow.*')
        
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        info = stock.info
    
    return {
        'symbol': ticker,
        'price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
        'bid': info.get('bid', 0),
        'ask': info.get('ask', 0),
        'volume': info.get('volume', 0),
        'market_cap': info.get('marketCap', 0),
        'pe_ratio': info.get('peRatio', 0),
        'dividend_yield': info.get('dividendYield', 0),
        '52w_high': info.get('fiftyTwoWeekHigh', 0),
        '52w_low': info.get('fiftyTwoWeekLow', 0),
    }


def get_history(
    ticker: str,
    start: Union[str, datetime, pd.Timestamp],
    end: Union[str, datetime, pd.Timestamp] = None,
    interval: str = "1d",
    adjust: bool = True
) -> pd.DataFrame:
    """Get historical price data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start: Start date
        end: End date (default: today)
        interval: Data interval (1d, 1h, 1m, etc.)
        adjust: Adjust for splits and dividends
        
    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf
    
    if end is None:
        end = pd.Timestamp.now()
    
    # Convert dates
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    
    # Download data
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end, interval=interval, auto_adjust=adjust)
    
    return df


def get_options(ticker: str, expiration: str = None) -> dict:
    """Get options chain from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        expiration: Specific expiration date (optional)
        
    Returns:
        Dictionary with calls and puts DataFrames
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    
    if expiration:
        # Get specific expiration
        try:
            options = stock.option_chain(expiration)
            return {
                'calls': options.calls,
                'puts': options.puts,
                'expiration': expiration
            }
        except (KeyError, ValueError) as e:
            return {'calls': pd.DataFrame(), 'puts': pd.DataFrame()}
    else:
        # Get all expirations
        expirations = stock.options
        return {
            'expirations': expirations
        }


def get_fundamentals(ticker: str) -> dict:
    """Get fundamental data from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with fundamental data
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    info = stock.info
    
    return {
        # Valuation
        'market_cap': info.get('marketCap'),
        'enterprise_value': info.get('enterpriseValue'),
        'trailing_pe': info.get('trailingPE'),
        'forward_pe': info.get('forwardPE'),
        'peg_ratio': info.get('pegRatio'),
        'price_to_book': info.get('priceToBook'),
        'price_to_sales': info.get('priceToSalesTrailing12Months'),
        'enterprise_to_revenue': info.get('enterpriseToRevenue'),
        'enterprise_to_ebitda': info.get('enterpriseToEbitda'),
        
        # Growth
        'revenue_growth': info.get('revenueGrowth'),
        'earnings_growth': info.get('earningsGrowth'),
        'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth'),
        'revenue_quarterly_growth': info.get('revenueQuarterlyGrowth'),
        
        # Dividends
        'dividend_yield': info.get('dividendYield'),
        'dividend_rate': info.get('dividendRate'),
        'ex_dividend_date': info.get('exDividendDate'),
        
        # Profitability
        'profit_margins': info.get('profitMargins'),
        'operating_margins': info.get('operatingMargins'),
        'roe': info.get('returnOnEquity'),
        'roa': info.get('returnOnAssets'),
        
        # Financial Health
        'debt_to_equity': info.get('debtToEquity'),
        'current_ratio': info.get('currentRatio'),
        'quick_ratio': info.get('quickRatio'),
        
        # Size
        'total_cash': info.get('totalCash'),
        'total_debt': info.get('totalDebt'),
        'total_revenue': info.get('totalRevenue'),
        'ebitda': info.get('ebitda'),
    }


def get_institutional_ownership(ticker: str) -> pd.DataFrame:
    """Get institutional ownership data.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with institutional holders
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    
    try:
        holders = stock.institutional_holders
        return holders
    except (KeyError, ValueError, Exception) as e:
        return pd.DataFrame()


def get_mutualfund_ownership(ticker: str) -> pd.DataFrame:
    """Get mutual fund ownership data.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        DataFrame with mutual fund holders
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    
    try:
        holders = stock.mutualfund_holders
        return holders
    except (KeyError, ValueError, Exception) as e:
        return pd.DataFrame()


def get_earnings(ticker: str) -> dict:
    """Get earnings data and history.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with earnings data
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    
    return {
        'calendar': stock.calendar,
        'earnings_dates': stock.earnings_dates,
        'financials': {
            'income': stock.financials,
            'balance_sheet': stock.balance_sheet,
            'cashflow': stock.cashflow,
        }
    }


def get_candles(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
) -> pd.DataFrame:
    """Get candle data (OHLCV) for charting.
    
    Args:
        ticker: Stock ticker symbol
        period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Interval (1d, 1h, 1m, etc.)
        
    Returns:
        DataFrame with OHLCV data
    """
    import yfinance as yf
    
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    
    return df


# Convenience functions for quick data access

def search_ticker(query: str) -> list[dict]:
    """Search for tickers matching a query.
    
    Args:
        query: Search query (company name or keyword)
        
    Returns:
        List of dictionaries with ticker information
    """
    import yfinance as yf
    
    try:
        # Use yfinance's Search class
        search_result = yf.Search(query)
        quotes = search_result.quotes
        
        if quotes is None:
            return []
        
        results = []
        for item in quotes:
            results.append({
                'symbol': item.get('symbol', ''),
                'shortname': item.get('shortname', ''),
                'longname': item.get('longname', ''),
                'type': item.get('quoteType', ''),
                'exchange': item.get('exchange', ''),
            })
        
        return results
    except Exception as e:
        return []

def get_price(ticker: str) -> float:
    """Get current price for a ticker."""
    quote = get_quote(ticker)
    return quote.get('price', 0)


def get_prices(tickers: list[str]) -> pd.Series:
    """Get current prices for multiple tickers."""
    import yfinance as yf
    
    data = yf.download(tickers, progress=False)['Close']
    
    if data.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")
    
    # Handle single column result (one ticker)
    if isinstance(data, pd.Series):
        return data.iloc[[-1]]
    
    return data.iloc[-1]


def get_returns(
    tickers: list[str],
    start: str = "1y",
    end: str = None
) -> pd.DataFrame:
    """Get returns for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start: Start date or period string
        end: End date
        
    Returns:
        DataFrame of daily returns
    """
    import yfinance as yf
    
    if end is None:
        end = datetime.utcnow()
    
    data = yf.download(tickers, start=start, end=end, progress=False)['Adj Close']
    returns = data.pct_change().dropna()
    
    return returns


def get_portfolio_returns(
    tickers: list[str],
    weights: list[float],
    start: str = "1y",
    end: str = None
) -> pd.Series:
    """Get weighted portfolio returns.
    
    Args:
        tickers: List of ticker symbols
        weights: Portfolio weights
        start: Start date or period string
        end: End date
        
    Returns:
        Series of portfolio returns
    """
    returns = get_returns(tickers, start, end)
    portfolio_returns = (returns * weights).sum(axis=1)
    
    return portfolio_returns
