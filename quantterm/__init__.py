"""QuantTerm - Institutional-grade quantitative finance CLI platform."""
import warnings

# Suppress ALL warnings BEFORE any imports - critical for yfinance
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Timestamp.utcnow.*')
warnings.filterwarnings('ignore', message='.*Pandas.*')

__version__ = "0.1.0"
__author__ = "QuantTerm Team"
__license__ = "Commercial"

from quantterm.core import models  # noqa: F401
