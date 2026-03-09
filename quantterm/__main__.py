"""QuantTerm main entry point."""
import sys
import os
import warnings

# Suppress ALL warnings BEFORE any imports - this must be first
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*Timestamp.utcnow.*')
warnings.filterwarnings('ignore', message='.*Pandas.*')

# yfinance warnings are handled by SuppressStderr in cli.main
# Do NOT suppress stderr globally - errors should be visible

from quantterm.cli.main import main

if __name__ == "__main__":
    main()
