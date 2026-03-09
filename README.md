# QuantTerm

> The open-source Bloomberg Terminal for Python quants. Event-driven backtesting, Bayesian optimization, and ML pipelines—completely free and local.

[![Tests](https://github.com/quantterm/quantterm/workflows/CI/badge.svg)](https://github.com/quantterm/quantterm/actions)
[![PyPI](https://img.shields.io/pypi/v/quantterm)](https://pypi.org/project/quantterm/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org/)

**QuantTerm** is an institutional-grade quantitative finance CLI for researchers, quants, and algorithmic traders. It combines the analytical power of Bloomberg with the flexibility of Python—completely open source.

## Why QuantTerm?

| Feature | QuantTerm | Bloomberg | QuantConnect | Zipline |
|---------|-----------|-----------|--------------|---------|
| **Cost** | Free | $24K/year | Free/$20/mo | Free (dead) |
| **Open Source** | Full | None | Partial | Full |
| **Local Execution** | Yes | Terminal only | Cloud only | Yes |
| **Event-Driven** | Yes | No | Yes | Yes |
| **Bayesian Optimization** | Built-in | No | No | No |
| **ML Pipeline** | Integrated | Limited | Yes | No |
| **Fixed Income** | Bonds/Yield | Yes | No | No |
| **Python Native** | Yes | Excel/API | C#/Python | Yes |

## 30 Seconds to Your First Backtest

```bash
# Install
pip install quantterm

# Run a backtest
quantterm backtest run --symbol SPY --start 2020-01-01 --end 2023-12-31
```

**Output:**
```
Running backtest...
Symbol: SPY
Period: 2022-01-01 to 2022-12-31
Capital: $1,000,000.00
Strategy: BuyAndHold
--------------------------------------------------

==================================================
PERFORMANCE SUMMARY
==================================================
Initial Capital:  $1,000,000.00
Final Value:      $822,392.60
Total Return:     -17.76%
Annualized Ret:   -17.82%
--------------------------------------------------
Sharpe Ratio:     -0.78
Max Drawdown:    -23.32%
Win Rate:        43.03%
```

## Features

### 1. Event-Driven Backtesting

Realistic simulation with slippage, commissions, and market impact. Nanosecond-precision event loop ensures deterministic results.

```bash
quantterm backtest run \
  --symbol SPY \
  --start 2020-01-01 \
  --end 2023-12-31 \
  --capital 1000000
```

### 2. Bayesian Optimization

Find optimal parameters 10x faster than grid search using Gaussian Processes.

```bash
quantterm optimize bayesian Rebalancing \
  --symbols SPY QQQ \
  --start 2020-01-01 \
  --end 2023-12-31 \
  --iterations 30
```

### 3. Machine Learning Pipeline

Train models with strict no-lookahead-bias guarantees. Time-series cross-validation prevents data leakage.

```bash
# Train on historical data
quantterm ml train \
  --symbol SPY \
  --start 2015-01-01 \
  --end 2020-12-31 \
  --output momentum_model.pkl

# Backtest on out-of-sample data
quantterm ml backtest momentum_model.pkl \
  --symbol SPY \
  --start 2021-01-01 \
  --end 2023-12-31
```

### 4. Fixed Income Analytics

Institutional-grade bond pricing, duration, convexity, and yield curve analysis.

```bash
quantterm fixed-income price --coupon 0.05 --maturity 2030-01-01 --yield 0.04
```

**Output:**
```
Bond Pricing
Maturity: 2030-01-01
Coupon: 5.00%
Yield: 4.00%
----------------------------------------
          Bond Prices           
+------------------------------+
| Price Type       | Amount    |
|------------------+-----------|
| Dirty Price      | $1,025.18 |
| Clean Price      | $1,016.01 |
| Accrued Interest | $9.17     |
| Face Value       | $1,000.00 |
+------------------------------+
```

### 5. Technical Analysis (50+ Indicators)

```bash
# RSI
quantterm tech indicator SPY rsi --period 14

# MACD
quantterm tech indicator AAPL macd --fast 12 --slow 26

# Bollinger Bands
quantterm tech indicator TSLA bollinger --period 20
```

### 6. Risk Management

```bash
# Value at Risk
quantterm risk var SPY --notional 100000 --confidence 0.95

# Historical VaR
quantterm risk var SPY --notional 100000 --method historical
```

## Quick Start

### Installation

```bash
pip install quantterm
```

Or with Poetry (recommended for development):

```bash
git clone https://github.com/quantterm/quantterm.git
cd quantterm
poetry install
```

### Your First Backtest

```bash
quantterm backtest run \
  --symbol SPY \
  --start 2020-01-01 \
  --end 2023-12-31
```

### Windows Users

If `quantterm` command is not found after installation on Windows:

1. **Use python -m** (recommended):
   ```powershell
   python -m quantterm quote AAPL
   ```

2. **Restart your terminal** and try again

3. **Use wrapper scripts** (included in `scripts/`):
   ```powershell
   .\scripts\quantterm.bat quote AAPL
   # Or in PowerShell:
   .\scripts\quantterm.ps1 quote AAPL
   ```

For detailed Windows installation instructions, see [docs/installation/windows.md](docs/installation/windows.md).

### Optimize a Strategy

```bash
quantterm optimize bayesian Rebalancing \
  --symbols SPY QQQ \
  --start 2020-01-01 \
  --end 2023-12-31 \
  --iterations 30
```

### Train an ML Model

```bash
quantterm ml train \
  --symbol SPY \
  --start 2015-01-01 \
  --end 2020-12-31 \
  --output my_model.pkl
```

### Price a Bond

```bash
quantterm fixed-income price \
  --coupon 0.05 \
  --maturity 2030-01-01 \
  --yield 0.04
```

## Architecture

QuantTerm uses an **event-driven architecture** for realistic, deterministic backtesting:

```
Market Data → BarEvents → Strategy.on_bar() → Signals → 
ExecutionEngine (slippage, costs) → Portfolio → Metrics
                ↑_________________________________________↓
                    (Realistic costs, delays, slippage)
```

### Key Design Principles

- **No lookahead bias**: Strict point-in-time data access guarantees
- **Deterministic**: Same random seed produces identical results
- **Extensible**: Plugin architecture for custom strategies
- **Fast**: Optimized for millions of events per second

### Module Organization

```
quantterm/
├── backtesting/     # Event-driven engines
├── ml/             # Feature engineering, model training
├── optimization/   # Bayesian, walk-forward
├── fixed_income/   # Bonds, yield curves
├── portfolio/      # Optimization, risk
├── analytics/      # Technical indicators
├── derivatives/    # Options pricing
├── data/           # Yahoo Finance, FRED
├── utils/          # Circuit breaker, telemetry
└── cli/            # Typer/Rich commands
```

## Documentation

- [User Guide](docs/guide/) - Comprehensive tutorials
- [API Reference](docs/api/) - Auto-generated from docstrings
- [Examples](examples/) - Ready-to-run strategies

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE)

## Support

- GitHub Issues: Report bugs and feature requests
- Discord: Join our community

---

**Disclaimer**: QuantTerm is for research and educational purposes. It is not investment advice. Past performance does not guarantee future results. Always validate strategies with proper out-of-sample testing before using with real capital.
