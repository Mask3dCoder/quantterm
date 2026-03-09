# Contributing to QuantTerm

Welcome to QuantTerm! We're excited to have you contribute to the open-source Bloomberg Terminal for Python quants.

## Quick Start

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/quantterm.git`
3. **Create** a virtual environment: `python -m venv venv && source venv/bin/activate`
   - On Windows: `python -m venv venv && venv\Scripts\activate`
4. **Install** dependencies: `poetry install`
5. **Create** a branch: `git checkout -b feature/your-feature-name`

## Windows Development

On Windows, you can test QuantTerm using multiple methods:

```powershell
# Method 1: Using python -m (recommended)
python -m quantterm quote AAPL

# Method 2: Using the wrapper script
.\scripts\quantterm.bat quote AAPL

# Method 3: After pip install -e
pip install -e .
quantterm quote AAPL  # May require terminal restart
```

**Important**: Test your changes with both `python -m quantterm` and `quantterm` invocation methods.

## Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_backtest.py

# Run with coverage
pytest --cov=quantterm --cov-report=html
```

### Code Style

We use several tools to maintain code quality:

```bash
# Format code
black .

# Lint
ruff check .

# Type checking
mypy quantterm/
```

### Pre-commit Hooks

We recommend installing pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Project Structure

```
quantterm/
├── quantterm/           # Main package
│   ├── backtesting/    # Event-driven backtesting engine
│   ├── ml/             # Machine learning pipeline
│   ├── optimization/    # Bayesian optimization, walk-forward
│   ├── fixed_income/   # Bond pricing, yield curves
│   ├── portfolio/      # Portfolio optimization, risk metrics
│   ├── analytics/      # Technical indicators
│   ├── derivatives/    # Options pricing (Black-Scholes)
│   ├── data/           # Data providers (Yahoo, FRED)
│   ├── utils/          # Resilience, telemetry
│   └── cli/            # Typer commands
├── tests/               # Test suite
├── docs/                # Documentation
└── examples/           # Example strategies
```

## Adding New Features

### CLI Commands

To add a new CLI command group:

1. Create a new file in `quantterm/cli/commands/`
2. Define your commands using Typer
3. Register in `quantterm/cli/main.py`

### Strategies

To add a new strategy:

1. Create a new file in `quantterm/backtesting/strategy/`
2. Extend the `Strategy` base class
3. Implement `on_bar()` method

### Indicators

To add a new technical indicator:

1. Add to `quantterm/analytics/technical/indicators.py`
2. Follow existing naming conventions
3. Add tests in `tests/`

## Submitting a Pull Request

1. Ensure all tests pass
2. Run code formatters: `black . && ruff check .`
3. Update documentation if needed
4. Submit PR with clear description

## Issues

- Use GitHub Issues for bug reports
- Use feature requests for new features
- Provide minimal reproduction steps for bugs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
