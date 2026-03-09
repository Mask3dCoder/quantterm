"""Backtest engine for QuantTerm.

Event-driven backtesting engine that processes bars sequentially
and generates trading signals from strategies.
Supports both single and multi-symbol backtesting.
"""

from typing import Dict, Optional, List

from quantterm.backtesting.events import BarEvent, OrderEvent
from quantterm.backtesting.portfolio import Portfolio
from quantterm.backtesting.data_handler import DataHandler, MultiSymbolDataHandler, IntradayDataHandler, IntradayFillModel
from quantterm.backtesting.execution import Execution


class BacktestEngine:
    """Event-driven backtest engine.
    
    Processes historical data bar-by-bar, generates trading signals
    from strategies, and simulates order execution.
    """
    
    def __init__(
        self,
        strategy_class,
        initial_capital: float = 1000000.0,
        data_handler: Optional[DataHandler] = None,
        execution: Optional[Execution] = None,
    ):
        """Initialize backtest engine.
        
        Args:
            strategy_class: Strategy class to use for signal generation.
            initial_capital: Starting capital for the backtest.
            data_handler: Data handler for fetching market data.
            execution: Execution handler for simulating trades.
        """
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.data_handler = data_handler or DataHandler()
        self.execution = execution or Execution()
    
    def run(
        self,
        symbol: str,
        start: str,  # "YYYY-MM-DD"
        end: str,    # "YYYY-MM-DD"
    ) -> dict:
        """Run backtest for a single symbol.
        
        Args:
            symbol: Ticker symbol to backtest.
            start: Start date in 'YYYY-MM-DD' format.
            end: End date in 'YYYY-MM-DD' format.
            
        Returns:
            Dictionary with backtest results including:
            - strategy: Strategy name
            - symbol: Symbol tested
            - start/end: Date range
            - initial_cash: Starting capital
            - final_value: Portfolio value at end
            - realized_pnl: Realized profit/loss
            - unrealized_pnl: Unrealized profit/loss
            - trades: List of FillEvents
            - portfolio: Portfolio instance
        """
        # 1. Load data
        df = self.data_handler.get_bars(symbol, start, end)
        
        # 2. Initialize portfolio and strategy
        portfolio = Portfolio(self.initial_capital)
        strategy = self.strategy_class(
            name=self.strategy_class.__name__,
            portfolio=portfolio,
            data_handler=self.data_handler,
            symbols=[symbol],
            target_weights={symbol: 1.0}
        )
        
        trades = []
        portfolio_values = []  # Track equity curve
        current_prices: Dict[str, float] = {}
        
        # 3. Event loop - process each bar in order
        for idx, row in df.iterrows():
            # Create bar event from DataFrame row
            bar = BarEvent(
                timestamp=idx,
                symbol=symbol,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['volume'])
            )
            
            # Update current prices for portfolio valuation
            current_prices = {symbol: bar.close}
            
            # Track portfolio value for equity curve
            portfolio_values.append(portfolio.get_total_value(current_prices))
            
            # Get trading signal from strategy
            order = strategy.on_bar(bar)
            
            if order is not None:
                # Execute order at current bar's close price
                fill = self.execution.execute(order, bar.close)
                
                # Update portfolio with fill
                portfolio.process_fill(fill)
                
                # Notify strategy of fill
                strategy.on_fill(fill)
                
                # Track trade
                trades.append(fill)
        
        # 4. Calculate final results
        final_value = portfolio.get_total_value(current_prices)
        unrealized_pnl = portfolio.get_unrealized_pnl(current_prices)
        
        # 5. Return results
        return {
            'strategy': strategy.name,
            'symbol': symbol,
            'start': start,
            'end': end,
            'initial_cash': self.initial_capital,
            'final_value': final_value,
            'realized_pnl': portfolio.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'trades': trades,
            'portfolio': portfolio,
            'portfolio_values': portfolio_values,
        }


class MultiSymbolEngine:
    """Multi-symbol backtest engine.
    
    Features:
    - Handles symbols with different start/end dates
    - Synchronized event processing across symbols
    - Graceful handling of halted/missing data
    """
    
    def __init__(
        self,
        strategy_class,
        symbols: List[str],
        initial_capital: float = 1000000.0,
        data_handler: Optional[MultiSymbolDataHandler] = None,
        execution: Optional[Execution] = None,
    ):
        """Initialize multi-symbol backtest engine.
        
        Args:
            strategy_class: Strategy class to use for signal generation.
            symbols: List of ticker symbols to backtest.
            initial_capital: Starting capital for the backtest.
            data_handler: Data handler for fetching market data.
            execution: Execution handler for simulating trades.
        """
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.data_handler = data_handler or MultiSymbolDataHandler()
        self.execution = execution or Execution()
    
    def run(
        self,
        start: str,
        end: str,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Run multi-symbol backtest.
        
        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            target_weights: Optional dict of symbol -> weight.
            
        Returns:
            dict with portfolio stats and trade history.
        """
        import pandas as pd
        
        # Load data for all symbols
        data = self.data_handler.get_bars(self.symbols, start, end)
        
        # Get common trading dates (where all symbols have data)
        common_dates = self.data_handler.get_common_dates(data)
        
        # Initialize default weights if not provided
        if target_weights is None:
            target_weights = {s: 1.0/len(self.symbols) for s in self.symbols}
        
        # Initialize portfolio and strategy
        portfolio = Portfolio(self.initial_capital)
        strategy = self.strategy_class(
            name=self.strategy_class.__name__,
            portfolio=portfolio,
            data_handler=self.data_handler,
            symbols=self.symbols,
            target_weights=target_weights
        )
        
        trades = []
        portfolio_values = []
        current_prices: Dict[str, float] = {}
        
        # Process each common date
        for date in common_dates:
            # Get latest bar for each symbol
            bars = self.data_handler.get_latest_bars(data, date)
            
            if not bars:
                continue
            
            # Update current prices for portfolio valuation
            current_prices = {symbol: bar['Close'] for symbol, bar in bars.items()}
            
            # Track portfolio value for equity curve
            portfolio_values.append(portfolio.get_total_value(current_prices))
            
            # Get signals from strategy
            orders = strategy.on_bar_multi(bars, date)
            
            # Execute orders
            for order in orders:
                if order is not None:
                    symbol = order.symbol
                    if symbol in current_prices:
                        fill = self.execution.execute(order, current_prices[symbol])
                        portfolio.process_fill(fill)
                        strategy.on_fill(fill)
                        trades.append(fill)
        
        # Calculate final results
        final_value = portfolio.get_total_value(current_prices)
        unrealized_pnl = portfolio.get_unrealized_pnl(current_prices)
        
        return {
            'strategy': strategy.name,
            'symbols': self.symbols,
            'start': start,
            'end': end,
            'initial_cash': self.initial_capital,
            'final_value': final_value,
            'realized_pnl': portfolio.realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'trades': trades,
            'portfolio': portfolio,
            'portfolio_values': portfolio_values,
        }


class IntradayEngine:
    """Intraday backtest engine.
    
    Key differences from daily:
    - Bar timestamps include time component
    - Orders can be placed intraday
    - Fill prices depend on bar OHLC
    """
    
    def __init__(
        self,
        strategy_class: type,
        symbols: List[str],
        initial_capital: float = 1000000.0,
        interval: str = '5m',
        data_handler: Optional[IntradayDataHandler] = None,
        execution: Optional[Execution] = None,
    ):
        """Initialize intraday backtest engine.
        
        Args:
            strategy_class: Strategy class to use for signal generation.
            symbols: List of ticker symbols to backtest.
            initial_capital: Starting capital for the backtest.
            interval: Intraday interval (1m, 5m, 15m, 1h).
            data_handler: Intraday data handler for fetching market data.
            execution: Execution handler for simulating trades.
        """
        self.strategy_class = strategy_class
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.interval = interval
        self.data_handler = data_handler or IntradayDataHandler(interval)
        self.execution = execution or Execution()
    
    def run(
        self,
        start: str,
        end: str,
        target_weights: Optional[Dict[str, float]] = None,
    ) -> dict:
        """Run intraday backtest.
        
        Args:
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            target_weights: Optional dict of symbol -> weight.
            
        Returns:
            dict with portfolio stats and trade history.
        """
        import pandas as pd
        from quantterm.backtesting.portfolio import Portfolio
        
        # Load daily data for signals
        daily_handler = DataHandler()
        daily_data = daily_handler.get_bars(self.symbols[0], start, end)
        
        # Initialize
        portfolio = Portfolio(self.initial_capital)
        
        # For simplicity, process daily bars with intraday fill model
        # In production, you'd iterate through intraday bars
        strategy = self.strategy_class(
            name=self.strategy_class.__name__,
            portfolio=portfolio,
            data_handler=self.data_handler,
            symbols=self.symbols,
            target_weights=target_weights,
        )
        
        trades = []
        
        # Process each day
        for date in daily_data.index:
            # Get intraday bars for this day
            day_bars = self.data_handler.get_day_bars(
                self.symbols[0], 
                date.strftime('%Y-%m-%d'),
                self.interval
            )
            
            if day_bars.empty:
                continue
            
            # Process each intraday bar
            for idx, row in day_bars.iterrows():
                bar = BarEvent(
                    timestamp=idx,
                    symbol=self.symbols[0],
                    open=row['Open'],
                    high=row['High'],
                    low=row['Low'],
                    close=row['Close'],
                    volume=int(row['Volume'])
                )
                
                # Get signal from strategy
                order = strategy.on_bar(bar)
                
                if order is not None:
                    # Use intraday fill model
                    fill_price = IntradayFillModel.get_market_fill(
                        row, order.quantity
                    )
                    
                    fill = self.execution.execute(order, fill_price)
                    portfolio.process_fill(fill)
                    strategy.on_fill(fill)
                    trades.append(fill)
        
        return {
            'strategy': strategy.name,
            'symbol': self.symbols[0],
            'start': start,
            'end': end,
            'initial_cash': self.initial_capital,
            'final_value': portfolio.get_total_value(
                {self.symbols[0]: daily_data.iloc[-1]['Close']}
            ),
            'realized_pnl': portfolio.realized_pnl,
            'trades': trades,
            'portfolio': portfolio,
        }
