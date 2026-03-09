"""QuantTerm Backtesting Framework.

A minimal event-driven backtesting engine.
"""

from quantterm.backtesting.events import (
    BarEvent,
    OrderEvent,
    FillEvent,
    EventQueue,
)
from quantterm.backtesting.data_handler import DataHandler
from quantterm.backtesting.execution import Execution
from quantterm.backtesting.portfolio import Portfolio
from quantterm.backtesting.engine import BacktestEngine
from quantterm.backtesting.strategy.base import Strategy, BuyAndHoldStrategy

__all__ = [
    "BarEvent",
    "OrderEvent",
    "FillEvent",
    "EventQueue",
    "DataHandler",
    "Execution",
    "Portfolio",
    "BacktestEngine",
    "Strategy",
    "BuyAndHoldStrategy",
]
