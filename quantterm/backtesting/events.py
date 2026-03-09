"""Event types for the QuantTerm backtesting engine.

Defines the core event types used in the event-driven backtesting system:
- BarEvent: OHLCV bar data
- OrderEvent: Trading orders
- FillEvent: Executed trades
- EventQueue: Simple FIFO queue for events
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List


@dataclass
class BarEvent:
    """OHLCV bar event representing a single time period of market data.

    Attributes:
        timestamp: The datetime of the bar.
        symbol: The ticker symbol (e.g., 'SPY').
        open: Opening price.
        high: Highest price in the period.
        low: Lowest price in the period.
        close: Closing price.
        volume: Trading volume.
    """
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class OrderEvent:
    """Order event representing a trading order.

    Attributes:
        timestamp: The datetime of the order.
        symbol: The ticker symbol.
        quantity: Positive for BUY, negative for SELL.
        order_type: Type of order (only "market" for now).
    """
    timestamp: datetime
    symbol: str
    quantity: int
    order_type: str = "market"


@dataclass
class FillEvent:
    """Fill event representing an executed trade.

    Attributes:
        timestamp: The datetime of the fill.
        symbol: The ticker symbol.
        quantity: Number of shares filled (positive for BUY, negative for SELL).
        price: Execution price.
        commission: Commission paid for the trade.
    """
    timestamp: datetime
    symbol: str
    quantity: int
    price: float
    commission: float = 0.0


# Type alias for any event type
Event = BarEvent | OrderEvent | FillEvent


class EventQueue:
    """Simple FIFO queue for events.

    A basic priority queue that maintains insertion order.
    Thread-unsafe, intended for synchronous use only.
    """

    def __init__(self) -> None:
        """Initialize an empty event queue."""
        self._queue: List[Event] = []

    def put(self, event: Event) -> None:
        """Add an event to the queue.

        Args:
            event: The event to add.
        """
        self._queue.append(event)

    def get(self) -> Optional[Event]:
        """Remove and return the next event from the queue.

        Returns:
            The next event, or None if the queue is empty.
        """
        if self._queue:
            return self._queue.pop(0)
        return None

    def empty(self) -> bool:
        """Check if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise.
        """
        return len(self._queue) == 0

    def __len__(self) -> int:
        """Return the number of events in the queue."""
        return len(self._queue)
