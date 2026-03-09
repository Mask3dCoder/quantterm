"""Execution simulator for the QuantTerm backtesting engine.

Enhanced execution handler with configurable slippage and commission models:
- Slippage models: Fixed bps, Volume-based
- Commission models: Per-share, Percentage of trade value
- Minimum order size handling
"""

from abc import ABC, abstractmethod
from typing import Optional

from quantterm.backtesting.events import OrderEvent, FillEvent


# ============================================================================
# Slippage Models
# ============================================================================

class SlippageModel(ABC):
    """Base class for slippage models."""
    
    @abstractmethod
    def get_price(self, order_price: float, quantity: int, volume: int = 1000000) -> float:
        """Calculate execution price with slippage.
        
        Args:
            order_price: Original order price.
            quantity: Order quantity (positive for BUY, negative for SELL).
            volume: Average daily volume for volume-based slippage.
            
        Returns:
            Execution price after slippage.
        """
        raise NotImplementedError


class FixedSlippage(SlippageModel):
    """Fixed slippage in basis points.
    
    Simple model that applies a constant slippage regardless of order size.
    """
    
    def __init__(self, bps: float = 5.0):
        """Initialize fixed slippage model.
        
        Args:
            bps: Slippage in basis points (default 5.0 = 0.05%).
        """
        self.bps = bps
    
    def get_price(self, order_price: float, quantity: int, volume: int = 1000000) -> float:
        """Apply fixed slippage to order price.
        
        Args:
            order_price: Original order price.
            quantity: Order quantity (positive for BUY, negative for SELL).
            volume: Ignored for fixed slippage.
            
        Returns:
            Execution price after slippage.
        """
        slippage = order_price * (self.bps / 10000)
        if quantity > 0:  # BUY: price goes up
            return order_price + slippage
        else:  # SELL: price goes down
            return order_price - slippage


class VolumeBasedSlippage(SlippageModel):
    """Slippage based on order size relative to volume.
    
    More realistic model where larger orders experience more slippage
    due to market impact.
    """
    
    def __init__(self, base_bps: float = 5.0, volume_participation: float = 0.1):
        """Initialize volume-based slippage model.
        
        Args:
            base_bps: Base slippage in bps (default 5.0 = 0.05%).
            volume_participation: Target participation rate (default 10%).
        """
        self.base_bps = base_bps
        self.volume_participation = volume_participation
    
    def get_price(self, order_price: float, quantity: int, volume: int = 1000000) -> float:
        """Apply volume-based slippage to order price.
        
        Args:
            order_price: Original order price.
            quantity: Order quantity (positive for BUY, negative for SELL).
            volume: Average daily volume for calculating participation.
            
        Returns:
            Execution price after slippage.
        """
        # More slippage for larger orders relative to volume
        order_value = abs(quantity * order_price)
        if volume > 0:
            participation = order_value / volume
            effective_bps = self.base_bps * (1 + participation / self.volume_participation)
        else:
            effective_bps = self.base_bps * 2  # Assume high impact if no volume data
        
        slippage = order_price * (effective_bps / 10000)
        if quantity > 0:  # BUY: price goes up
            return order_price + slippage
        else:  # SELL: price goes down
            return order_price - slippage


# ============================================================================
# Commission Models
# ============================================================================

class CommissionModel(ABC):
    """Base class for commission models."""
    
    @abstractmethod
    def calculate(self, quantity: int, price: float) -> float:
        """Calculate commission for a trade.
        
        Args:
            quantity: Number of shares traded.
            price: Execution price per share.
            
        Returns:
            Commission amount.
        """
        raise NotImplementedError


class PerShareCommission(CommissionModel):
    """Per-share commission.
    
    Common for many brokerages that charge a fixed rate per share.
    """
    
    def __init__(self, per_share: float = 0.01):
        """Initialize per-share commission model.
        
        Args:
            per_share: Commission per share (default $0.01).
        """
        self.per_share = per_share
    
    def calculate(self, quantity: int, price: float) -> float:
        """Calculate per-share commission.
        
        Args:
            quantity: Number of shares traded.
            price: Execution price per share (unused for per-share model).
            
        Returns:
            Total commission.
        """
        return abs(quantity) * self.per_share


class PercentageCommission(CommissionModel):
    """Percentage of trade value.
    
    Common for brokers that charge a percentage of trade value.
    """
    
    def __init__(self, percentage: float = 0.001):
        """Initialize percentage commission model.
        
        Args:
            percentage: Commission as decimal (default 0.001 = 0.1%).
        """
        self.percentage = percentage
    
    def calculate(self, quantity: int, price: float) -> float:
        """Calculate percentage-based commission.
        
        Args:
            quantity: Number of shares traded.
            price: Execution price per share.
            
        Returns:
            Total commission.
        """
        return abs(quantity) * price * self.percentage


# ============================================================================
# Execution Handler
# ============================================================================

class Execution:
    """Execution simulator for market orders.
    
    Executes orders at current price with configurable slippage and commission.
    Uses FixedSlippage and PerShareCommission by default for backward compatibility.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        commission_model: Optional[CommissionModel] = None,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.01,
        min_order_size: int = 1
    ):
        """Initialize execution handler.
        
        Args:
            slippage_model: Custom slippage model (default: FixedSlippage).
            commission_model: Custom commission model (default: PerShareCommission).
            slippage_bps: Slippage in bps (used if slippage_model not provided).
            commission_per_share: Commission per share (used if commission_model not provided).
            min_order_size: Minimum order size to execute (default: 1).
        """
        self.slippage_model = slippage_model or FixedSlippage(bps=slippage_bps)
        self.commission_model = commission_model or PerShareCommission(per_share=commission_per_share)
        self.min_order_size = min_order_size
    
    def execute(
        self,
        order: OrderEvent,
        current_price: float,
        volume: int = 1000000
    ) -> Optional[FillEvent]:
        """Execute order at current price with slippage and commission.
        
        Args:
            order: The order event to execute.
            current_price: Current market price for the symbol.
            volume: Average daily volume for volume-based slippage.
            
        Returns:
            FillEvent representing the executed trade, or None if order too small.
        """
        # Check minimum order size
        if abs(order.quantity) < self.min_order_size:
            return None
        
        # Apply slippage
        fill_price = self.slippage_model.get_price(current_price, order.quantity, volume)
        
        # Calculate commission
        commission = self.commission_model.calculate(order.quantity, fill_price)
        
        # Create fill event
        fill = FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            price=fill_price,
            commission=commission
        )
        
        return fill
