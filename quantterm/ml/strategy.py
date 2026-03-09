"""
ML-Based Trading Strategy.

Combines feature engineering → Model prediction → Position sizing
"""
from typing import Optional, List, Dict
import pandas as pd
from quantterm.backtesting.strategy.base import Strategy
from quantterm.backtesting.events import OrderEvent, BarEvent, FillEvent


class MLStrategy(Strategy):
    """
    Strategy driven by machine learning predictions.
    
    Features:
    - Signal generation from ML model
    - Confidence-based position sizing
    - Volatility-based risk management
    """
    
    def __init__(
        self,
        name: str = "MLStrategy",
        portfolio = None,
        data_handler = None,
        model_trainer: 'MLModelTrainer' = None,
        prediction_threshold: float = 0.6,
        confidence_filter: float = 0.1,
        max_position_pct: float = 0.25,
        symbols: Optional[List[str]] = None,
        target_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize ML strategy.
        
        Args:
            name: Strategy name
            portfolio: Portfolio instance
            data_handler: DataHandler instance
            model_trainer: Trained MLModelTrainer
            prediction_threshold: Probability threshold for signals
            confidence_filter: Minimum confidence to trade
            max_position_pct: Maximum position as % of portfolio
            symbols: List of symbols
            target_weights: Dict of symbol -> target weight
        """
        super().__init__(
            name=name,
            portfolio=portfolio,
            data_handler=data_handler,
            symbols=symbols or [],
            target_weights=target_weights or {}
        )
        self.model = model_trainer
        self.threshold = prediction_threshold
        self.confidence_filter = confidence_filter
        self.max_position_pct = max_position_pct
        
        # State
        self.last_features = None
        self.predictions = []
        self._feature_cache: Dict[str, pd.DataFrame] = {}
    
    def on_bar(self, bar: BarEvent) -> Optional[OrderEvent]:
        """
        Generate trading signal based on ML prediction.
        """
        # This needs feature computation from historical data
        # In practice, you'd maintain a rolling data buffer in data_handler
        # For now, return None (strategy needs data handler integration)
        return None
    
    def on_bar_with_features(
        self,
        bar: BarEvent,
        features: pd.Series
    ) -> Optional[OrderEvent]:
        """
        Generate order based on features.
        
        Args:
            bar: Current bar
            features: Pre-computed features
            
        Returns:
            OrderEvent or None
        """
        self.last_features = features
        
        if self.model is None or self.model.model is None:
            return None
        
        try:
            prob_up, confidence = self.model.predict(features)
        except Exception:
            return None
        
        # Log prediction
        self.predictions.append({
            'timestamp': bar.timestamp,
            'symbol': bar.symbol,
            'prob_up': prob_up,
            'confidence': confidence
        })
        
        # Check confidence threshold
        if confidence < self.confidence_filter:
            return None
        
        # Get portfolio value
        prices = {bar.symbol: bar.close}
        total_value = self._get_portfolio_value(prices)
        
        # Calculate position size
        if prob_up > self.threshold:
            # Long signal
            size_pct = min(confidence * self.max_position_pct * 2, self.max_position_pct)
            target_value = total_value * size_pct
            shares = int(target_value / bar.close)
            
            current = self.portfolio.get_position(bar.symbol)
            diff = shares - current
            
            if diff > 0:
                return OrderEvent(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    quantity=diff,
                    order_type="market"
                )
                
        elif prob_up < (1 - self.threshold):
            # Short signal
            size_pct = min(confidence * self.max_position_pct * 2, self.max_position_pct)
            target_value = total_value * size_pct
            shares = int(target_value / bar.close)
            
            current = self.portfolio.get_position(bar.symbol)
            diff = -shares - current
            
            if diff < 0:
                return OrderEvent(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    quantity=diff,
                    order_type="market"
                )
        
        return None
    
    def on_bar_multi(
        self, 
        bars: Dict[str, pd.Series], 
        date: pd.Timestamp
    ) -> List[Optional[OrderEvent]]:
        """Multi-symbol bar handler for ML strategy.
        
        Args:
            bars: Dict of symbol -> bar data
            date: Current date
            
        Returns:
            List of OrderEvents
        """
        orders = []
        for symbol, bar_data in bars.items():
            # Create a BarEvent from the series data
            bar = BarEvent(
                timestamp=date,
                symbol=symbol,
                open=bar_data.get('Open', bar_data.get('close', 0)),
                high=bar_data.get('High', bar_data.get('close', 0)),
                low=bar_data.get('Low', bar_data.get('close', 0)),
                close=bar_data.get('Close', bar_data.get('close', 0)),
                volume=int(bar_data.get('Volume', 0))
            )
            
            # Get cached features for this symbol
            if symbol in self._feature_cache:
                features = self._feature_cache[symbol].iloc[-1] if len(self._feature_cache[symbol]) > 0 else None
                if features is not None:
                    order = self.on_bar_with_features(bar, features)
                    if order:
                        orders.append(order)
        
        return orders
    
    def on_fill(self, fill: FillEvent):
        """Track fills."""
        pass
    
    def _get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Get total portfolio value.
        
        Args:
            prices: Dict of symbol -> current price
            
        Returns:
            Total portfolio value
        """
        if self.portfolio is None:
            return 0.0
        
        total = self.portfolio.cash
        
        # Add value of all positions
        for symbol, position in self.portfolio.positions.items():
            if position != 0:
                price = prices.get(symbol, 0)
                total += position * price
        
        return total
    
    def update_features(self, symbol: str, features_df: pd.DataFrame):
        """Update cached features for a symbol.
        
        Args:
            symbol: Symbol to update
            features_df: DataFrame of features
        """
        self._feature_cache[symbol] = features_df
    
    def get_predictions(self) -> pd.DataFrame:
        """Get prediction history.
        
        Returns:
            DataFrame of predictions
        """
        if not self.predictions:
            return pd.DataFrame()
        return pd.DataFrame(self.predictions)
