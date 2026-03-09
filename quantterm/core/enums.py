"""Core enumerations for QuantTerm."""
from enum import Enum, auto


class AssetClass(str, Enum):
    """Asset class categories."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTO = "crypto"
    DERIVATIVE = "derivative"
    OPTIONS = "options"
    FUTURES = "futures"
    SWAP = "swap"


class Exchange(str, Enum):
    """Major exchange identifiers."""
    # US
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    CBOE = "CBOE"
    CME = "CME"
    ICE = "ICE"

    # International
    LSE = "LSE"
    Euronext = "Euronext"
    TSE = "TSE"
    HKEX = "HKEX"
    SGX = "SGX"

    # Crypto
    COINBASE = "Coinbase"
    BINANCE = "Binance"
    KRAKEN = "Kraken"


class OptionType(str, Enum):
    """Option type (call/put)."""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """Option exercise style."""
    EUROPEAN = "european"
    AMERICAN = "american"
    BERMDAN = "bermudan"


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    MARKET_ON_CLOSE = "market_on_close"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class OrderSide(str, Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"


class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeFrame(str, Enum):
    """Chart timeframes."""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1mo"


class AdjustmentType(str, Enum):
    """Corporate action adjustment types."""
    SPLIT = "split"
    DIVIDEND = "dividend"
    SPINOFF = "spinoff"
    MERGER = "merger"
    RIGHTS = "rights"
    BONUS = "bonus"


class VaRMethod(str, Enum):
    """Value at Risk calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"


class OptionPricingModel(str, Enum):
    """Option pricing models."""
    BLACK_SCHOLES = "black_scholes"
    BINOMIAL = "binomial"
    TRINOMIAL = "trinomial"
    MONTE_CARLO = "monte_carlo"
    HESTON = "heston"
    LOCAL_VOL = "local_vol"
    SABR = "sabr"


class TrendDirection(str, Enum):
    """Trend direction classification."""
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    SIDEWAYS = "sideways"
    UNKNOWN = "unknown"


class SignalType(str, Enum):
    """Trading signal types."""
    ENTRY_LONG = "entry_long"
    EXIT_LONG = "exit_long"
    ENTRY_SHORT = "entry_short"
    EXIT_SHORT = "exit_short"
    NEUTRAL = "neutral"


class Greeks(str, Enum):
    """Option Greeks."""
    DELTA = "delta"
    GAMMA = "gamma"
    THETA = "theta"
    VEGA = "vega"
    RHO = "rho"
    # Second order
    VANNA = "vanna"
    CHARM = "charm"
    VOMMA = "vomma"
    VETA = "veta"
    VERA = "vera"
    # Third order
    SPEED = "speed"
    ZOMMA = "zomma"
    COLOR = "color"
    ULTIMA = "ultima"


class DataProvider(str, Enum):
    """Data provider identifiers."""
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    POLYGON = "polygon"
    ALPACA = "alpaca"
    YAHOO = "yahoo"
    FRED = "fred"
    CUSTOM = "custom"


class Frequency(str, Enum):
    """Data frequency for calculations."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class RollingWindow(str, Enum):
    """Standard rolling window sizes."""
    TICK = "tick"
    VOLATILITY_5D = "5d"
    VOLATILITY_10D = "10d"
    VOLATILITY_20D = "20d"
    VOLATILITY_60D = "60d"
    VOLATILITY_90D = "90d"
    VOLATILITY_252D = "252d"
