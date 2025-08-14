# Backtesting System

> **Professional-grade backtesting framework for algorithmic trading strategies**

A clean, efficient, and production-ready backtesting system designed around the **Signal → Order → Backtest** workflow. Built for developing and testing trading strategies with realistic market conditions.

## 📖 Overview

This backtesting framework provides a complete solution for testing algorithmic trading strategies with historical market data. The system emphasizes simplicity, efficiency, and realistic trading conditions including transaction costs, slippage, and proper order execution modeling.

### Key Features

- 🎯 **Clean Architecture**: Clear separation of concerns with Signal → Order → Backtest flow
- 📊 **Comprehensive Analytics**: 20+ performance metrics including Sharpe ratio, Sortino ratio, drawdown analysis
- 💰 **Realistic Costs**: Transaction fees, slippage, and market impact modeling
- 🚀 **High Performance**: Vectorized operations and efficient pandas-based calculations
- 🔧 **Extensible**: Easy to add new strategies and technical indicators
- 📈 **Real Data Integration**: Built-in support for live market data through DataLoader

## 🏗️ Architecture

### System Flow

```
Market Data → Strategy → Signals → Orders → Execution → Portfolio → Performance
     ↑           ↑         ↑        ↑         ↑          ↑           ↑
DataLoader → strategies.py → SignalType → executors.py → models.py → performance.py
```

### Component Interaction

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loader   │    │   Strategies    │    │   Executors     │
│                 │    │                 │    │                 │
│ • Market Data   │───-│ • Signal Gen    │───-│ • Order Creation│
│ • Time Ranges   │    │ • Indicators    │    │ • Order Execut. │
│ • Symbol Info   │    │ • Entry/Exit    │    │ • Position Mgmt │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Engine      │    │     Models      │    │  Performance    │
│                 │    │                 │    │                 │
│ • Orchestration │◄───│ • Portfolio     │───-│ • Metrics Calc  │
│ • Main Loop     │    │ • Positions     │    │ • Risk Analysis │
│ • Result Gen    │    │ • Orders        │    │ • Report Gen    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 File Structure & Responsibilities

### Common Types (`types.py`)
**Primary Responsibility**: Centralized type definitions

```python
class ActionType(Enum):     # Unified BUY/SELL/CLOSE actions
@dataclass
class Signal:               # Trading signal data structure
class TransactionCost:      # Fee and slippage configuration
```

**Key Features**:
- Eliminates code duplication across modules
- Provides consistent type definitions
- SOLID principle compliance

### Core Engine (`engine.py`)
**Primary Responsibility**: Orchestrates the entire backtesting process

```python
class BacktestEngine:
    def run_backtest(signals, data, capital, symbol) -> BacktestResult
```

**Key Functions**:
- Coordinates all components (strategies, executors, portfolio)
- Manages the main backtesting loop with logging
- Handles signal processing and order execution timing
- Generates comprehensive results with equity curves and trade logs
- Calculates portfolio metrics at each time step

### Strategy Framework (`strategies.py`)
**Primary Responsibility**: Signal generation and technical analysis

```python
class Strategy(ABC):                    # Base strategy interface
    def generate_signal() -> Signal     # Single signal generation
    def generate_all_signals() -> DataFrame  # Batch processing

class GoldenCrossStrategy(Strategy):    # Example implementation
class SignalCombiner:                   # Multi-strategy aggregation

# Technical Indicators
calculate_moving_average()
calculate_ema()
calculate_rsi()
detect_crossover()
```

**Key Features**:
- Abstract Strategy base class for extensibility
- Built-in Golden Cross strategy with Fibonacci exits
- Batch signal processing for efficiency
- Comprehensive technical indicator library
- Signal strength and metadata support

### Order Management (`executors.py`)
**Primary Responsibility**: Convert signals to orders and execute them

```python
class OrderExecutor:
    def create_order_from_signal_data() -> Order
    def execute_order() -> (success, position)
    def validate_order() -> bool
```

**Key Functions**:
- Converts trading signals into executable orders
- Handles position sizing and risk management
- Manages order validation and execution logic
- Calculates transaction costs and slippage
- Updates portfolio state after execution

### Data Models (`models.py`)
**Primary Responsibility**: Core data structures and business logic

```python
# Trading Models
class Order:            # Individual trade orders
class Position:         # Open/closed positions
class Portfolio:        # Multi-position portfolio management

# Result Models
class BacktestResult:   # Complete backtest output
```

**Key Features**:
- Clean separation from type definitions
- Comprehensive position and portfolio tracking
- Built-in P&L calculations and risk metrics
- Uses unified ActionType from types.py

### Performance Analytics (`performance.py`)
**Primary Responsibility**: Calculate trading performance metrics

```python
class PerformanceAnalyzer:
    def analyze_performance() -> PerformanceMetrics
    def calculate_sharpe_ratio()
    def calculate_max_drawdown()
    def generate_report()

@dataclass
class PerformanceMetrics:
    # 20+ comprehensive metrics
```

**Calculated Metrics**:
- **Return Metrics**: Total, annualized returns
- **Risk Metrics**: Sharpe, Sortino, max drawdown, Calmar ratio
- **Trade Statistics**: Win rate, profit factor, average trade metrics
- **Distribution Analysis**: Skewness, kurtosis, information ratio
- **Exposure Analysis**: Market exposure time, recovery factors

### Logging System (`logger.py`)
**Primary Responsibility**: Centralized logging for all modules

```python
class BacktestLogger:           # Singleton logger manager
def get_logger(name) -> Logger  # Get module-specific logger
def set_log_level(level)        # Global log level control
def enable_file_logging()       # Enable file output
```

**Key Features**:
- Consistent logging format across all modules
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Console and file logging support
- Module-specific logger instances

### Usage Examples (`main.py`)
**Primary Responsibility**: Demonstrate real-world usage

```python
def run_simple_backtest():
    # Real data loading via DataLoader
    exchange = Exchange(id="binance")
    base_path = Path("../../fetch")
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    data = loader.load(symbol, timeframe, time_range)
    
    # Strategy setup and execution with logging
    strategy = GoldenCrossStrategy()
    engine = BacktestEngine()
    result = engine.run_backtest(signals, data, capital, symbol)
```

**Features**:
- Integration with real market data via DataLoader
- Simplified workflow demonstration
- Automatic logging of backtest progress
- Error handling and validation examples


## 🚀 Quick Start

### Basic Usage

```python
from bt.backtest import BacktestEngine, GoldenCrossStrategy, TransactionCost
from bt.core import Factory, DataLoader, Symbol, TimeFrame, TimeRange, DataType, Exchange
from decimal import Decimal
from pathlib import Path

# 1. Load real market data
exchange = Exchange(id="binance")
base_path = Path("../../fetch")
factory = Factory(exchange, base_path)
loader = DataLoader(factory)
symbol = Symbol.from_string("BTC/USDT")
data = loader.load(symbol, TimeFrame.D1, DataType.OHLCV, TimeRange.days(365))

# 2. Setup strategy
strategy = GoldenCrossStrategy()
signals = strategy.generate_all_signals(data)

# 3. Configure backtesting engine
engine = BacktestEngine(
    transaction_cost=TransactionCost(
        maker_fee=Decimal("0.001"),  # 0.1%
        taker_fee=Decimal("0.001"),  # 0.1%
        slippage=Decimal("0.0005")   # 0.05%
    )
)

# 4. Run backtest
result = engine.run_backtest(
    signals=signals,
    ohlcv_data=data,
    initial_capital=Decimal("10000"),
    symbol="BTCUSDT"
)

# 5. Analyze results
print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
```

### Running Examples

```bash
# Simple backtest with real data
python -m bt.backtest.main
```

### Logging Configuration

```python
from backtest.logger import get_logger, set_log_level, enable_file_logging

# Basic usage
logger = get_logger(__name__)
logger.info("Starting backtest...")

# Set global log level
set_log_level("DEBUG")  # DEBUG, INFO, WARNING, ERROR

# Enable file logging
enable_file_logging("my_backtest.log", level="DEBUG")
```

## 📊 Core Classes and Functions

### Signal Generation

```python
@dataclass(frozen=True)
class Signal:
    type: SignalType           # BUY, SELL, CLOSE, HOLD
    strength: float           # Signal confidence [0, 1]
    price: Optional[Decimal]  # Target price
    quantity: Optional[Decimal] # Position size
    stop_loss: Optional[Decimal]
    take_profit: Optional[Decimal]
    metadata: Dict[str, Any]  # Additional signal data
```

### Strategy Configuration

```python
strategy = GoldenCrossStrategy(
    ma_periods={
        "daily": [112, 224, 448],  # Golden cross MAs
        "3min": [360],             # 3-minute timeframe
        "30min": [60],             # 30-minute timeframe
        "60min": [60]              # 1-hour timeframe
    },
    position_sizing={
        "initial_size": 0.01,      # 1% of capital
        "scale_factor": 1.4,       # Scaling factor for entries
        "max_entries": 10          # Maximum position entries
    },
    exit_levels={
        "tp1_level": 0.382,        # First take-profit (38.2% Fib)
        "tp1_size": 0.5,           # Exit 50% at TP1
        "tp2_level": 0.5,          # Second take-profit (50% Fib)
        "tp2_size": 0.5            # Exit remaining 50%
    }
)
```

### Portfolio Management

```python
portfolio = Portfolio(initial_capital=Decimal("10000"))

# Portfolio state
portfolio.cash                    # Available cash
portfolio.total_value             # Total portfolio value
portfolio.open_positions          # List of open positions
portfolio.position_count          # Number of open positions

# Portfolio operations
portfolio.add_position(position)
portfolio.close_position(position, price, timestamp)
portfolio.calculate_metrics(current_prices)
```

## ⚙️ Configuration Options

### Transaction Costs

```python
transaction_cost = TransactionCost(
    maker_fee=Decimal("0.001"),    # 0.1% maker fee
    taker_fee=Decimal("0.001"),    # 0.1% taker fee
    slippage=Decimal("0.0005"),    # 0.05% slippage
    fixed_cost=Decimal("0")        # Fixed cost per trade
)
```

### Performance Analysis

```python
analyzer = PerformanceAnalyzer(risk_free_rate=0.02)  # 2% risk-free rate

# Generate comprehensive metrics
metrics = analyzer.analyze_performance(
    equity_curve=result.equity_curve,
    trades=result.trades,
    initial_capital=10000
)

# Generate formatted report
report = analyzer.generate_report(metrics)
print(report)
```

## 🔧 Extending the System

### Adding a New Strategy

```python
class MyCustomStrategy(Strategy):
    def __init__(self):
        super().__init__(
            name="MyCustomStrategy",
            parameters={"param1": value1, "param2": value2}
        )
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add your technical indicators here."""
        data = data.copy()
        data['my_indicator'] = calculate_my_indicator(data['close'])
        return data
    
    def generate_signal(
        self, 
        data: pd.DataFrame, 
        position=None, 
        portfolio=None
    ) -> Optional[Signal]:
        """Implement your signal logic here."""
        if self.should_buy(data):
            return Signal(
                type=SignalType.BUY,
                strength=0.8,
                price=Decimal(str(data['close'].iloc[-1]))
            )
        elif self.should_sell(data):
            return Signal(
                type=SignalType.SELL,
                strength=0.7,
                price=Decimal(str(data['close'].iloc[-1]))
            )
        return None
    
    def should_buy(self, data: pd.DataFrame) -> bool:
        """Your buy logic here."""
        return True  # Replace with actual logic
    
    def should_sell(self, data: pd.DataFrame) -> bool:
        """Your sell logic here."""
        return False  # Replace with actual logic
```

### Adding Technical Indicators

```python
def calculate_bollinger_bands(data: pd.Series, window: int = 20, 
                             num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return upper_band, sma, lower_band

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, 
                   signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD indicator."""
    ema_fast = data.ewm(span=fast).mean()
    ema_slow = data.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram
```

## 📈 Performance Metrics Reference

### Return Metrics
- **Total Return**: (Final Value - Initial Capital) / Initial Capital
- **Annualized Return**: Compound annual growth rate
- **Cumulative Return**: Total percentage gain/loss

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return metric
- **Sortino Ratio**: Downside risk-adjusted return
- **Information Ratio**: Excess return per unit of tracking error
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annualized return / Max Drawdown
- **Recovery Factor**: Net Profit / Max Drawdown

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Mean profit/loss per trade
- **Largest Win/Loss**: Single best/worst trade
- **Total Trades**: Number of completed trades
- **Average Trade Duration**: Mean time in trades

### Distribution Analysis
- **Skewness**: Asymmetry of return distribution
- **Kurtosis**: Tail heaviness of return distribution
- **Exposure Time**: Percentage of time with open positions

## 🎯 Best Practices

### Strategy Development
1. **Start Simple**: Begin with basic signal logic and iterate
2. **Test Thoroughly**: Use different time periods and market conditions
3. **Consider Costs**: Always include realistic transaction costs
4. **Validate Signals**: Implement proper signal validation logic
5. **Monitor Performance**: Track key metrics during development

### Risk Management
1. **Position Sizing**: Implement proper position sizing rules
2. **Stop Losses**: Include stop-loss logic in strategies
3. **Diversification**: Test across multiple assets and timeframes
4. **Drawdown Control**: Monitor and limit maximum drawdown
5. **Parameter Sensitivity**: Test strategy robustness to parameter changes

### Performance Analysis
1. **Multiple Metrics**: Don't rely on a single performance metric
2. **Walk-Forward Testing**: Test strategies on out-of-sample data
3. **Benchmark Comparison**: Compare against buy-and-hold returns
4. **Transaction Cost Impact**: Analyze cost sensitivity
5. **Market Regime Analysis**: Test across different market conditions

## 🔍 Troubleshooting

### Common Issues

**No Signals Generated**
- Check data length (need sufficient bars for indicators)
- Verify strategy parameters
- Ensure proper data format (OHLCV columns)

**Poor Performance**
- Review transaction cost settings
- Check signal quality and frequency
- Analyze trade timing and execution
- Validate indicator calculations

**Memory Issues**
- Use batch processing for large datasets
- Optimize indicator calculations
- Consider data sampling for initial testing

**Import Errors**
- Verify all dependencies are installed
- Check Python path configuration
- Ensure proper module structure

## 📚 Further Reading

- **Signal Processing**: Advanced signal filtering and combination techniques
- **Risk Management**: Portfolio-level risk controls and position sizing
- **Performance Attribution**: Understanding sources of returns
- **Market Microstructure**: Order book dynamics and execution modeling
- **Alternative Data**: Incorporating non-price data sources

---

*This backtesting system is designed for educational and research purposes. Past performance does not guarantee future results. Always validate strategies with paper trading before live deployment.*