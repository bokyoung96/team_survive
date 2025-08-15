"""
Example Usage of Enhanced Backtesting System

This script demonstrates various ways to use the enhanced backtesting system
with report generation and visualization capabilities.
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.factory import Factory
from core.models import Symbol, TimeFrame, TimeRange, Exchange, MarketType
from core.loader import DataLoader
from utils import KST
from backtest.timeframe import MultiTimeframeData
from backtest.strats.dolpha1 import GoldenCrossStrategy
from backtest.types import TransactionCost
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.plot import BacktestVisualizer, PlotConfig


def example_basic_backtest():
    """Example 1: Basic backtest without comprehensive analysis."""
    print("=== Example 1: Basic Backtest ===")
    
    # Setup (simplified from main.py)
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    symbol = Symbol.from_string("BTC/USDT:USDT")
    start_date = KST.localize(datetime(2024, 11, 1))
    end_date = KST.localize(datetime(2024, 12, 1))
    date_range = TimeRange(start_date, end_date)
    
    # Load data
    data = (MultiTimeframeData(loader)
            .add(symbol, TimeFrame.D1, date_range)
            .add(symbol, TimeFrame.H1, date_range))
    
    # Run strategy
    strategy = GoldenCrossStrategy(data=data)
    signals = strategy.generate_all_signals()
    
    if signals.empty:
        print("No signals generated for this period")
        return None
    
    # Run backtest
    engine = BacktestEngine(transaction_cost=TransactionCost())
    result = engine.run_backtest(
        signals=signals,
        ohlcv_data=data["1d"],
        initial_capital=Decimal("10000"),
        symbol=f"{symbol.base}{symbol.quote}"
    )
    
    # Basic results
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Total Trades: {result.metrics['total_trades']}")
    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
    
    return result


def example_comprehensive_analysis(result):
    """Example 2: Comprehensive performance analysis."""
    if result is None:
        print("No result to analyze")
        return
    
    print("\n=== Example 2: Comprehensive Analysis ===")
    
    # Initialize performance analyzer
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Calculate detailed metrics
    metrics = analyzer.analyze_performance(
        equity_curve=result.equity_curve,
        trades=result.trades,
        initial_capital=float(result.portfolio.initial_capital)
    )
    
    # Generate report
    report = analyzer.generate_report(metrics)
    print(report)
    
    return metrics


def example_custom_visualizations(result, metrics):
    """Example 3: Custom visualization generation."""
    if result is None or metrics is None:
        print("No data to visualize")
        return
    
    print("\n=== Example 3: Custom Visualizations ===")
    
    # Configure custom plot settings
    config = PlotConfig(
        figure_size=(14, 10),
        dpi=150,  # Lower DPI for faster generation
        save_format='png',
        output_dir='custom_analysis',
        show_plots=False,
        color_scheme={
            'equity': '#1f77b4',
            'buy_signal': '#ff7f0e',
            'sell_signal': '#d62728',
            'win_trade': '#2ca02c',
            'loss_trade': '#d62728',
            'drawdown': '#ff1744'
        }
    )
    
    # Initialize visualizer
    visualizer = BacktestVisualizer(config)
    
    # Generate individual plots
    print("Generating equity curve...")
    equity_fig = visualizer.plot_equity_curve(result, save_file='custom_equity.png')
    
    print("Generating drawdown analysis...")
    drawdown_fig = visualizer.plot_drawdown_periods(result, save_file='custom_drawdown.png')
    
    print("Generating performance dashboard...")
    dashboard_fig = visualizer.plot_performance_metrics_dashboard(metrics, save_file='custom_dashboard.png')
    
    print("Custom visualizations complete!")
    
    # You can also display plots if running interactively
    # import matplotlib.pyplot as plt
    # plt.show()


def example_programmatic_usage():
    """Example 4: Programmatic usage for integration."""
    print("\n=== Example 4: Programmatic Integration ===")
    
    # Run basic backtest
    result = example_basic_backtest()
    if result is None:
        return
    
    # Analyze performance
    metrics = example_comprehensive_analysis(result)
    
    # Generate visualizations
    example_custom_visualizations(result, metrics)
    
    # Access specific metrics for decision making
    if metrics:
        print(f"\nKey Metrics for Decision Making:")
        print(f"- Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"- Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"- Win Rate: {metrics.win_rate:.2%}")
        print(f"- Profit Factor: {metrics.profit_factor:.2f}")
        
        # Example: Simple strategy evaluation
        if metrics.sharpe_ratio > 1.0 and metrics.max_drawdown < 0.15:
            print("✓ Strategy meets basic performance criteria")
        else:
            print("✗ Strategy needs improvement")


if __name__ == "__main__":
    print("Backtesting Enhanced System - Usage Examples")
    print("=" * 50)
    
    try:
        # Run all examples
        example_programmatic_usage()
        
        print("\n" + "=" * 50)
        print("Examples completed successfully!")
        print("\nFor command-line usage, try:")
        print("  python backtest/main.py --help")
        print("  python backtest/main.py --output-dir my_results")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nNote: Make sure you have market data available")
        print("Run the data fetching system first if needed")