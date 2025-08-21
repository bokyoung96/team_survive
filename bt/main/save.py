from pathlib import Path
from datetime import datetime
import pandas as pd

from backtest.storage import TradeStorage
from backtest.performance import PerformanceAnalyzer


def load_trades(session_id: str, fallback_df: pd.DataFrame = None) -> pd.DataFrame:
    try:
        storage = TradeStorage(base_dir="bt_results")
        trades = storage.load_trades(session_id)
        if not trades:
            return fallback_df or pd.DataFrame()
        
        data = []
        for i in range(0, len(trades), 2):
            if i + 1 < len(trades):
                entry, exit = trades[i], trades[i + 1]
                data.append({
                    'entry_time': pd.Timestamp(entry.timestamp, unit='ms'),
                    'exit_time': pd.Timestamp(exit.timestamp, unit='ms'),
                    'entry_price': entry.price,
                    'exit_price': exit.price,
                    'quantity': entry.quantity,
                    'pnl': exit.pnl,
                    'side': entry.side
                })
        
        df = pd.DataFrame(data) if data else (fallback_df or pd.DataFrame())
        print(f"Loaded {len(df)} trades from storage")
        return df
    except Exception as e:
        print(f"Storage load error: {e}")
        return fallback_df or pd.DataFrame()


def get_periods_per_year(equity_curve: pd.DataFrame) -> int:
    if len(equity_curve) < 2:
        return 365
    
    seconds = (equity_curve.index[1] - equity_curve.index[0]).total_seconds()
    
    if seconds >= 86400:
        return 365
    elif seconds >= 3600:
        return 365 * 24
    else:
        periods_per_day = 86400 / seconds
        return int(365 * periods_per_day)


def generate_report(result, session_id: str, **context) -> None:
    all_trades = load_trades(session_id, result.trades)
    
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_performance(
        equity_curve=result.equity_curve,
        trades=all_trades,
        initial_capital=float(result.portfolio.initial_capital),
        periods_per_year=get_periods_per_year(result.equity_curve)
    )
    
    print(analyzer.generate_report(metrics))
    
    report_dir = Path("bt_results") / session_id
    report_dir.mkdir(exist_ok=True, parents=True)
    
    report = f"""# Backtest Report

## Session Info
- **ID**: {session_id}
- **Strategy**: {context.get('strategy_name', 'N/A')}
- **Symbol**: {context.get('symbol_str', 'N/A')}
- **Period**: {context.get('start', 'N/A')} to {context.get('end', 'N/A')}
- **Initial**: ${float(result.portfolio.initial_capital):,.2f}
- **Final**: ${float(result.portfolio.total_value):,.2f}
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Performance
- **Return**: {metrics.total_return:.2%}
- **Annual**: {metrics.annualized_return:.2%}
- **Sharpe**: {metrics.sharpe_ratio:.2f}
- **Max DD**: {metrics.max_drawdown:.2%}
- **Win Rate**: {metrics.win_rate:.2%}
- **Trades**: {metrics.total_trades:,}

## Full Metrics
```
{analyzer.generate_report(metrics)}
```

## Data
- Storage trades: {len(all_trades):,}
- Memory trades: {len(result.trades) if hasattr(result, 'trades') and not result.trades.empty else 0:,}
"""
    
    # NOTE: Save files
    (report_dir / "report.md").write_text(report, encoding='utf-8')
    print(f"\nSaved: {report_dir}/report.md")