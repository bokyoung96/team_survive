from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime
import warnings

from backtest.models import BacktestResult

warnings.filterwarnings('ignore')

"""
Color scheme for professional-looking plots
- 'strategy': Red for strategy
- 'benchmark': Black/dark gray for benchmark
- 'profit': Green for profits
- 'loss': Dark red for losses
- 'drawdown': Red for drawdown
- 'neutral': Gray for neutral elements
"""
COLORS = {
    'strategy': '#E74C3C',
    'benchmark': '#2C3E50',
    'profit': '#27AE60',
    'loss': '#C0392B',
    'drawdown': '#E74C3C',
    'neutral': '#95A5A6',
}


def create_backtest_report(
    result: BacktestResult,
    benchmark_data: Optional[pd.DataFrame] = None,
    strategy_name: str = 'Strategy',
    output_dir: str = "bt_results",
    show_plots: bool = False,
    session_id: Optional[str] = None
) -> Dict[str, plt.Figure]:

    # NOTE: Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    if session_id is None:
        session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # NOTE: Create figure dictionary
    figures = {}
    
    # NOTE: Create performance plot
    try:
        fig_perf = create_performance_plot(
            result.equity_curve, 
            benchmark_data, 
            strategy_name=strategy_name
        )
        if fig_perf:
            figures['performance'] = fig_perf
            date_suffix = session_id.split('_', 1)[1] if '_' in session_id else session_id
            fig_perf.savefig(output_path / f"performance_{date_suffix}.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error creating performance plot: {e}")
    
    # NOTE: Create trades plot
    try:
        fig_trades = create_trades_plot(
            result.equity_curve,
            result.trades,
            strategy_name=strategy_name
        )
        if fig_trades:
            figures['trades'] = fig_trades
            date_suffix = session_id.split('_', 1)[1] if '_' in session_id else session_id
            fig_trades.savefig(output_path / f"trades_{date_suffix}.png", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"Error creating trades plot: {e}")
    
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    return figures


def create_performance_plot(
    equity_curve: pd.DataFrame, 
    benchmark: Optional[pd.DataFrame] = None,
    strategy_name: str = 'Strategy'
) -> Optional[plt.Figure]:
    if equity_curve is None or equity_curve.empty:
        return None
    
    equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).dropna(how='all')
    if equity_curve.empty:
        return None
    
    equity_col = None
    for col in ['total_value', 'equity', 'portfolio_value']:
        if col in equity_curve.columns:
            equity_col = col
            break
    
    if not equity_col:
        return None
    
    equity_data = (equity_curve[equity_col]
                   .replace([np.inf, -np.inf], np.nan)
                   .fillna(method='ffill')
                   .fillna(method='bfill'))
    
    if equity_data.empty or equity_data.isna().all():
        return None
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), height_ratios=[1, 1, 0.7])
    
    initial_equity = equity_data.iloc[0]
    strategy_cumulative_returns = (equity_data / initial_equity - 1) * 100
    
    # NOTE: Subplot 1: Cumulative Returns Comparison
    ax1.plot(equity_curve.index, strategy_cumulative_returns, 
             label=strategy_name, color=COLORS['strategy'], linewidth=2)
    
    if benchmark is not None and 'close' in benchmark.columns:
        try:
            benchmark_close = (benchmark['close']
                              .replace([np.inf, -np.inf], np.nan)
                              .fillna(method='ffill'))
            if not benchmark_close.empty:
                benchmark_returns = (benchmark_close / benchmark_close.iloc[0] - 1) * 100
                ax1.plot(benchmark.index, benchmark_returns, 
                        label='Benchmark (Buy & Hold)', 
                        color=COLORS['benchmark'], 
                        linewidth=1.5, alpha=0.8)
        except:
            pass
    
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Cumulative Returns (%)', fontsize=11)
    ax1.set_title(f'{strategy_name} vs Benchmark - Cumulative Returns', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    final_return = strategy_cumulative_returns.iloc[-1]
    stats_text = f"Total Return: {final_return:.2f}%\nFinal Multiple: {(final_return/100 + 1):.2f}x"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # NOTE: Subplot 2: Strategy-Only Cumulative Returns
    ax2.plot(equity_curve.index, strategy_cumulative_returns, 
             label=f'{strategy_name} Returns', 
             color=COLORS['strategy'], linewidth=2)
    
    ax2.fill_between(equity_curve.index, 0, strategy_cumulative_returns, 
                     where=(strategy_cumulative_returns >= 0), 
                     color=COLORS['profit'], alpha=0.1, interpolate=True)
    ax2.fill_between(equity_curve.index, 0, strategy_cumulative_returns, 
                     where=(strategy_cumulative_returns < 0), 
                     color=COLORS['loss'], alpha=0.1, interpolate=True)
    
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax2.set_ylabel('Strategy Returns (%)', fontsize=11)
    ax2.set_title(f'{strategy_name} - Detailed Returns', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    if not strategy_cumulative_returns.empty:
        max_return = strategy_cumulative_returns.max()
        min_return = strategy_cumulative_returns.min()
        max_idx = strategy_cumulative_returns.idxmax()
        min_idx = strategy_cumulative_returns.idxmin()
        
        ax2.scatter([max_idx], [max_return], color=COLORS['profit'], s=50, zorder=5)
        ax2.scatter([min_idx], [min_return], color=COLORS['loss'], s=50, zorder=5)
        
        ax2.annotate(f'Peak: {max_return:.1f}%', 
                    xy=(max_idx, max_return),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, color=COLORS['profit'])
        ax2.annotate(f'Trough: {min_return:.1f}%', 
                    xy=(min_idx, min_return),
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=9, color=COLORS['loss'])
    
    # NOTE: Subplot 3: Portfolio Drawdown
    try:
        running_max = equity_data.expanding().max()
        drawdown = ((equity_data - running_max) / running_max * 100).fillna(0)
        
        ax3.fill_between(equity_curve.index, 0, drawdown, 
                        color=COLORS['drawdown'], alpha=0.3)
        ax3.plot(equity_curve.index, drawdown, 
                color=COLORS['drawdown'], linewidth=1.5)
        
        if not drawdown.empty and not drawdown.isna().all():
            max_dd = drawdown.min()
            if not np.isnan(max_dd) and max_dd < 0:
                max_dd_idx = drawdown.idxmin()
                ax3.scatter([max_dd_idx], [max_dd], 
                           color=COLORS['drawdown'], s=100, zorder=5)
                ax3.annotate(f'Max Drawdown: {max_dd:.2f}%', 
                           xy=(max_dd_idx, max_dd),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold')
    except:
        pass
    
    ax3.set_ylabel('Drawdown (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)
    ax3.set_title('Portfolio Drawdown', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(top=0)
    
    for ax in [ax1, ax2, ax3]:
        format_date_axis(ax, equity_curve.index)
    
    plt.suptitle(f'{strategy_name} - Performance Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    return fig


def create_trades_plot(
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame,
    strategy_name: str = 'Strategy'
) -> Optional[plt.Figure]:
    if trades_df is None or trades_df.empty:
        return None
    
    fig = plt.figure(figsize=(14, 10))
    
    ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2, rowspan=3)
    ax2 = plt.subplot2grid((5, 2), (3, 0), rowspan=2)
    ax3 = plt.subplot2grid((5, 2), (3, 1), rowspan=2)
    
    # NOTE: Top Plot: Equity Curve with Clean Trade Visualization
    if equity_curve is not None and not equity_curve.empty:
        equity_col = None
        for col in ['total_value', 'equity', 'portfolio_value']:
            if col in equity_curve.columns:
                equity_col = col
                break
        
        if equity_col:
            ax1.plot(equity_curve.index, equity_curve[equity_col], 
                    color=COLORS['strategy'], linewidth=2, 
                    label='Portfolio Value', zorder=2)
            
            try:
                for idx, trade in trades_df.iterrows():
                    if idx > 20:
                        break
                        
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    
                    color = COLORS['profit'] if trade.get('pnl', 0) > 0 else COLORS['loss']
                    ax1.axvspan(entry_time, exit_time, alpha=0.1, color=color, zorder=1)
            except:
                pass
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df.get('pnl', pd.Series([0])) > 0])
            losing_trades = total_trades - winning_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            stats_text = (f'Total Trades: {total_trades}\n'
                         f'Winners: {winning_trades}\n'
                         f'Losers: {losing_trades}\n'
                         f'Win Rate: {win_rate:.1f}%')
            
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
            ax1.set_title(f'{strategy_name} - Equity Curve with Trade Periods', 
                         fontsize=13, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            format_date_axis(ax1, equity_curve.index)
    
    # NOTE: Bottom Left: Trade Duration Distribution
    plot_trade_durations(ax2, trades_df)
    
    # NOTE: Bottom Right: Trade Returns Distribution
    plot_trade_returns(ax3, trades_df)
    
    plt.suptitle(f'{strategy_name} - Trade Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_trade_durations(ax, trades_df):
    try:
        if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
            entry_times = pd.to_datetime(trades_df['entry_time'])
            exit_times = pd.to_datetime(trades_df['exit_time'])
            
            durations = (exit_times - entry_times).dt.total_seconds() / 3600
            durations = durations[durations.notna() & (durations > 0)]
            
            if len(durations) > 0:
                n, bins, patches = ax.hist(durations, bins=min(30, len(durations)), 
                                          color=COLORS['neutral'], alpha=0.7, 
                                          edgecolor='black', linewidth=0.5)
                
                quartiles = np.percentile(durations, [25, 50, 75])
                for i, patch in enumerate(patches):
                    if bins[i] < quartiles[0]:
                        patch.set_facecolor(COLORS['profit'])
                        patch.set_alpha(0.5)
                    elif bins[i] > quartiles[2]:
                        patch.set_facecolor(COLORS['loss'])
                        patch.set_alpha(0.5)
                
                mean_duration = durations.mean()
                median_duration = durations.median()
                
                ax.axvline(mean_duration, color='red', linestyle='--', 
                          linewidth=1.5, label=f'Mean: {mean_duration:.1f}h')
                ax.axvline(median_duration, color='blue', linestyle='--', 
                          linewidth=1.5, label=f'Median: {median_duration:.1f}h')
                
                ax.set_xlabel('Trade Duration (hours)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title('Trade Duration Distribution', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                return
    except Exception as e:
        pass
    
    ax.text(0.5, 0.5, 'No duration data available', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Trade Duration Distribution', fontsize=11, fontweight='bold')


def plot_trade_returns(ax, trades_df):
    try:
        pnl_col = None
        for col in ['pnl', 'realized_pnl', 'profit_loss']:
            if col in trades_df.columns:
                pnl_col = col
                break
        
        if pnl_col:
            if 'entry_price' in trades_df.columns and 'quantity' in trades_df.columns:
                entry_values = trades_df['entry_price'] * trades_df['quantity']
                returns_pct = (trades_df[pnl_col] / entry_values * 100).replace(
                    [np.inf, -np.inf], np.nan).dropna()
            else:
                avg_position = trades_df[pnl_col].abs().mean()
                if avg_position > 0:
                    returns_pct = (trades_df[pnl_col] / avg_position * 100).replace(
                        [np.inf, -np.inf], np.nan).dropna()
                else:
                    returns_pct = pd.Series()
            
            if len(returns_pct) > 0:
                profits = returns_pct[returns_pct > 0]
                losses = returns_pct[returns_pct < 0]
                
                all_returns = returns_pct.values
                min_ret, max_ret = all_returns.min(), all_returns.max()
                bins = np.linspace(min_ret - 1, max_ret + 1, min(31, len(returns_pct)))
                
                if len(profits) > 0:
                    ax.hist(profits, bins=bins, color=COLORS['profit'], 
                           alpha=0.6, label=f'Profits ({len(profits)})', 
                           edgecolor='black', linewidth=0.5)
                if len(losses) > 0:
                    ax.hist(losses, bins=bins, color=COLORS['loss'], 
                           alpha=0.6, label=f'Losses ({len(losses)})', 
                           edgecolor='black', linewidth=0.5)
                
                mean_return = returns_pct.mean()
                median_return = returns_pct.median()
                
                ax.axvline(mean_return, color='blue', linestyle='--', 
                          linewidth=1.5, label=f'Mean: {mean_return:.2f}%')
                ax.axvline(0, color='black', linewidth=1, alpha=0.5)
                
                avg_win = profits.mean() if len(profits) > 0 else 0
                avg_loss = abs(losses.mean()) if len(losses) > 0 else 0
                profit_factor = avg_win / avg_loss if avg_loss > 0 else np.inf
                
                stats_text = (f'Avg Win: {avg_win:.2f}%\n'
                             f'Avg Loss: {avg_loss:.2f}%\n'
                             f'P/L Ratio: {profit_factor:.2f}')
                
                ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                
                ax.set_xlabel('Trade Returns (%)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.set_title('Trade Returns Distribution', fontsize=11, fontweight='bold')
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3, axis='y')
                
                return
    except Exception as e:
        pass
    
    ax.text(0.5, 0.5, 'No returns data available', 
           ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Trade Returns Distribution', fontsize=11, fontweight='bold')


def format_date_axis(ax, dates):
    try:
        date_range = (dates[-1] - dates[0]).days
        
        if date_range > 365:
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        elif date_range > 90:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        elif date_range > 30:
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    except:
        pass


