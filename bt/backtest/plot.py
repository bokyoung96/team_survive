"""Compact plotting module for backtesting visualization."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

from backtest.models import BacktestResult

warnings.filterwarnings('ignore')


def create_backtest_report(
    result: BacktestResult,
    benchmark_data: Optional[pd.DataFrame] = None,
    strategy_name: str = 'Strategy',
    output_dir: str = "bt_results",
    show_plots: bool = False
) -> Dict[str, plt.Figure]:
    """Create compact backtest visualization report."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    figures = {}
    
    # Create performance plot
    try:
        fig_perf = create_performance_plot(
            result.equity_curve, 
            benchmark_data, 
            strategy_name=strategy_name
        )
        if fig_perf:
            figures['performance'] = fig_perf
            fig_perf.savefig(f"{output_dir}/performance.png", dpi=100, bbox_inches='tight')
    except Exception as e:
        print(f"Error creating performance plot: {e}")
    
    # Create trades plot
    try:
        fig_trades = create_trades_plot(
            result.equity_curve,
            result.trades,
            strategy_name=strategy_name
        )
        if fig_trades:
            figures['trades'] = fig_trades
            fig_trades.savefig(f"{output_dir}/trades.png", dpi=100, bbox_inches='tight')
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
    """Create performance visualization with equity curve and drawdown."""
    if equity_curve is None or equity_curve.empty:
        return None
    
    # Clean data
    equity_curve = equity_curve.replace([np.inf, -np.inf], np.nan).dropna(how='all')
    if equity_curve.empty:
        return None
    
    # Create 2x2 subplot layout
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Find equity column
    equity_col = None
    for col in ['total_value', 'equity', 'portfolio_value']:
        if col in equity_curve.columns:
            equity_col = col
            break
    
    if not equity_col:
        return None
    
    # Clean equity data
    equity_data = equity_curve[equity_col].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
    if equity_data.empty or equity_data.isna().all():
        return None
    
    # === Subplot 1: Absolute Equity Curve with Benchmark ===
    ax1.plot(equity_curve.index, equity_data, 
             label=strategy_name, color='#2E86AB', linewidth=2)
    
    # Plot benchmark if available
    if benchmark is not None and 'close' in benchmark.columns and len(benchmark) > 0:
        try:
            initial_value = equity_data.iloc[0]
            benchmark_close = benchmark['close'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
            if not benchmark_close.empty and not benchmark_close.isna().all():
                benchmark_norm = (benchmark_close / benchmark_close.iloc[0]) * initial_value
                ax1.plot(benchmark.index, benchmark_norm, 
                        label='Buy & Hold', color='#A23B72', linewidth=1, alpha=0.7)
        except:
            pass
    
    # Mark position periods
    if 'position_count' in equity_curve.columns:
        try:
            positions = equity_curve['position_count'].fillna(0)
            in_position = positions > 0
            position_changes = in_position.astype(int).diff()
            entries = equity_curve.index[position_changes == 1].tolist()
            exits = equity_curve.index[position_changes == -1].tolist()
            
            for i in range(len(entries)):
                exit_time = exits[i] if i < len(exits) else equity_curve.index[-1]
                ax1.axvspan(entries[i], exit_time, alpha=0.1, color='green')
        except:
            pass
    
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.set_title(f'{strategy_name} - Absolute Equity Curve')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === Subplot 2: Cumulative Returns (both start at 1) ===
    try:
        # Calculate cumulative returns for strategy
        initial_equity = equity_data.iloc[0]
        strategy_cumulative_returns = equity_data / initial_equity
        
        ax2.plot(equity_curve.index, strategy_cumulative_returns, 
                 label=strategy_name, color='#2E86AB', linewidth=2)
        
        # Calculate cumulative returns for benchmark
        if benchmark is not None and 'close' in benchmark.columns and len(benchmark) > 0:
            benchmark_close = benchmark['close'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
            if not benchmark_close.empty and not benchmark_close.isna().all():
                benchmark_cumulative_returns = benchmark_close / benchmark_close.iloc[0]
                ax2.plot(benchmark.index, benchmark_cumulative_returns, 
                        label='Buy & Hold', color='#A23B72', linewidth=1, alpha=0.7)
        
        # Add horizontal line at 1
        ax2.axhline(y=1, color='black', linewidth=0.5, linestyle='--', alpha=0.5)
        
        # Add performance stats
        final_return = strategy_cumulative_returns.iloc[-1]
        total_return_pct = (final_return - 1) * 100
        stats_text = f"Return: {total_return_pct:.1f}%\nMultiple: {final_return:.2f}x"
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    except:
        pass
    
    ax2.set_ylabel('Cumulative Returns (Starting at 1)')
    ax2.set_title('Cumulative Returns Comparison')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # === Subplot 3: Strategy Only (Zoomed) ===
    ax3.plot(equity_curve.index, equity_data, 
             label=strategy_name, color='#2E86AB', linewidth=2)
    
    # Mark position periods with different shading
    if 'position_count' in equity_curve.columns:
        try:
            positions = equity_curve['position_count'].fillna(0)
            in_position = positions > 0
            position_changes = in_position.astype(int).diff()
            entries = equity_curve.index[position_changes == 1].tolist()
            exits = equity_curve.index[position_changes == -1].tolist()
            
            for i in range(len(entries)):
                exit_time = exits[i] if i < len(exits) else equity_curve.index[-1]
                ax3.axvspan(entries[i], exit_time, alpha=0.15, color='green')
        except:
            pass
    
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.set_xlabel('Date')
    ax3.set_title(f'{strategy_name} Only - Detailed View')
    ax3.grid(True, alpha=0.3)
    
    # === Subplot 4: Drawdown ===
    try:
        equity_clean = equity_data.dropna()
        if len(equity_clean) > 0:
            running_max = equity_clean.expanding().max()
            drawdown = ((equity_clean - running_max) / running_max * 100).fillna(0)
            
            ax4.fill_between(equity_curve.index, 0, drawdown, color='#E74C3C', alpha=0.3)
            ax4.plot(equity_curve.index, drawdown, color='#E74C3C', linewidth=1)
            
            # Mark max drawdown
            if not drawdown.empty and not drawdown.isna().all():
                max_dd = drawdown.min()
                if not np.isnan(max_dd) and max_dd < 0:
                    max_dd_idx = drawdown.idxmin()
                    ax4.scatter([max_dd_idx], [max_dd], color='#E74C3C', s=100, zorder=5)
                    ax4.annotate(f'Max: {max_dd:.1f}%', 
                               xy=(max_dd_idx, max_dd),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9)
    except:
        pass
    
    ax4.set_ylabel('Drawdown (%)')
    ax4.set_xlabel('Date')
    ax4.set_title('Portfolio Drawdown')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(top=0)
    
    # Format x-axis dates for bottom plots
    for ax in [ax3, ax4]:
        try:
            date_range = (equity_curve.index[-1] - equity_curve.index[0]).days
            if date_range > 365:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            elif date_range > 90:
                ax.xaxis.set_major_locator(mdates.MonthLocator())
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            else:
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        except:
            pass
    
    plt.suptitle(f'{strategy_name} - Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def create_trades_plot(
    equity_curve: pd.DataFrame,
    trades_df: pd.DataFrame,
    strategy_name: str = 'Strategy'
) -> Optional[plt.Figure]:
    """Create trades analysis visualization."""
    # Create custom layout - position timeline at top, two analysis plots below
    fig = plt.figure(figsize=(14, 8))
    
    # Top row: Long position timeline spanning full width (takes 60% height)
    ax1 = plt.subplot2grid((5, 2), (0, 0), colspan=2, rowspan=3)
    
    # Bottom row: Duration and returns distribution (takes 40% height)
    ax2 = plt.subplot2grid((5, 2), (3, 0), rowspan=2)
    ax3 = plt.subplot2grid((5, 2), (3, 1), rowspan=2)
    
    # === Subplot 1: Trade Summary Visualization ===
    if trades_df is not None and not trades_df.empty:
        try:
            # Calculate trade statistics
            total_trades = len(trades_df)
            
            # Get entry counts
            entry_counts = []
            for idx, trade in trades_df.iterrows():
                if 'metadata' in trades_df.columns and isinstance(trade['metadata'], dict):
                    entry_counts.append(trade['metadata'].get('entry_count', 1))
                else:
                    entry_counts.append(1)
            
            # Create price chart with trade markers
            if equity_curve is not None and not equity_curve.empty:
                # Plot equity curve
                ax1.plot(equity_curve.index, equity_curve['total_value'], 
                        color='#2E86AB', linewidth=1.5, alpha=0.7, label='Portfolio Value')
                
                # Mark trades with different colors based on entry count
                colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))
                
                for idx, trade in trades_df.iterrows():
                    entry_time = pd.to_datetime(trade['entry_time'])
                    exit_time = pd.to_datetime(trade['exit_time'])
                    entry_count = entry_counts[idx]
                    
                    # Color based on entry count
                    color = colors[min(entry_count - 1, 9)]
                    
                    # Draw vertical span for trade duration
                    ax1.axvspan(entry_time, exit_time, alpha=0.2, color=color)
                    
                    # Mark entry point
                    if entry_time in equity_curve.index:
                        val = equity_curve.loc[entry_time, 'total_value']
                        ax1.scatter(entry_time, val, s=50, c=[color], 
                                   marker='^', edgecolors='black', linewidth=0.5, zorder=5)
                    
                    # Mark exit point
                    if exit_time in equity_curve.index:
                        val = equity_curve.loc[exit_time, 'total_value']
                        ax1.scatter(exit_time, val, s=50, c='red' if trade['pnl'] < 0 else 'green', 
                                   marker='v', edgecolors='black', linewidth=0.5, zorder=5)
                
                # Add statistics text box
                stats_text = f'Total Trades: {total_trades}\n'
                stats_text += f'Single Entry: {entry_counts.count(1)}\n'
                stats_text += f'Multi Entry: {total_trades - entry_counts.count(1)}\n'
                stats_text += f'Max Entries: {max(entry_counts)}'
                
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
                ax1.set_xlabel('Date', fontsize=11)
                ax1.set_title(f'{strategy_name} - Equity Curve with Trade Markers', fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend(loc='upper right', fontsize=9)
                
                # Format dates
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax1.xaxis.set_major_locator(mdates.MonthLocator())
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                # Fallback: Show trade count by entry type
                entry_dist = pd.Series(entry_counts).value_counts().sort_index()
                ax1.bar(entry_dist.index, entry_dist.values, color='#2E86AB', alpha=0.7)
                ax1.set_xlabel('Number of Entries', fontsize=11)
                ax1.set_ylabel('Trade Count', fontsize=11)
                ax1.set_title(f'{strategy_name} - Trade Distribution by Entry Count', fontsize=13, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Add value labels on bars
                for x, y in zip(entry_dist.index, entry_dist.values):
                    ax1.text(x, y, str(y), ha='center', va='bottom', fontsize=10)
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title(f'{strategy_name} - Trade Timeline')
    else:
        ax1.text(0.5, 0.5, 'No trade data available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'{strategy_name} - Trade Timeline')
    
    # === Subplot 2: Trade Duration Distribution ===
    if trades_df is not None and not trades_df.empty:
        try:
            # Calculate trade durations if entry and exit times are available
            if 'entry_time' in trades_df.columns and 'exit_time' in trades_df.columns:
                # Convert to datetime if needed
                entry_times = pd.to_datetime(trades_df['entry_time'])
                exit_times = pd.to_datetime(trades_df['exit_time'])
                
                # Calculate duration in hours
                durations = (exit_times - entry_times).dt.total_seconds() / 3600
                durations = durations[durations.notna() & (durations > 0)]
                
                if len(durations) > 0:
                    # Create histogram
                    ax2.hist(durations, bins=20, color='#2E86AB', alpha=0.7, edgecolor='black')
                    ax2.axvline(durations.mean(), color='red', linestyle='--', linewidth=1, 
                               label=f'Mean: {durations.mean():.1f}h')
                    ax2.set_xlabel('Trade Duration (hours)')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Trade Duration Distribution')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                else:
                    ax2.text(0.5, 0.5, 'No duration data', ha='center', va='center', transform=ax2.transAxes)
                    ax2.set_title('Trade Duration Distribution')
            else:
                ax2.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Trade Duration Distribution')
        except:
            ax2.text(0.5, 0.5, 'Error calculating durations', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Trade Duration Distribution')
    else:
        ax2.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Trade Duration Distribution')
    
    # === Subplot 3: Trade Returns Distribution (%) ===
    if trades_df is not None and not trades_df.empty:
        try:
            # Calculate returns percentage
            returns_pct = None
            
            # Try to find P&L and entry price columns
            pnl_col = None
            for col in ['pnl', 'realized_pnl', 'profit_loss']:
                if col in trades_df.columns:
                    pnl_col = col
                    break
            
            # Calculate returns as percentage
            if pnl_col and 'entry_price' in trades_df.columns and 'quantity' in trades_df.columns:
                # Returns = (P&L / (entry_price * quantity)) * 100
                entry_values = trades_df['entry_price'] * trades_df['quantity']
                returns_pct = (trades_df[pnl_col] / entry_values * 100).replace([np.inf, -np.inf], np.nan).dropna()
            elif pnl_col:
                # If we don't have entry price, estimate from P&L relative to some baseline
                # Assume average position size of 1000 for normalization
                returns_pct = (trades_df[pnl_col] / 1000 * 100).replace([np.inf, -np.inf], np.nan).dropna()
            
            if returns_pct is not None and len(returns_pct) > 0:
                # Create histogram with separate colors for profit/loss
                profits = returns_pct[returns_pct > 0]
                losses = returns_pct[returns_pct < 0]
                
                # Determine appropriate bins
                all_returns = returns_pct.values
                min_ret, max_ret = all_returns.min(), all_returns.max()
                bins = np.linspace(min_ret, max_ret, 31)
                
                if len(profits) > 0:
                    ax3.hist(profits, bins=bins, color='#27AE60', alpha=0.7, label=f'Profits ({len(profits)})', edgecolor='black')
                if len(losses) > 0:
                    ax3.hist(losses, bins=bins, color='#E74C3C', alpha=0.7, label=f'Losses ({len(losses)})', edgecolor='black')
                
                # Add mean line
                mean_return = returns_pct.mean()
                ax3.axvline(mean_return, color='blue', linestyle='--', linewidth=1,
                           label=f'Mean: {mean_return:.2f}%')
                
                # Add zero line
                ax3.axvline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.5)
                
                ax3.set_xlabel('Trade Returns (%)')
                ax3.set_ylabel('Frequency')
                ax3.set_title('Trade Returns Distribution')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No returns data', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Trade Returns Distribution')
        except Exception as e:
            ax3.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Trade Returns Distribution')
    else:
        ax3.text(0.5, 0.5, 'No trade data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Trade Returns Distribution')
    
    plt.suptitle(f'{strategy_name} - Trade Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig