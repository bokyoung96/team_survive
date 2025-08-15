import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

from backtest.models import BacktestResult
from backtest.performance import PerformanceMetrics

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


@dataclass
class PlotConfig:
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = 'png'
    output_dir: str = 'backtest_results'
    show_plots: bool = False
    color_scheme: Dict[str, str] = None
    
    def __post_init__(self):
        if self.color_scheme is None:
            self.color_scheme = {
                'equity': '#2E86AB',
                'benchmark': '#A23B72',
                'buy_signal': '#F18F01',
                'sell_signal': '#C73E1D',
                'drawdown': '#FF6B6B',
                'win_trade': '#28A745',
                'loss_trade': '#DC3545',
                'grid': '#E5E5E5'
            }


class BacktestVisualizer:
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(exist_ok=True)
    
    def plot_equity_curve(self, 
                         result: BacktestResult, 
                         benchmark_data: Optional[pd.DataFrame] = None,
                         save_file: Optional[str] = None) -> plt.Figure:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size, 
                                       height_ratios=[3, 1], sharex=True)
        
        equity_curve = result.equity_curve.copy()
        
        ax1.plot(equity_curve.index, equity_curve['total_value'], 
                color=self.config.color_scheme['equity'], linewidth=2, 
                label='Portfolio Value', alpha=0.9)
        
        if benchmark_data is not None and 'close' in benchmark_data.columns:
            initial_value = float(equity_curve['total_value'].iloc[0])
            benchmark_normalized = (benchmark_data['close'] / benchmark_data['close'].iloc[0]) * initial_value
            ax1.plot(benchmark_data.index, benchmark_normalized, 
                    color=self.config.color_scheme['benchmark'], linewidth=1.5,
                    label='Buy & Hold', alpha=0.7, linestyle='--')
        
        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        if 'total_value' in equity_curve.columns:
            running_max = equity_curve['total_value'].expanding().max()
            drawdown = (equity_curve['total_value'] - running_max) / running_max
            
            ax2.fill_between(equity_curve.index, drawdown, 0, 
                           color=self.config.color_scheme['drawdown'], 
                           alpha=0.3, label='Drawdown')
            ax2.plot(equity_curve.index, drawdown, 
                    color=self.config.color_scheme['drawdown'], 
                    linewidth=1, alpha=0.8)
            
        ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.grid(True, alpha=0.3)
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_file:
            self._save_plot(fig, save_file)
            
        return fig
    
    def plot_trades_on_price(self, 
                           result: BacktestResult, 
                           price_data: pd.DataFrame,
                           save_file: Optional[str] = None) -> plt.Figure:

        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        if all(col in price_data.columns for col in ['open', 'high', 'low', 'close']):
            ax.plot(price_data.index, price_data['close'], 
                   color='black', linewidth=1, alpha=0.7, label='Price')
            
            ax.fill_between(price_data.index, price_data['low'], price_data['high'], 
                          alpha=0.1, color='gray')
        else:
            price_col = next((col for col in ['close', 'price'] if col in price_data.columns), None)
            if price_col:
                ax.plot(price_data.index, price_data[price_col], 
                       color='black', linewidth=1, alpha=0.7, label='Price')
        
        trades = result.trades
        if not trades.empty:
            if 'entry_time' in trades.columns and 'entry_price' in trades.columns:
                buy_trades = trades[trades['side'] == 'BUY'] if 'side' in trades.columns else trades
                sell_trades = trades[trades['side'] == 'SELL'] if 'side' in trades.columns else pd.DataFrame()
                
                if not buy_trades.empty:
                    ax.scatter(pd.to_datetime(buy_trades['entry_time']), 
                             buy_trades['entry_price'], 
                             color=self.config.color_scheme['buy_signal'], 
                             marker='^', s=60, alpha=0.8, 
                             label='Buy Entry', zorder=5)
                
                if not sell_trades.empty:
                    ax.scatter(pd.to_datetime(sell_trades['entry_time']), 
                             sell_trades['entry_price'], 
                             color=self.config.color_scheme['sell_signal'], 
                             marker='v', s=60, alpha=0.8, 
                             label='Sell Entry', zorder=5)
            
            if 'exit_time' in trades.columns and 'exit_price' in trades.columns:
                profitable_trades = trades[trades['pnl'] > 0] if 'pnl' in trades.columns else trades
                losing_trades = trades[trades['pnl'] <= 0] if 'pnl' in trades.columns else pd.DataFrame()
                
                if not profitable_trades.empty:
                    ax.scatter(pd.to_datetime(profitable_trades['exit_time']), 
                             profitable_trades['exit_price'], 
                             color=self.config.color_scheme['win_trade'], 
                             marker='x', s=40, alpha=0.8, 
                             label='Profitable Exit', zorder=5)
                
                if not losing_trades.empty:
                    ax.scatter(pd.to_datetime(losing_trades['exit_time']), 
                             losing_trades['exit_price'], 
                             color=self.config.color_scheme['loss_trade'], 
                             marker='x', s=40, alpha=0.8, 
                             label='Loss Exit', zorder=5)
        
        ax.set_title('Trade Entry/Exit Points on Price Chart', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price ($)', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.2f}'))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_file:
            self._save_plot(fig, save_file)
            
        return fig
    
    def plot_drawdown_periods(self, 
                            result: BacktestResult, 
                            save_file: Optional[str] = None) -> plt.Figure:

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size, 
                                       height_ratios=[2, 1], sharex=True)
        
        equity_curve = result.equity_curve.copy()
        
        if 'total_value' in equity_curve.columns:
            running_max = equity_curve['total_value'].expanding().max()
            drawdown = (equity_curve['total_value'] - running_max) / running_max
            
            ax1.plot(equity_curve.index, equity_curve['total_value'], 
                    color=self.config.color_scheme['equity'], linewidth=2, 
                    label='Portfolio Value', alpha=0.9)
            ax1.plot(equity_curve.index, running_max, 
                    color='red', linewidth=1, linestyle='--', 
                    label='Running Maximum', alpha=0.7)
            
            significant_dd = drawdown < -0.05
            if significant_dd.any():
                dd_periods = []
                in_drawdown = False
                start_idx = None
                
                for idx, is_dd in significant_dd.items():
                    if is_dd and not in_drawdown:
                        start_idx = idx
                        in_drawdown = True
                    elif not is_dd and in_drawdown:
                        dd_periods.append((start_idx, idx))
                        in_drawdown = False
                
                if in_drawdown and start_idx is not None:
                    dd_periods.append((start_idx, equity_curve.index[-1]))
                
                for start, end in dd_periods:
                    ax1.axvspan(start, end, alpha=0.2, color=self.config.color_scheme['drawdown'], 
                              label='Significant Drawdown (>5%)' if (start, end) == dd_periods[0] else "")
            
            ax1.set_title('Drawdown Periods Analysis', fontsize=16, fontweight='bold', pad=20)
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.grid(True, alpha=0.3)
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            ax2.fill_between(equity_curve.index, drawdown, 0, 
                           color=self.config.color_scheme['drawdown'], 
                           alpha=0.4, label='Drawdown')
            ax2.plot(equity_curve.index, drawdown, 
                    color=self.config.color_scheme['drawdown'], 
                    linewidth=1.5, alpha=0.8)
            
            max_dd_idx = drawdown.idxmin()
            max_dd_value = drawdown.min()
            ax2.scatter([max_dd_idx], [max_dd_value], 
                      color='red', s=100, zorder=5, 
                      label=f'Max Drawdown: {max_dd_value:.2%}')
        
        ax2.set_title('Drawdown Timeline', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Drawdown (%)', fontsize=10)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.legend(loc='lower right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_file:
            self._save_plot(fig, save_file)
            
        return fig
    
    def plot_monthly_returns_heatmap(self, 
                                   result: BacktestResult, 
                                   save_file: Optional[str] = None) -> plt.Figure:

        fig, ax = plt.subplots(figsize=(14, 8))
        
        equity_curve = result.equity_curve.copy()
        
        if 'total_value' in equity_curve.columns:
            returns = equity_curve['total_value'].pct_change().fillna(0)
            
            monthly_returns = (1 + returns).resample('M').prod() - 1
            
            monthly_data = monthly_returns.to_frame('returns')
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month
            
            heatmap_data = monthly_data.pivot(index='year', columns='month', values='returns')
            
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            heatmap_data.columns = [month_names[i-1] for i in heatmap_data.columns]
            
            sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                       center=0, cbar_kws={'label': 'Monthly Return'}, 
                       linewidths=0.5, ax=ax)
            
            ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Year', fontsize=12)
            
        plt.tight_layout()
        
        if save_file:
            self._save_plot(fig, save_file)
            
        return fig
    
    def plot_trade_distribution(self, 
                              result: BacktestResult, 
                              save_file: Optional[str] = None) -> plt.Figure:

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        trades = result.trades
        
        if not trades.empty and 'pnl' in trades.columns:
            pnl_values = trades['pnl'].astype(float)
            
            ax1.hist(pnl_values, bins=30, alpha=0.7, color=self.config.color_scheme['equity'], 
                    edgecolor='black', linewidth=0.5)
            ax1.axvline(0, color='red', linestyle='--', alpha=0.8, label='Breakeven')
            ax1.axvline(pnl_values.mean(), color='orange', linestyle='-', alpha=0.8, 
                       label=f'Mean: ${pnl_values.mean():.2f}')
            ax1.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Profit/Loss ($)', fontsize=10)
            ax1.set_ylabel('Frequency', fontsize=10)
            ax1.legend(fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            winners = (pnl_values > 0).sum()
            losers = (pnl_values < 0).sum()
            breakeven = (pnl_values == 0).sum()
            
            sizes = [winners, losers, breakeven] if breakeven > 0 else [winners, losers]
            labels = ['Winners', 'Losers', 'Breakeven'] if breakeven > 0 else ['Winners', 'Losers']
            colors = [self.config.color_scheme['win_trade'], self.config.color_scheme['loss_trade'], 'gray'][:len(sizes)]
            
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, 
                   startangle=90, explode=(0.05, 0.05, 0) if breakeven > 0 else (0.05, 0.05))
            ax2.set_title(f'Trade Outcome Distribution\n({len(trades)} total trades)', 
                         fontsize=12, fontweight='bold')
            
            cumulative_pnl = pnl_values.cumsum()
            ax3.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                    color=self.config.color_scheme['equity'], linewidth=2)
            ax3.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0, 
                           alpha=0.3, color=self.config.color_scheme['equity'])
            ax3.set_title('Cumulative P&L by Trade', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Trade Number', fontsize=10)
            ax3.set_ylabel('Cumulative P&L ($)', fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            if 'entry_time' in trades.columns and 'exit_time' in trades.columns:
                try:
                    entry_times = pd.to_datetime(trades['entry_time'])
                    exit_times = pd.to_datetime(trades['exit_time'])
                    durations = (exit_times - entry_times).dt.total_seconds() / 3600  # in hours
                    
                    ax4.hist(durations, bins=20, alpha=0.7, color=self.config.color_scheme['equity'], 
                           edgecolor='black', linewidth=0.5)
                    ax4.set_title('Trade Duration Distribution', fontsize=12, fontweight='bold')
                    ax4.set_xlabel('Duration (hours)', fontsize=10)
                    ax4.set_ylabel('Frequency', fontsize=10)
                    ax4.grid(True, alpha=0.3)
                except:
                    if 'quantity' in trades.columns:
                        quantities = trades['quantity'].astype(float)
                        ax4.hist(quantities, bins=20, alpha=0.7, color=self.config.color_scheme['equity'], 
                               edgecolor='black', linewidth=0.5)
                        ax4.set_title('Trade Size Distribution', fontsize=12, fontweight='bold')
                        ax4.set_xlabel('Quantity', fontsize=10)
                        ax4.set_ylabel('Frequency', fontsize=10)
                        ax4.grid(True, alpha=0.3)
                    else:
                        ax4.text(0.5, 0.5, 'No Duration Data Available', 
                               ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                        ax4.set_title('Trade Duration Analysis', fontsize=12, fontweight='bold')
            else:
                ax4.text(0.5, 0.5, 'No Duration Data Available', 
                       ha='center', va='center', transform=ax4.transAxes, fontsize=12)
                ax4.set_title('Trade Duration Analysis', fontsize=12, fontweight='bold')
        else:
            for ax, title in zip([ax1, ax2, ax3, ax4], 
                               ['Trade P&L Distribution', 'Trade Outcome Distribution', 
                                'Cumulative P&L by Trade', 'Trade Duration Analysis']):
                ax.text(0.5, 0.5, 'No Trade Data Available', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
        
        plt.suptitle('Trade Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_file:
            self._save_plot(fig, save_file)
            
        return fig
    
    def plot_performance_metrics_dashboard(self, 
                                         metrics: PerformanceMetrics, 
                                         save_file: Optional[str] = None) -> plt.Figure:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        return_metrics = {
            'Total Return': metrics.total_return,
            'Annualized Return': metrics.annualized_return,
        }
        
        bars1 = ax1.bar(return_metrics.keys(), [v*100 for v in return_metrics.values()], 
                       color=[self.config.color_scheme['win_trade'], self.config.color_scheme['equity']], 
                       alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Return Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Return (%)', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        for bar, value in zip(bars1, return_metrics.values()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.2%}', ha='center', va='bottom', fontweight='bold')
        
        risk_metrics = {
            'Sharpe Ratio': metrics.sharpe_ratio,
            'Sortino Ratio': metrics.sortino_ratio,
            'Calmar Ratio': metrics.calmar_ratio,
            'Max Drawdown': -metrics.max_drawdown * 100
        }
        
        colors = [self.config.color_scheme['equity'], self.config.color_scheme['equity'], 
                 self.config.color_scheme['equity'], self.config.color_scheme['drawdown']]
        bars2 = ax2.bar(risk_metrics.keys(), risk_metrics.values(), 
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax2.set_title('Risk Metrics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Ratio / Percentage', fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        for bar, (key, value) in zip(bars2, risk_metrics.items()):
            if 'Drawdown' in key:
                label = f'{value:.1f}%'
            else:
                label = f'{value:.2f}'
            ax2.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (0.5 if value >= 0 else -0.8), 
                    label, ha='center', va='bottom' if value >= 0 else 'top', fontweight='bold')
        
        total_trades = metrics.total_trades
        if total_trades > 0:
            sizes = [metrics.winning_trades, metrics.losing_trades]
            labels = [f'Winners\n({metrics.winning_trades})', f'Losers\n({metrics.losing_trades})']
            colors_pie = [self.config.color_scheme['win_trade'], self.config.color_scheme['loss_trade']]
            
            ax3.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, 
                   startangle=90, explode=(0.05, 0.05))
            ax3.set_title(f'Trade Distribution\nWin Rate: {metrics.win_rate:.1%}', 
                         fontsize=12, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Trades\nExecuted', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, fontweight='bold')
            ax3.set_title('Trade Distribution', fontsize=12, fontweight='bold')
        
        additional_metrics = {
            'Profit Factor': f'{metrics.profit_factor:.2f}',
            'Avg Win': f'${metrics.avg_win:.2f}',
            'Avg Loss': f'${metrics.avg_loss:.2f}',
            'Largest Win': f'${metrics.largest_win:.2f}',
            'Largest Loss': f'${metrics.largest_loss:.2f}',
            'Avg Duration': f'{metrics.avg_trade_duration:.1f}h',
            'Exposure Time': f'{metrics.exposure_time:.1%}',
            'Recovery Factor': f'{metrics.recovery_factor:.2f}'
        }
        
        ax4.axis('off')
        
        table_data = []
        for i, (key, value) in enumerate(additional_metrics.items()):
            table_data.append([key, value])
        
        table = ax4.table(cellText=table_data, 
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#E5E5E5')
            else:
                cell.set_facecolor('#F8F9FA')
            cell.set_edgecolor('black')
            cell.set_linewidth(1)
        
        ax4.set_title('Additional Metrics', fontsize=12, fontweight='bold', pad=20)
        
        plt.suptitle('Performance Metrics Dashboard', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_file:
            self._save_plot(fig, save_file)
            
        return fig
    
    def generate_all_plots(self, 
                          result: BacktestResult, 
                          metrics: PerformanceMetrics,
                          price_data: Optional[pd.DataFrame] = None,
                          benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, plt.Figure]:

        plots = {}
        
        try:
            plots['equity_curve'] = self.plot_equity_curve(
                result, benchmark_data, 'equity_curve.png')
            
            if price_data is not None:
                plots['trades_on_price'] = self.plot_trades_on_price(
                    result, price_data, 'trades_on_price.png')
            
            plots['drawdown_periods'] = self.plot_drawdown_periods(
                result, 'drawdown_periods.png')
            
            plots['monthly_returns'] = self.plot_monthly_returns_heatmap(
                result, 'monthly_returns_heatmap.png')
            
            plots['trade_distribution'] = self.plot_trade_distribution(
                result, 'trade_distribution.png')
            
            plots['metrics_dashboard'] = self.plot_performance_metrics_dashboard(
                metrics, 'performance_metrics_dashboard.png')
            
            print(f"Generated {len(plots)} visualization plots in {self.output_path}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
            
        return plots
    
    def _save_plot(self, fig: plt.Figure, filename: str) -> None:
        try:
            filepath = self.output_path / filename
            fig.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight', 
                       format=self.config.save_format, facecolor='white')
            print(f"Saved plot: {filepath}")
            
            if not self.config.show_plots:
                plt.close(fig)
                
        except Exception as e:
            print(f"Error saving plot {filename}: {e}")


def create_equity_curve_plot(result: BacktestResult, 
                            benchmark_data: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
    config = PlotConfig(output_dir=save_path or 'backtest_results')
    visualizer = BacktestVisualizer(config)
    return visualizer.plot_equity_curve(result, benchmark_data, 'equity_curve.png')


def create_comprehensive_report_plots(result: BacktestResult, 
                                    metrics: PerformanceMetrics,
                                    price_data: Optional[pd.DataFrame] = None,
                                    save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
    config = PlotConfig(output_dir=save_path or 'backtest_results')
    visualizer = BacktestVisualizer(config)
    return visualizer.generate_all_plots(result, metrics, price_data)