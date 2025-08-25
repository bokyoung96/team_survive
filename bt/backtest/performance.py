from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import numpy as np
import polars as pl
from scipy import stats


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    exposure_time: float
    recovery_factor: float
    calmar_ratio: float
    
    skewness: float = 0.0
    kurtosis: float = 0.0
    information_ratio: float = 0.0


class PerformanceAnalyzer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, equity_curve: pd.DataFrame) -> pd.Series:
        # Use polars for faster pct_change calculation
        col_name = "total_value" if "total_value" in equity_curve.columns else "equity"
        if col_name not in equity_curve.columns:
            raise ValueError("Equity curve must have 'total_value' or 'equity' column")
        
        temp = pl.from_pandas(equity_curve[[col_name]])
        returns_pl = temp.select(
            pl.col(col_name).pct_change().alias("returns")
        )
        returns = returns_pl["returns"].to_pandas()
        returns.index = equity_curve.index
        return returns.fillna(0)
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        std_returns = excess_returns.std()
        if std_returns == 0 or np.isnan(std_returns):
            return 0.0
        
        mean_returns = excess_returns.mean()
        if np.isnan(mean_returns):
            return 0.0
            
        return np.sqrt(periods_per_year) * (mean_returns / std_returns)
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_std = np.sqrt(np.mean(downside_returns ** 2))
        
        if downside_std == 0 or np.isnan(downside_std):
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        mean_returns = excess_returns.mean()
        if np.isnan(mean_returns):
            return 0.0
            
        return np.sqrt(periods_per_year) * (mean_returns / downside_std)
    
    def calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> Dict[str, Any]:
        if "total_value" in equity_curve.columns:
            equity = equity_curve["total_value"]
        elif "equity" in equity_curve.columns:
            equity = equity_curve["equity"]
        else:
            raise ValueError("Equity curve must have 'total_value' or 'equity' column")
        
        temp = pl.from_pandas(pd.DataFrame({'equity': equity}))
        result = temp.with_columns([
            pl.col("equity").cum_max().alias("running_max")
        ]).with_columns([
            pl.when(pl.col("running_max") != 0)
            .then((pl.col("equity") - pl.col("running_max")) / pl.col("running_max"))
            .otherwise(0.0)
            .alias("drawdown")
        ])
        
        running_max = result["running_max"].to_pandas()
        running_max.index = equity.index
        
        drawdown = result["drawdown"].to_pandas()
        drawdown.index = equity.index
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        peak_idx = equity[:max_dd_idx].idxmax()
        
        recovery_idx = None
        max_dd_position = equity.index.get_loc(max_dd_idx)
        if max_dd_position < len(equity) - 1:
            post_dd_equity = equity[max_dd_idx:]
            peak_value = equity[peak_idx]
            recovery_points = post_dd_equity[post_dd_equity >= peak_value]
            if not recovery_points.empty:
                recovery_idx = recovery_points.index[0]
        
        return {
            "max_drawdown": abs(max_dd),
            "max_dd_date": max_dd_idx,
            "peak_date": peak_idx,
            "recovery_date": recovery_idx,
            "drawdown_series": drawdown
        }
    
    def calculate_trade_statistics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        if trades.empty:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "avg_trade_duration": 0.0
            }
        
        if "pnl" in trades.columns:
            pnl_col = "pnl"
        elif "realized_pnl" in trades.columns:
            pnl_col = "realized_pnl"
        else:
            if all(col in trades.columns for col in ["entry_price", "exit_price", "quantity"]):
                pnl = (trades["exit_price"] - trades["entry_price"]) * trades["quantity"]
                pnl_col = None
            else:
                raise ValueError("Trades must have P&L information")
        
        if pnl_col:
            temp = pl.from_pandas(trades[[pnl_col]])
            temp = temp.rename({pnl_col: "pnl"})
        else:
            temp = pl.from_pandas(pd.DataFrame({"pnl": pnl}))
        
        stats = temp.select([
            pl.col("pnl").count().alias("total_trades"),
            pl.col("pnl").filter(pl.col("pnl") > 0).count().alias("num_winners"),
            pl.col("pnl").filter(pl.col("pnl") < 0).count().alias("num_losers"),
            pl.col("pnl").filter(pl.col("pnl") > 0).sum().alias("total_wins"),
            pl.col("pnl").filter(pl.col("pnl") < 0).sum().alias("total_losses"),
            pl.col("pnl").filter(pl.col("pnl") > 0).mean().alias("avg_win"),
            pl.col("pnl").filter(pl.col("pnl") < 0).mean().alias("avg_loss"),
            pl.col("pnl").filter(pl.col("pnl") > 0).max().alias("largest_win"),
            pl.col("pnl").filter(pl.col("pnl") < 0).min().alias("largest_loss"),
        ]).to_dicts()[0]
        
        total_trades = stats["total_trades"]
        num_winners = stats["num_winners"] or 0
        num_losers = stats["num_losers"] or 0
        
        win_rate = num_winners / total_trades if total_trades > 0 else 0.0
        
        total_wins = stats["total_wins"] or 0
        total_losses = abs(stats["total_losses"] or 0)
        
        if total_losses > 0:
            profit_factor = total_wins / total_losses
        elif total_wins > 0:
            profit_factor = float('inf')
        else:
            profit_factor = 0.0
        
        avg_win = stats["avg_win"] or 0.0
        avg_loss = stats["avg_loss"] or 0.0
        largest_win = stats["largest_win"] or 0.0
        largest_loss = stats["largest_loss"] or 0.0
        
        avg_duration = 0.0
        if "entry_time" in trades.columns and "exit_time" in trades.columns:
            durations = pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
            avg_duration = durations.mean().total_seconds() / 3600
        
        return {
            "total_trades": total_trades,
            "winning_trades": num_winners,
            "losing_trades": num_losers,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "largest_win": float(largest_win),
            "largest_loss": float(largest_loss),
            "avg_trade_duration": avg_duration
        }
    
    def calculate_exposure_time(
        self,
        equity_curve: pd.DataFrame,
        positions_column: str = "position_count"
    ) -> float:
        if positions_column not in equity_curve.columns:
            return 0.0
        
        periods_with_position = (equity_curve[positions_column] > 0).sum()
        total_periods = len(equity_curve)
        
        if total_periods > 0:
            return periods_with_position / total_periods
        else:
            return 0.0
    
    
    def calculate_advanced_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        if len(returns) < 10:
            return {
                'skewness': 0.0, 'kurtosis': 0.0, 'information_ratio': 0.0
            }
        
        returns_clean = returns.dropna()
        
        skewness = stats.skew(returns_clean)
        kurtosis = stats.kurtosis(returns_clean, fisher=True)
        
        benchmark_return = 0.0
        excess_returns = returns_clean - benchmark_return
        std_excess = excess_returns.std()
        if std_excess > 0 and not np.isnan(std_excess):
            mean_excess = excess_returns.mean()
            if not np.isnan(mean_excess):
                information_ratio = mean_excess / std_excess
            else:
                information_ratio = 0.0
        else:
            information_ratio = 0.0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'information_ratio': information_ratio
        }
    
    def analyze_performance(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        initial_capital: float,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:

        returns = self.calculate_returns(equity_curve)
        
        if "total_value" in equity_curve.columns:
            final_equity = equity_curve["total_value"].iloc[-1]
        elif "equity" in equity_curve.columns:
            final_equity = equity_curve["equity"].iloc[-1]
        else:
            final_equity = initial_capital
        
        if initial_capital > 0:
            total_return = (final_equity - initial_capital) / initial_capital
        else:
            total_return = 0.0
        
        num_periods = len(equity_curve)
        if num_periods > 0:
            years = num_periods / periods_per_year
            if years > 0 and total_return > -1:
                annualized_return = (1 + total_return) ** (1 / years) - 1
            else:
                annualized_return = total_return
        else:
            annualized_return = 0.0
        
        sharpe_ratio = self.calculate_sharpe_ratio(returns, periods_per_year)
        sortino_ratio = self.calculate_sortino_ratio(returns, periods_per_year)
        
        dd_stats = self.calculate_max_drawdown(equity_curve)
        max_drawdown = dd_stats["max_drawdown"]
        
        trade_stats = self.calculate_trade_statistics(trades)
        
        exposure_time = self.calculate_exposure_time(equity_curve)
        
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf') if total_return > 0 else 0.0
        
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf') if annualized_return > 0 else 0.0
        
        advanced_metrics = self.calculate_advanced_risk_metrics(returns)
        
        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            largest_win=trade_stats["largest_win"],
            largest_loss=trade_stats["largest_loss"],
            avg_trade_duration=trade_stats["avg_trade_duration"],
            exposure_time=exposure_time,
            recovery_factor=recovery_factor,
            calmar_ratio=calmar_ratio,
            **advanced_metrics
        )
    
    def generate_report(self, metrics: PerformanceMetrics) -> str:
        report = f"""
╔══════════════════════════════════════════════════════════╗
║             BACKTESTING PERFORMANCE REPORT               ║
╚══════════════════════════════════════════════════════════╝

═══ RETURNS ═══════════════════════════════════════════════
  Total Return:          {metrics.total_return:>10.2%}
  Annualized Return:     {metrics.annualized_return:>10.2%}
  
═══ RISK METRICS ═══════════════════════════════════════════
  Sharpe Ratio:          {metrics.sharpe_ratio:>10.2f}
  Sortino Ratio:         {metrics.sortino_ratio:>10.2f}
  Information Ratio:     {metrics.information_ratio:>10.2f}
  Max Drawdown:          {metrics.max_drawdown:>10.2%}
  Calmar Ratio:          {metrics.calmar_ratio:>10.2f}
  Recovery Factor:       {metrics.recovery_factor:>10.2f}
  
═══ DISTRIBUTION METRICS ═══════════════════════════════════
  Skewness:              {metrics.skewness:>10.2f}
  Kurtosis:              {metrics.kurtosis:>10.2f}
  
═══ TRADE STATISTICS ═══════════════════════════════════════
  Total Trades:          {metrics.total_trades:>10d}
  Winning Trades:        {metrics.winning_trades:>10d}
  Losing Trades:         {metrics.losing_trades:>10d}
  Win Rate:              {metrics.win_rate:>10.2%}
  Profit Factor:         {metrics.profit_factor:>10.2f}
  
═══ TRADE ANALYSIS ═════════════════════════════════════════
  Average Win:           {metrics.avg_win:>10.2f}
  Average Loss:          {metrics.avg_loss:>10.2f}
  Largest Win:           {metrics.largest_win:>10.2f}
  Largest Loss:          {metrics.largest_loss:>10.2f}
  Avg Trade Duration:    {metrics.avg_trade_duration:>10.1f} hours
  
═══ EXPOSURE ═══════════════════════════════════════════════
  Market Exposure:       {metrics.exposure_time:>10.2%}
        """
        return report