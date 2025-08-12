from __future__ import annotations
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Tuple


class PlotLogic:
    def _prettify_label(self, raw: str) -> str:
        lower = raw.lower()
        if lower.startswith("ma_") and lower.split("_")[-1].isdigit():
            length = lower.split("_")[-1]
            return f"MA({length})"
        if lower.startswith("rsi"):
            parts = lower.split("_")
            return f"RSI({parts[-1]})" if parts[-1].isdigit() else "RSI"
        if lower.startswith("macd"):
            if lower.endswith("_signal"):
                return "MACD Signal"
            if lower.endswith("_hist"):
                return "MACD Hist"
            return "MACD"
        if lower.startswith("ichimoku"):
            tail = raw[len("ichimoku_") :] if lower.startswith("ichimoku_") else raw
            return f"Ichimoku {tail.replace('_', ' ').title()}"
        return raw.replace("_", " ").title()

    def _is_oscillator(self, name: str, series: pd.Series) -> bool:
        lname = name.lower()
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", lname)))
        if "rsi" in tokens or lname.startswith("rsi"):
            return True
        if "macd" in tokens or lname.startswith("macd"):
            return True
        if "stoch" in tokens or lname.startswith("stoch"):
            return True
        if "momentum" in tokens or lname.startswith("momentum"):
            return True
        values = series.dropna()
        if values.empty:
            return False
        q_hi = float(values.quantile(0.95))
        q_lo = float(values.quantile(0.05))
        span = abs(q_hi - q_lo)
        if span <= 200 and values.max() <= 1000 and values.min() >= -1000:
            return True
        return False


class Plots(PlotLogic):
    def __init__(self, title: str = "Technical Analysis", figsize: Tuple[int, int] = (12, 8)):
        self.title = title
        self.figsize = figsize
        self.fig, (self.price_ax, self.osc_ax, self.volume_ax) = plt.subplots(
            3,
            1,
            figsize=figsize,
            gridspec_kw={"height_ratios": [4, 2, 1]},
        )
        self.fig.suptitle(title)

        self.price_ax.set_ylabel("Price")
        self.price_ax.grid(True, alpha=0.3)

        self.osc_ax.set_ylabel("Oscillator")
        self.osc_ax.grid(True, alpha=0.3)
        self.osc_ax.set_visible(False)

        self.volume_ax.set_ylabel("Volume")
        self.volume_ax.set_xlabel("Date")
        self.volume_ax.grid(True, alpha=0.3)

    def add_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        self.price_ax.plot(
            ohlcv.index,
            ohlcv["close"],
            color="black",
            linewidth=2.2,
            label="Close",
            zorder=5,
        )
        if "volume" in ohlcv.columns:
            self.volume_ax.bar(ohlcv.index, ohlcv["volume"], alpha=0.6)

    def add_line(self, data: pd.Series, label: str | None = None, on_osc: bool = False) -> None:
        target_ax = self.osc_ax if on_osc else self.price_ax
        target_ax.set_visible(True) if on_osc else None
        target_ax.plot(data.index, data.values, label=label, linewidth=1.2)

    def add_indicators(self, indicators: pd.DataFrame) -> None:
        for col in indicators.columns:
            series = indicators[col].dropna()
            if series.empty:
                continue
            label = self._prettify_label(col)
            on_osc = self._is_oscillator(col, series)
            self.add_line(series, label=label, on_osc=on_osc)

    def finalize(self) -> None:
        if len(self.price_ax.get_lines()) > 0:
            self.price_ax.legend(loc="upper left")
        if self.osc_ax.get_visible() and len(self.osc_ax.get_lines()) > 0:
            self.osc_ax.legend(loc="upper left")
        plt.tight_layout()

    def save(self, filename: str = "chart.png") -> None:
        self.finalize()
        self.fig.savefig(filename, dpi=150, bbox_inches="tight")
        print(f"Saved: {filename}")

    def show(self) -> None:
        self.finalize()
        plt.show()

    def plot(self, data: Dict[str, pd.DataFrame]) -> None:
        if "ohlcv" in data:
            self.add_ohlcv(data["ohlcv"])
        if "indicators" in data:
            self.add_indicators(data["indicators"])
        self.show()


class GetPlots(PlotLogic):
    def __init__(self, title: str = "Technical Analysis", width: int = 1400, height: int = 900):
        self.title = title
        self.fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.25, 0.15],
            vertical_spacing=0.02,
        )
        self.fig.update_layout(
            title=dict(text=title, y=0.98),
            showlegend=True,
            width=width,
            height=height,
            margin=dict(l=60, r=30, t=70, b=90),
            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="left", x=0),
        )
        self._has_osc = False

    def add_ohlcv(self, ohlcv: pd.DataFrame) -> None:
        self.fig.add_trace(
            go.Scatter(x=ohlcv.index, y=ohlcv["close"], name="Close", mode="lines", line=dict(color="black", width=2.2)),
            row=1,
            col=1,
        )
        if "volume" in ohlcv.columns:
            self.fig.add_trace(
                go.Bar(x=ohlcv.index, y=ohlcv["volume"], name="Volume", marker=dict(opacity=0.6)),
                row=3,
                col=1,
            )
        self.fig.update_yaxes(title_text="Price", row=1, col=1)
        self.fig.update_yaxes(title_text="Oscillator", row=2, col=1)
        self.fig.update_yaxes(title_text="Volume", row=3, col=1)

    def add_line(self, data: pd.Series, label: str | None = None, on_osc: bool = False) -> None:
        row = 2 if on_osc else 1
        if on_osc:
            self._has_osc = True
        self.fig.add_trace(
            go.Scatter(x=data.index, y=data.values, name=label or "" , mode="lines"),
            row=row,
            col=1,
        )

    def add_indicators(self, indicators: pd.DataFrame) -> None:
        for col in indicators.columns:
            series = indicators[col].dropna()
            if series.empty:
                continue
            label = self._prettify_label(col)
            on_osc = self._is_oscillator(col, series)
            self.add_line(series, label=label, on_osc=on_osc)

    def finalize(self) -> None:
        if not self._has_osc:
            self.fig.update_yaxes(visible=False, row=2, col=1)

    def show(self) -> None:
        self.finalize()
        self.fig.show()

    def save_html(self, filename: str = "chart.html") -> None:
        self.finalize()
        html = self.fig.to_html(full_html=True, include_plotlyjs="cdn")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(html)

    def save_png(self, filename: str = "chart.png", scale: float = 2.0) -> None:
        self.finalize()
        try:
            self.fig.write_image(filename, scale=scale)
        except Exception as e:
            raise RuntimeError(str(e))

    def plot(self, data: Dict[str, pd.DataFrame]) -> None:
        if "ohlcv" in data:
            self.add_ohlcv(data["ohlcv"])
        if "indicators" in data:
            self.add_indicators(data["indicators"])
        self.show()