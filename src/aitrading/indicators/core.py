"""テクニカル指標。すべて「時点tまでの情報しか使わない」純関数。

TA-Lib を使わないのは移植性のためだけではない。全指標をレジストリに集め、
トランケーション不変性検査を機械的に適用できるようにするため。
外部ライブラリではこの保証を自分たちで持てない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aitrading.indicators.registry import indicator, mid
from aitrading.timeutil import trading_day_start

__all__ = ["sma", "ema", "rsi", "macd", "atr", "bbands", "vwap", "donchian", "hist_vol"]


@indicator("sma")
def sma(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    return mid(bars, "close").rolling(period, min_periods=period).mean().rename("sma")


@indicator("ema")
def ema(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    return mid(bars, "close").ewm(span=period, adjust=False).mean().rename("ema")


@indicator("rsi")
def rsi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = mid(bars, "close").diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder の平滑化。ewm は過去のみを見る
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    # 下落がゼロの区間は RS が発散するので RSI は上限の100とするのが定義どおり
    # （avg_loss==0 → 100）。avg_loss が NaN（ウォームアップ未達）の間は
    # avg_loss==0 が False になり、rs=NaN 経由でそのまま NaN が伝播する。
    values = np.where(avg_loss.to_numpy() == 0.0, 100.0, 100.0 - 100.0 / (1.0 + rs.to_numpy()))
    return pd.Series(values, index=bars.index, name="rsi")


@indicator("macd")
def macd(
    bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    close = mid(bars, "close")
    line = (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "signal": signal_line, "histogram": line - signal_line}
    )


@indicator("atr")
def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = mid(bars, "high"), mid(bars, "low")
    prev_close = mid(bars, "close").shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return (
        true_range.ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
        .rename("atr")
    )


@indicator("bbands")
def bbands(bars: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    close = mid(bars, "close")
    middle = close.rolling(period, min_periods=period).mean()
    spread = close.rolling(period, min_periods=period).std(ddof=0) * num_std
    return pd.DataFrame(
        {"lower": middle - spread, "middle": middle, "upper": middle + spread}
    )


@indicator("vwap")
def vwap(bars: pd.DataFrame) -> pd.Series:
    """当日の累積VWAP。日境界はNY基準。

    全期間の合計で割ると未来を見ることになるので、必ず累積和で計算する。
    """
    typical = (mid(bars, "high") + mid(bars, "low") + mid(bars, "close")) / 3.0
    day = pd.Series(trading_day_start(pd.DatetimeIndex(bars.index), "ny"), index=bars.index)
    volume = bars["volume"]
    cum_pv = (typical * volume).groupby(day).cumsum()
    cum_v = volume.groupby(day).cumsum()
    return (cum_pv / cum_v).rename("vwap")


@indicator("donchian")
def donchian(bars: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    upper = mid(bars, "high").rolling(period, min_periods=period).max()
    lower = mid(bars, "low").rolling(period, min_periods=period).min()
    return pd.DataFrame({"lower": lower, "upper": upper, "width": upper - lower})


@indicator("hist_vol")
def hist_vol(bars: pd.DataFrame, period: int = 60) -> pd.Series:
    returns = np.log(mid(bars, "close")).diff()
    return returns.rolling(period, min_periods=period).std(ddof=0).rename("hist_vol")
