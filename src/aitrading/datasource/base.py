"""バーの共通スキーマ。データソースが何であれ上位層が見る形はこれ1つ。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from aitrading.timeutil import Timeframe, ensure_utc

TIME_COLUMNS = ["open_time", "close_time"]

PRICE_COLUMNS = [
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
]

BAR_COLUMNS = TIME_COLUMNS + PRICE_COLUMNS + ["volume"]


class BarSource(Protocol):
    """市場データの取得元。Dukascopy / OANDA / MT5 はすべてこれを実装する。"""

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """[start, end) のバーを BAR_COLUMNS のスキーマで返す。"""
        ...


def validate_bars(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """スキーマを検証して正規化する。違反は握りつぶさず ValueError にする。

    ここを緩めると、壊れたデータが静かにレイクに入って後段すべてを汚染する。
    """
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"列が足りない: {missing}")

    out = df.loc[:, BAR_COLUMNS].copy()

    for column in TIME_COLUMNS:
        index = pd.DatetimeIndex(out[column])
        if index.tz is None:
            raise ValueError(f"{column} が tz-aware でない。UTCで渡すこと")
        out[column] = index.tz_convert("UTC")

    out = out.sort_values("open_time").reset_index(drop=True)

    if out["open_time"].duplicated().any():
        dupes = out.loc[out["open_time"].duplicated(), "open_time"].tolist()
        raise ValueError(f"open_time が重複している: {dupes[:5]}")

    delta = timeframe.delta
    if delta is not None:
        bad = out["close_time"] - out["open_time"] != delta
        if bad.any():
            raise ValueError(
                f"close_time が open_time + {delta} になっていない行が {int(bad.sum())} 件ある"
            )

    for column in PRICE_COLUMNS + ["volume"]:
        out[column] = out[column].astype("float64")

    crossed = out["ask_close"] < out["bid_close"]
    if crossed.any():
        raise ValueError(f"Ask が Bid を下回る行が {int(crossed.sum())} 件ある")

    return out
