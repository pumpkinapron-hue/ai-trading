"""バーの共通スキーマ。データソースが何であれ上位層が見る形はこれ1つ。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np
import pandas as pd

from aitrading.timeutil import Timeframe, ensure_utc

TIME_COLUMNS = ["open_time", "close_time"]

PRICE_COLUMNS = [
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
]

BAR_COLUMNS = TIME_COLUMNS + PRICE_COLUMNS + ["volume"]

#: 可変長期間（日足・週足）1本あたりの最大長。週足＋夏時間切り替えの余裕を見て8日。
#: これを超えるのは集約ミスを疑う。
MAX_VARIABLE_BAR_SPAN = pd.Timedelta(days=8)


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
        try:
            out[column] = ensure_utc(pd.DatetimeIndex(out[column]))
        except ValueError as exc:
            raise ValueError(f"{column} が tz-aware でない。UTCで渡すこと") from exc

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
    else:
        span = out["close_time"] - out["open_time"]
        non_positive = span <= pd.Timedelta(0)
        if non_positive.any():
            raise ValueError(
                f"close_time が open_time 以下の行が {int(non_positive.sum())} 件ある"
            )
        too_long = span > MAX_VARIABLE_BAR_SPAN
        if too_long.any():
            raise ValueError(
                f"close_time - open_time が {MAX_VARIABLE_BAR_SPAN} を超える行が"
                f" {int(too_long.sum())} 件ある"
            )

    # 時間軸に関わらず、バー同士は重なってはいけない（重複 open_time だけでは
    # 期間がまたがる壊れ方を捕まえられない）。ソート済み前提で隣接行だけ見ればよい。
    overlap = out["close_time"] > out["open_time"].shift(-1)
    if overlap.any():
        raise ValueError(f"バーが重なっている行が {int(overlap.sum())} 件ある")

    value_columns = PRICE_COLUMNS + ["volume"]
    for column in value_columns:
        out[column] = out[column].astype("float64")

    finite = np.isfinite(out[value_columns].to_numpy())
    bad_rows = ~finite.all(axis=1)
    if bad_rows.any():
        raise ValueError(
            f"価格または出来高に NaN/inf を含む行が {int(bad_rows.sum())} 件ある"
        )

    for field in ("open", "high", "low", "close"):
        crossed = out[f"ask_{field}"] < out[f"bid_{field}"]
        if crossed.any():
            raise ValueError(
                f"Ask が Bid を {field} で下回る行が {int(crossed.sum())} 件ある"
            )

    return out
