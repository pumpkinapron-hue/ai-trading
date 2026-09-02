"""Dukascopy からの取得。外部サービスに触るのはこのファイルだけ。

ライブラリの都合はすべてここに閉じ込める。dukascopy-python が不安定なら
.bi5 の直接ダウンロードに差し替えるが、normalize より上は影響を受けない。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from aitrading.datasource.base import BAR_COLUMNS, validate_bars
from aitrading.timeutil import Timeframe, ensure_utc_timestamp, ensure_utc

#: 内部の Timeframe → dukascopy-python の interval 定数名
#: dukascopy-python==4.0.1 で実物確認済み（すべて dir(dukascopy_python) に存在）。
_INTERVAL_NAMES = {
    Timeframe.M1: "INTERVAL_MIN_1",
    Timeframe.M5: "INTERVAL_MIN_5",
    Timeframe.M15: "INTERVAL_MIN_15",
    Timeframe.H1: "INTERVAL_HOUR_1",
    Timeframe.H4: "INTERVAL_HOUR_4",
}


def _as_utc_index(frame: pd.DataFrame, side: str) -> pd.DatetimeIndex:
    """frame の index を UTC の DatetimeIndex にする。

    naive な index は ValueError にする（Global Constraints）。ensure_utc に
    委譲し、bid/ask のどちら側が悪かったか呼び出し側が分かるよう、側名を
    メッセージに足して re-raise する（normalize() は2枚のフレームを受け取る
    ため、側が分からないと直せない）。
    """
    try:
        return ensure_utc(pd.DatetimeIndex(frame.index))
    except ValueError as exc:
        raise ValueError(f"{side} 側の index が tz-aware でない。UTCで渡すこと") from exc


def normalize(bid: pd.DataFrame, ask: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """ライブラリの生出力（bid側・ask側の2枚）を共通スキーマ1枚にする。

    片側にしか無い時刻は落とす。片側だけのバーはスプレッドが計算できず、
    約定モデルに使えないため。
    """
    delta = timeframe.delta
    if delta is None:
        raise ValueError(f"{timeframe} は取得対象ではない（1分足から生成する）")

    bid = bid.copy()
    ask = ask.copy()
    bid.index = _as_utc_index(bid, "bid")
    ask.index = _as_utc_index(ask, "ask")

    common = bid.index.intersection(ask.index).sort_values()
    bid = bid.loc[common]
    ask = ask.loc[common]

    body = {"open_time": common, "close_time": common + delta}
    for side, frame in (("bid", bid), ("ask", ask)):
        for field in ("open", "high", "low", "close"):
            body[f"{side}_{field}"] = frame[field].to_numpy()
    body["volume"] = bid["volume"].to_numpy()

    return validate_bars(pd.DataFrame(body, columns=BAR_COLUMNS), timeframe)


class DukascopySource:
    """BarSource の Dukascopy 実装。"""

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        import dukascopy_python
        from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_USD_JPY

        instruments = {"USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY}
        if symbol not in instruments:
            raise ValueError(f"未対応のシンボル: {symbol!r}")
        if timeframe not in _INTERVAL_NAMES:
            raise ValueError(
                f"未対応の timeframe: {timeframe!r}。Dukascopy から直接取得できる"
                "のは M1/M5/M15/H1/H4 のみ（日足・週足は1分足から生成する）"
            )
        start_ts = ensure_utc_timestamp(start, "start")
        end_ts = ensure_utc_timestamp(end, "end")

        interval = getattr(dukascopy_python, _INTERVAL_NAMES[timeframe])
        sides = {}
        for name, offer_side in (
            ("bid", dukascopy_python.OFFER_SIDE_BID),
            ("ask", dukascopy_python.OFFER_SIDE_ASK),
        ):
            sides[name] = dukascopy_python.fetch(
                instrument=instruments[symbol],
                interval=interval,
                offer_side=offer_side,
                start=start_ts.to_pydatetime(),
                end=end_ts.to_pydatetime(),
            )

        return normalize(sides["bid"], sides["ask"], timeframe)
