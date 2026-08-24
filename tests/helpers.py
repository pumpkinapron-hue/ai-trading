"""テスト全体で共有するバー生成ヘルパ。

テスト同士が互いを import し合わないよう、共有物はここに集める。
"""

from __future__ import annotations

import pandas as pd


def make_bars(n: int = 3, tz: str | None = "UTC") -> pd.DataFrame:
    """open_time を列に持つ素のバー（validate_bars に渡す形）。"""
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz=tz)
    body: dict[str, object] = {
        "open_time": open_time,
        "close_time": open_time + pd.Timedelta(minutes=1),
    }
    for side, base in (("bid", 150.0), ("ask", 150.02)):
        for field, bump in (("open", 0.0), ("high", 0.03), ("low", -0.03), ("close", 0.01)):
            body[f"{side}_{field}"] = [base + bump] * n
    body["volume"] = [100.0] * n
    return pd.DataFrame(body)


def minute_bars(start: str, periods: int) -> pd.DataFrame:
    """open_time を index に持つ1分足（Lake.load の返り値と同じ形）。

    価格は1分ごとに +1pip 進む決定的な系列。
    """
    open_time = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    n = len(open_time)
    body: dict[str, object] = {
        "close_time": open_time + pd.Timedelta(minutes=1),
        "bid_open": [150.0 + i * 0.01 for i in range(n)],
        "bid_high": [150.5 + i * 0.01 for i in range(n)],
        "bid_low": [149.5 + i * 0.01 for i in range(n)],
        "bid_close": [150.2 + i * 0.01 for i in range(n)],
        "volume": [10.0] * n,
    }
    for field in ("open", "high", "low", "close"):
        body[f"ask_{field}"] = [v + 0.02 for v in body[f"bid_{field}"]]
    return pd.DataFrame(body, index=open_time).rename_axis("open_time")
