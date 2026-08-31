"""指標レジストリ。

登録された指標には、テスト側からトランケーション不変性検査が自動で適用される
（`tests/test_indicators_lookahead.py`）。新しい指標を足したとき、テストを
書き忘れても先読み検査だけは必ず走るのが狙い。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

Indicator = Callable[..., "pd.Series | pd.DataFrame"]

INDICATORS: dict[str, Indicator] = {}


def indicator(name: str) -> Callable[[Indicator], Indicator]:
    def register(fn: Indicator) -> Indicator:
        if name in INDICATORS:
            raise ValueError(f"指標名が重複している: {name!r}")
        INDICATORS[name] = fn
        return fn

    return register


def mid(bars: pd.DataFrame, field: str) -> pd.Series:
    """Bid と Ask の中間値。指標はミッドで計算し、コストは約定側で扱う。"""
    return (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0
