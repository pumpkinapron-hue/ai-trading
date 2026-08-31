"""指標テスト全体で共有するフィクスチャ。

ここのフィクスチャは全テストから見える。既存テストと名前が衝突しないことは
実装時に確認済み（`sample_bars` / `multi_day_bars` という名前は他で未使用）。
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """再現可能な合成バー。指標テスト全体で共有する。

    300分（5時間）はNYの取引日1日に収まる長さ。日境界をまたぐ検証（vwap）には
    `multi_day_bars` を使う——このフィクスチャだけでは groupby が1グループにしか
    分かれず、日次リセットが一度も検証されないため。
    """
    rng = np.random.default_rng(20260824)
    n = 300
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    walk = 150.0 + np.cumsum(rng.normal(0, 0.02, n))

    body = {"close_time": open_time + pd.Timedelta(minutes=1)}
    body["bid_close"] = walk
    body["bid_open"] = np.concatenate([[walk[0]], walk[:-1]])
    body["bid_high"] = np.maximum(body["bid_open"], body["bid_close"]) + 0.03
    body["bid_low"] = np.minimum(body["bid_open"], body["bid_close"]) - 0.03
    for f in ("open", "high", "low", "close"):
        body[f"ask_{f}"] = body[f"bid_{f}"] + 0.02
    body["volume"] = rng.uniform(50, 150, n)

    return pd.DataFrame(body, index=open_time).rename_axis("open_time")


@pytest.fixture
def multi_day_bars() -> pd.DataFrame:
    """NYの取引日境界（17:00 America/New_York）を1回またぐ合成バー。

    2026-01-05 00:00Z 起点で2000分（約33時間）。2026-01-04 22:00Z（=NY 17:00、
    1月なので夏時間なし）の境界をまたぎ、日1に1320本・日2に680本が入る
    （実測済み）。平日のみに収まるよう月曜始まりにして週末クローズを避ける。
    """
    rng = np.random.default_rng(20260825)
    n = 2000
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    walk = 150.0 + np.cumsum(rng.normal(0, 0.02, n))

    body = {"close_time": open_time + pd.Timedelta(minutes=1)}
    body["bid_close"] = walk
    body["bid_open"] = np.concatenate([[walk[0]], walk[:-1]])
    body["bid_high"] = np.maximum(body["bid_open"], body["bid_close"]) + 0.03
    body["bid_low"] = np.minimum(body["bid_open"], body["bid_close"]) - 0.03
    for f in ("open", "high", "low", "close"):
        body[f"ask_{f}"] = body[f"bid_{f}"] + 0.02
    body["volume"] = rng.uniform(50, 150, n)

    return pd.DataFrame(body, index=open_time).rename_axis("open_time")
