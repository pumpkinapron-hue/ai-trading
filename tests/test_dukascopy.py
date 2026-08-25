from datetime import datetime

import pandas as pd
import pytest

from aitrading.datasource.base import BAR_COLUMNS
from aitrading.datasource.dukascopy import DukascopySource, normalize
from aitrading.timeutil import Timeframe


def raw_side(base: float) -> pd.DataFrame:
    """dukascopy-python が返す形（timestamp index + OHLCV）を模した生データ。"""
    index = pd.date_range("2026-01-05 00:00", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [base, base + 0.01, base + 0.02],
            "high": [base + 0.03] * 3,
            "low": [base - 0.03] * 3,
            "close": [base + 0.01] * 3,
            "volume": [100.0, 110.0, 120.0],
        },
        index=index,
    )


def test_normalize_produces_schema():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert list(got.columns) == BAR_COLUMNS
    assert len(got) == 3


def test_normalize_maps_bid_and_ask_separately():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert got.loc[0, "bid_open"] == pytest.approx(150.00)
    assert got.loc[0, "ask_open"] == pytest.approx(150.02)


def test_normalize_sets_close_time():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert got.loc[0, "close_time"] - got.loc[0, "open_time"] == pd.Timedelta(minutes=1)


def test_normalize_takes_volume_from_bid_side():
    # bid と ask で volume をわざと変える。brief 原文は raw_side(base) をそのまま
    # 両側に使っており、両側の volume が常に等しいため「ask 側を取ってしまう」
    # 退行があってもこのテストは検知できない。ask 側だけ上書きして実際に
    # bid 由来であることを判別できるようにする。
    bid = raw_side(150.00)
    ask = raw_side(150.02)
    ask["volume"] = [999.0, 999.0, 999.0]
    got = normalize(bid, ask, Timeframe.M1)
    assert got["volume"].tolist() == [100.0, 110.0, 120.0]


def test_normalize_drops_rows_missing_on_one_side():
    bid = raw_side(150.00)
    ask = raw_side(150.02).iloc[1:]
    got = normalize(bid, ask, Timeframe.M1)
    assert len(got) == 2


def test_normalize_rejects_naive_index():
    # Global Constraints: naive な入力は ValueError。normalize() は bid/ask
    # 2枚を受け取るため、メッセージからどちら側が悪いか判別できることも確認する。
    naive_bid = raw_side(150.00)
    naive_bid.index = naive_bid.index.tz_localize(None)
    with pytest.raises(ValueError, match="bid"):
        normalize(naive_bid, raw_side(150.02), Timeframe.M1)

    naive_ask = raw_side(150.02)
    naive_ask.index = naive_ask.index.tz_localize(None)
    with pytest.raises(ValueError, match="ask"):
        normalize(raw_side(150.00), naive_ask, Timeframe.M1)


def test_normalize_converts_non_utc_tz_aware_index_to_utc():
    # naive はエラーになる一方で、UTC以外の tz-aware な入力は引き続き正しく
    # UTC に変換されること（tz変換そのものは壊していないことの確認）。
    bid = raw_side(150.00)
    bid.index = bid.index.tz_convert("Asia/Tokyo")
    got = normalize(bid, raw_side(150.02), Timeframe.M1)
    assert got.loc[0, "open_time"] == pd.Timestamp("2026-01-05 00:00", tz="UTC")


def test_fetch_rejects_naive_start():
    # Global Constraints: 「naive な入力は ValueError」。fetch() の start/end は
    # 呼び出し側から渡される境界であり、normalize() が処理するライブラリの生
    # データ（UTCと分かっているので localize する）とは信頼の前提が違う。
    source = DukascopySource()
    with pytest.raises(ValueError, match="tz-aware"):
        source.fetch(
            "USDJPY",
            Timeframe.M1,
            datetime(2026, 1, 5),
            pd.Timestamp("2026-01-05 01:00", tz="UTC"),
        )


def test_fetch_rejects_naive_end():
    source = DukascopySource()
    with pytest.raises(ValueError, match="tz-aware"):
        source.fetch(
            "USDJPY",
            Timeframe.M1,
            pd.Timestamp("2026-01-05 00:00", tz="UTC"),
            datetime(2026, 1, 5, 1, 0),
        )


def test_fetch_rejects_unsupported_symbol():
    source = DukascopySource()
    with pytest.raises(ValueError, match="シンボル"):
        source.fetch(
            "EURUSD",
            Timeframe.M1,
            pd.Timestamp("2026-01-05 00:00", tz="UTC"),
            pd.Timestamp("2026-01-05 01:00", tz="UTC"),
        )


def test_fetch_rejects_daily_timeframe():
    # 日足・週足は Dukascopy から直接は取らない設計（1分足から生成する）。
    # _INTERVAL_NAMES に無いキーで素の KeyError が漏れるのではなく、
    # 他の未対応入力と同じ ValueError に揃える。
    source = DukascopySource()
    with pytest.raises(ValueError, match="timeframe"):
        source.fetch(
            "USDJPY",
            Timeframe.D1_NY,
            pd.Timestamp("2026-01-05 00:00", tz="UTC"),
            pd.Timestamp("2026-01-06 00:00", tz="UTC"),
        )


@pytest.mark.network
def test_generated_5m_matches_dukascopy_5m():
    """1分足から生成した5分足が、配信元の5分足と一致するか。

    設計§11の要求。ここが合わないと、リサンプルの規約か集約ロジックが
    配信元とずれている。保存対象は1分足だけなので、この照合専用に5分足を取る。
    """
    from aitrading.bars import resample

    source = DukascopySource()
    start = pd.Timestamp("2026-01-05 00:00", tz="UTC")
    end = pd.Timestamp("2026-01-05 04:00", tz="UTC")

    m1 = source.fetch("USDJPY", Timeframe.M1, start, end).set_index("open_time")
    m5_direct = source.fetch("USDJPY", Timeframe.M5, start, end).set_index("open_time")
    m5_derived = resample(m1, Timeframe.M5)

    common = m5_direct.index.intersection(m5_derived.index)
    assert len(common) > 10, "照合できるバーが少なすぎる"
    for column in ("bid_open", "bid_high", "bid_low", "bid_close"):
        pd.testing.assert_series_equal(
            m5_derived.loc[common, column],
            m5_direct.loc[common, column],
            check_names=False,
            rtol=0,
            atol=1e-6,
        )


@pytest.mark.network
def test_fetch_real_data():
    """実サーバーに触る唯一のテスト。既定では -m 'not network' で除外される。"""
    source = DukascopySource()
    got = source.fetch(
        "USDJPY",
        Timeframe.M1,
        pd.Timestamp("2026-01-05 00:00", tz="UTC"),
        pd.Timestamp("2026-01-05 01:00", tz="UTC"),
    )
    assert not got.empty
    assert list(got.columns) == BAR_COLUMNS
    assert (got["ask_close"] >= got["bid_close"]).all()
