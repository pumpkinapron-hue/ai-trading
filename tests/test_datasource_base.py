import pandas as pd
import pytest

from aitrading.datasource.base import BAR_COLUMNS, validate_bars
from aitrading.timeutil import Timeframe

from tests.helpers import make_bars


def test_accepts_valid_bars():
    got = validate_bars(make_bars(), Timeframe.M1)
    assert list(got.columns) == BAR_COLUMNS
    assert got["bid_open"].dtype == "float64"


def test_rejects_naive_timestamps():
    with pytest.raises(ValueError, match="tz-aware"):
        validate_bars(make_bars(tz=None), Timeframe.M1)


def test_converts_non_utc_tz_to_utc():
    bars = make_bars(tz="Asia/Tokyo")
    got = validate_bars(bars, Timeframe.M1)
    assert got["open_time"].iloc[0] == pd.Timestamp("2026-01-04 15:00", tz="UTC")


def test_rejects_missing_column():
    bars = make_bars().drop(columns=["ask_high"])
    with pytest.raises(ValueError, match="ask_high"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_wrong_close_time():
    bars = make_bars()
    bars.loc[1, "close_time"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="close_time"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_duplicate_open_time():
    bars = make_bars()
    bars.loc[1, "open_time"] = bars.loc[0, "open_time"]
    bars.loc[1, "close_time"] = bars.loc[0, "close_time"]
    with pytest.raises(ValueError, match="重複"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_ask_below_bid_open():
    bars = make_bars()
    bars.loc[1, "ask_open"] = bars.loc[1, "bid_open"] - 0.10
    with pytest.raises(ValueError, match="Ask"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_ask_below_bid_high():
    bars = make_bars()
    bars.loc[1, "ask_high"] = bars.loc[1, "bid_high"] - 0.10
    with pytest.raises(ValueError, match="Ask"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_ask_below_bid_low():
    bars = make_bars()
    bars.loc[1, "ask_low"] = bars.loc[1, "bid_low"] - 0.10
    with pytest.raises(ValueError, match="Ask"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_ask_below_bid_close():
    bars = make_bars()
    bars.loc[1, "ask_close"] = bars.loc[1, "bid_close"] - 0.10
    with pytest.raises(ValueError, match="Ask"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_nan_price():
    bars = make_bars()
    bars.loc[1, "bid_close"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_inf_price():
    bars = make_bars()
    bars.loc[1, "ask_high"] = float("inf")
    with pytest.raises(ValueError, match="inf"):
        validate_bars(bars, Timeframe.M1)


def test_sorts_by_open_time():
    bars = make_bars().iloc[::-1].reset_index(drop=True)
    got = validate_bars(bars, Timeframe.M1)
    assert got["open_time"].is_monotonic_increasing


def test_rejects_zero_length_daily_bar():
    bars = make_bars()
    bars.loc[1, "close_time"] = bars.loc[1, "open_time"]
    with pytest.raises(ValueError, match="close_time"):
        validate_bars(bars, Timeframe.D1_NY)


def test_rejects_too_long_daily_bar():
    bars = make_bars()
    bars.loc[1, "close_time"] = bars.loc[1, "open_time"] + pd.Timedelta(days=9)
    with pytest.raises(ValueError, match="close_time"):
        validate_bars(bars, Timeframe.D1_NY)


def test_rejects_overlapping_bars():
    bars = make_bars()
    bars.loc[0, "close_time"] = bars.loc[1, "open_time"] + pd.Timedelta(minutes=1)
    with pytest.raises(ValueError, match="重な"):
        validate_bars(bars, Timeframe.D1_NY)
