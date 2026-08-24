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


def test_rejects_ask_below_bid():
    bars = make_bars()
    bars.loc[1, "ask_close"] = bars.loc[1, "bid_close"] - 0.10
    with pytest.raises(ValueError, match="Ask"):
        validate_bars(bars, Timeframe.M1)


def test_sorts_by_open_time():
    bars = make_bars().iloc[::-1].reset_index(drop=True)
    got = validate_bars(bars, Timeframe.M1)
    assert got["open_time"].is_monotonic_increasing
