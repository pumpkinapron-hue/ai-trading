import pandas as pd
import pytest

from aitrading.config import load_settings


@pytest.fixture
def settings():
    return load_settings()


def test_loads_symbol_and_periods(settings):
    assert settings.symbol == "USDJPY"
    assert set(settings.periods) == {"training", "validation", "oos", "forward"}


def test_oos_is_locked_by_default(settings):
    assert settings.period_for("oos").locked is True
    assert settings.period_for("training").locked is False


def test_periods_do_not_overlap(settings):
    ordered = sorted(settings.periods.values(), key=lambda p: p.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.end < later.start, f"{earlier.name} と {later.name} が重なっている"


def test_models_are_configured_not_hardcoded(settings):
    assert settings.models["strategy_generation"] == "claude-fable-5"
    assert settings.models["news_classification"] == "claude-haiku-4-5"


def _bars(index):
    return pd.DataFrame({"bid_close": range(len(index))}, index=index)


def test_slice_bars_returns_only_the_period(settings):
    index = pd.DatetimeIndex(
        pd.to_datetime(["2021-06-01", "2022-06-01", "2024-06-01"], utc=True)
    )
    got = settings.slice_bars(_bars(index), "training")
    assert len(got) == 1
    assert got.index[0] == pd.Timestamp("2021-06-01", tz="UTC")


def test_slice_bars_refuses_locked_period(settings):
    index = pd.DatetimeIndex(pd.to_datetime(["2024-06-01"], utc=True))
    with pytest.raises(PermissionError, match="oos"):
        settings.slice_bars(_bars(index), "oos")


def test_slice_bars_allows_locked_period_when_explicit(settings):
    index = pd.DatetimeIndex(pd.to_datetime(["2024-06-01"], utc=True))
    got = settings.slice_bars(_bars(index), "oos", allow_locked=True)
    assert len(got) == 1
