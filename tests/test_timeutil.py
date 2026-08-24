import pandas as pd
import pytest

from aitrading.timeutil import (
    Session,
    Timeframe,
    ensure_utc,
    is_market_open,
    session_labels,
    trading_day_start,
)


def idx(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(stamps), utc=True))


# 注: session_labels / is_market_open は pd.Series を返すため、先頭要素の取得は
# .iloc[0] を使う（素の [0] は pandas 2.x で FutureWarning、3.0 以降は KeyError）。
# trading_day_start / trading_day_label は pd.DatetimeIndex を返すので [0] のままでよい。


def test_timeframe_delta():
    assert Timeframe.M5.delta == pd.Timedelta(minutes=5)
    assert Timeframe.H4.delta == pd.Timedelta(hours=4)
    assert Timeframe.D1_NY.delta is None


def test_ensure_utc_rejects_naive():
    naive = pd.DatetimeIndex(pd.to_datetime(["2026-01-05 00:00"]))
    with pytest.raises(ValueError, match="tz-aware"):
        ensure_utc(naive)


def test_ensure_utc_converts_other_zone():
    tokyo = pd.DatetimeIndex(pd.to_datetime(["2026-01-05 09:00"])).tz_localize("Asia/Tokyo")
    assert ensure_utc(tokyo)[0] == pd.Timestamp("2026-01-05 00:00", tz="UTC")


def test_session_labels_tokyo():
    # 2026-01-05 は月曜。JST 10:00 = UTC 01:00 は東京単独。
    assert session_labels(idx("2026-01-05 01:00Z")).iloc[0] == Session.TOKYO


def test_session_labels_overlap_follows_dst():
    # 冬（EST/GMT）: NY 09:00 = UTC 14:00、ロンドンは GMT 14:00 で 16:30 まで開いている
    assert session_labels(idx("2026-01-05 14:00Z")).iloc[0] == Session.LDN_NY_OVERLAP
    # 夏（EDT/BST）: NY 09:00 = UTC 13:00。固定オフセットで書いていると外れる
    assert session_labels(idx("2026-07-06 13:00Z")).iloc[0] == Session.LDN_NY_OVERLAP


def test_session_labels_off_hours():
    # NY冬季クローズ 17:00 EST = 22:00 UTC。JST 08:00 = UTC 23:00(前日)は
    # NYクローズ後・東京オープン(9:00 JST)前の薄商い帯。
    # 注: ブリーフ原文は "21:00Z" だったが、その時刻はNY 16:00 EST でまだ
    # NEWYORKセッション中（コード上のNY窓は8:00-17:00 local）のため
    # Session.OFF を検証できなかった。test_market_closed_on_weekend が
    # 同じ冬季NYクローズを22:00Zとしているのと整合させ、23:00Zに修正した。
    assert session_labels(idx("2026-01-05 23:00Z")).iloc[0] == Session.OFF


def test_market_closed_on_weekend():
    # 土曜はクローズ
    assert not is_market_open(idx("2026-01-10 12:00Z")).iloc[0]
    # 金曜 NY 17:00 EST = 22:00 UTC 以降はクローズ
    assert not is_market_open(idx("2026-01-09 22:30Z")).iloc[0]
    assert is_market_open(idx("2026-01-09 20:00Z")).iloc[0]
    # 日曜 NY 17:00 EST = 22:00 UTC 以降はオープン
    assert is_market_open(idx("2026-01-11 23:00Z")).iloc[0]


def test_trading_day_start_ny_winter():
    # 冬: NY 17:00 = 22:00 UTC。2026-01-06 01:00Z は 2026-01-05 22:00Z 始まりの日に属する
    got = trading_day_start(idx("2026-01-06 01:00Z"), "ny")[0]
    assert got == pd.Timestamp("2026-01-05 22:00", tz="UTC")


def test_trading_day_start_ny_summer():
    # 夏: NY 17:00 = 21:00 UTC
    got = trading_day_start(idx("2026-07-07 01:00Z"), "ny")[0]
    assert got == pd.Timestamp("2026-07-06 21:00", tz="UTC")


def test_trading_day_start_jst_has_no_dst():
    # JST 00:00 = 前日 15:00 UTC。夏でも冬でも同じ
    assert trading_day_start(idx("2026-01-06 01:00Z"), "jst")[0] == pd.Timestamp(
        "2026-01-05 15:00", tz="UTC"
    )
    assert trading_day_start(idx("2026-07-07 01:00Z"), "jst")[0] == pd.Timestamp(
        "2026-07-06 15:00", tz="UTC"
    )


def test_ny_and_jst_day_boundaries_differ():
    # 同じ瞬間が、NY基準とJST基準で別の日に属することがある
    ts = idx("2026-01-06 01:00Z")
    assert trading_day_start(ts, "ny")[0] != trading_day_start(ts, "jst")[0]


def test_trading_day_label_treats_sunday_open_as_monday():
    from aitrading.timeutil import trading_day_label

    # 日曜 22:00Z（NY日曜17:00）に始まる足は「月曜」の取引日
    got = trading_day_label(idx("2026-01-11 23:00Z"), "ny")[0]
    assert got.tz_convert("America/New_York").dayofweek == 0


def test_trading_day_label_differs_between_conventions():
    from aitrading.timeutil import trading_day_label

    ts = idx("2026-01-06 01:00Z")
    assert trading_day_label(ts, "ny")[0] != trading_day_label(ts, "jst")[0]
