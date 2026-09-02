import pandas as pd
import pytest

from aitrading.timeutil import (
    Session,
    Timeframe,
    ensure_utc,
    fx_holidays,
    is_fx_holiday,
    is_market_open,
    local_trading_date,
    session_labels,
    trading_day_label,
    trading_day_start,
    trading_period_start,
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


# --- 夏時間の切り替え日 ---
#
# ここが無かったせいで「区切りが1時間ずれる／取引日が1本消える」バグが
# リサンプルまで素通りした。America/New_York を固定オフセットに差し替えても
# 全テストが通ってしまう状態だったので、切り替え日そのものを直接押さえる。


@pytest.mark.parametrize(
    ("transition", "before_tz", "after_tz"),
    [
        ("2026-03-08", "EST", "EDT"),  # 春: 23時間の日
        ("2026-11-01", "EDT", "EST"),  # 秋: 25時間の日
    ],
)
def test_trading_day_start_switches_exactly_at_ny_1700_on_dst_days(
    transition, before_tz, after_tz
):
    """切り替え日でも区切りはローカル17:00ちょうど。1分前は前の取引日に属する。"""
    day = pd.Timestamp(transition)
    boundary = (
        pd.DatetimeIndex([day + pd.Timedelta(hours=17)])
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    assert boundary[0].tz_convert("America/New_York").strftime("%Z") == after_tz

    # 区切りちょうどは新しい取引日の開始そのもの
    assert trading_day_start(boundary, "ny")[0] == boundary[0]
    # その1分前はまだ前の取引日（＝前日の17:00が開始）
    previous = trading_day_start(boundary - pd.Timedelta(minutes=1), "ny")[0]
    assert previous < boundary[0]
    assert previous.tz_convert("America/New_York").hour == 17
    assert previous.tz_convert("America/New_York").strftime("%Z") == before_tz


@pytest.mark.parametrize("transition", ["2026-03-08", "2026-11-01"])
def test_trading_day_label_advances_across_dst_transition(transition):
    """切り替え日の日曜17:00に始まる取引日は「月曜」。ラベルが日曜のままだと
    月曜ぶんが同じ足に飲み込まれ、24時間ぶんの先読みになる（実際に起きていた）。"""
    day = pd.Timestamp(transition)
    boundary = (
        pd.DatetimeIndex([day + pd.Timedelta(hours=17)])
        .tz_localize("America/New_York")
        .tz_convert("UTC")
    )
    assert boundary[0].tz_convert("America/New_York").dayofweek == 6  # 日曜

    before = trading_day_label(boundary - pd.Timedelta(minutes=1), "ny")[0]
    after = trading_day_label(boundary, "ny")[0]
    assert after != before
    assert after.tz_convert("America/New_York").dayofweek == 0  # 月曜


def test_trading_day_start_is_idempotent_across_a_full_dst_year():
    """取引日の開始時刻をもう一度 trading_day_start に通しても動かない。

    「開始時刻がその取引日に属していない」＝区切りがズレている、を年単位で検出する。
    """
    dense = pd.date_range("2026-01-01", "2027-01-01", freq="37min", tz="UTC")
    for convention in ("ny", "jst"):
        starts = trading_day_start(dense, convention)
        again = trading_day_start(starts, convention)
        assert (starts == again).all(), convention


def test_trading_day_start_spans_are_23_24_or_25_hours():
    """取引日の長さは夏時間の切り替え日を除いて24時間ちょうど。
    切り替え日だけ23時間/25時間になり、それ以外の長さは出ない。"""
    dense = pd.date_range("2026-01-01", "2027-01-01", freq="17min", tz="UTC")
    starts = pd.DatetimeIndex(trading_day_start(dense, "ny")).unique().sort_values()
    spans = starts.to_series().diff().dropna()
    assert set(spans.unique()) == {
        pd.Timedelta(hours=23),
        pd.Timedelta(hours=24),
        pd.Timedelta(hours=25),
    }
    assert (spans == pd.Timedelta(hours=23)).sum() == 1
    assert (spans == pd.Timedelta(hours=25)).sum() == 1

    # JSTは夏時間が無いので常に24時間
    jst = pd.DatetimeIndex(trading_day_start(dense, "jst")).unique().sort_values()
    assert set(jst.to_series().diff().dropna().unique()) == {pd.Timedelta(hours=24)}


@pytest.mark.parametrize("convention", ["ny", "jst"])
def test_group_key_and_boundary_agree_across_a_dst_year(convention):
    """グループ化の経路と境界計算の経路が同じ答えを出すこと。

    bars.resample は「どの足に入れるか」を trading_day_label 経由で、「その足がいつ
    始まりいつ終わるか」を _period_start で決めている。この2つが食い違うと、
    足の中身と close_time がズレる（＝先読み）。夏時間をまたぐ1年ぶんで突き合わせる。
    """
    dense = pd.date_range("2026-01-01", "2027-01-01", freq="41min", tz="UTC")
    round_trip = trading_period_start(local_trading_date(dense, convention), convention)
    assert (pd.DatetimeIndex(round_trip) == trading_day_start(dense, convention)).all()


# --- 祝日クローズ ---
#
# 入れる前は、祝日が丸1日(1440分)の欠損として毎回報告されていた。10年で20件前後に
# なり、longest_gap_minutes がその値に貼り付いて品質アラートが使い物にならなくなる。
# （実データ2021年1月の取得で「最長 1320 分」と出ていたのは元日の休場だった。）


@pytest.mark.parametrize(
    ("year", "good_friday"),
    [
        (2021, "2021-04-02"),
        (2024, "2024-03-29"),
        (2025, "2025-04-18"),
        (2026, "2026-04-03"),
    ],
)
def test_good_friday_is_computed_not_tabulated(year, good_friday):
    """グッドフライデーは復活祭から計算する。

    年ごとに表へ書き足す方式にすると「今年ぶんを足し忘れて、その年だけ祝日が
    休場扱いされない」という、静かで気づきにくい壊れ方をする。
    """
    holidays = sorted(fx_holidays(year))
    assert pd.Timestamp(good_friday) in holidays


def test_good_friday_is_always_a_friday():
    """計算式の健全性。名前のとおり金曜でなければ式が間違っている。"""
    for year in range(2015, 2036):
        friday = sorted(fx_holidays(year))[1]
        assert friday.dayofweek == 4, f"{year}: {friday} が金曜でない"


def test_holidays_are_exactly_three_per_year():
    """恣意的に増やさない。基準は「主要な流動性供給者が揃って休むか」であって
    「どこかの国の祝日か」ではない。感謝祭や独立記念日はNYが薄くなるだけで
    市場自体は開いており、休場扱いにすると本物の欠損を隠すことになる。"""
    for year in (2015, 2020, 2026, 2030):
        assert len(fx_holidays(year)) == 3


@pytest.mark.parametrize(
    ("moment", "expected_open"),
    [
        ("2021-01-01 12:00Z", False),  # 元日
        ("2021-12-24 12:00Z", True),   # クリスマスイブは通常の取引日
        ("2021-12-25 12:00Z", False),  # クリスマス（この年は土曜でもある）
        ("2021-04-02 12:00Z", False),  # グッドフライデー
        ("2021-04-01 12:00Z", True),   # その前日
        ("2024-03-29 12:00Z", False),  # グッドフライデー（別の年）
        ("2024-12-25 12:00Z", False),  # クリスマス（水曜）
    ],
)
def test_is_market_open_excludes_holidays(moment, expected_open):
    assert bool(is_market_open(idx(moment)).iloc[0]) is expected_open


def test_holiday_closure_is_judged_on_the_new_york_calendar_date():
    """判定の単位は NY のローカル暦日（trading_day_start の NY基準と同じ土俵）。

    近似であることは承知のうえ。目的は「毎年の偽陽性で品質アラートが使い物に
    ならなくなる」のを防ぐことで、休場時刻を1分単位で当てることではない。
    """
    # 2024-12-25 の NY 00:30 は UTC では 12-25 05:30
    assert not is_market_open(idx("2024-12-25 05:30Z")).iloc[0]
    # 同じ UTC 日付でも NY ではまだ 12-24 の 19:00 なので開いている
    assert is_market_open(idx("2024-12-25 00:00Z")).iloc[0]


def test_holiday_lookup_handles_an_empty_index():
    assert is_fx_holiday(pd.DatetimeIndex([], tz="UTC")).empty
