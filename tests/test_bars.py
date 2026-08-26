import pandas as pd
import pytest

from aitrading.bars import resample
from aitrading.datasource.base import validate_bars
from aitrading.timeutil import Timeframe, is_market_open

from tests.helpers import minute_bars


def test_m5_aggregates_ohlc_correctly():
    got = resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M5)
    assert len(got) == 2
    first = got.iloc[0]
    assert first["bid_open"] == pytest.approx(150.00)   # 1本目の open
    assert first["bid_close"] == pytest.approx(150.24)  # 5本目の close
    assert first["bid_high"] == pytest.approx(150.54)   # 5本の最大
    assert first["bid_low"] == pytest.approx(149.50)    # 5本の最小
    assert first["volume"] == pytest.approx(50.0)


def test_close_time_is_end_of_period():
    """5分足が使えるようになるのは5分が終わった瞬間。ここを誤ると先読みになる。"""
    got = resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M5)
    assert got.index[0] == pd.Timestamp("2026-01-05 00:00", tz="UTC")
    assert got.iloc[0]["close_time"] == pd.Timestamp("2026-01-05 00:05", tz="UTC")


def test_drops_incomplete_trailing_period():
    """埋まりきっていない最後の期間は返さない(第1層: 確定足しか出さない)。"""
    got = resample(minute_bars("2026-01-05 00:00", 7), Timeframe.M5)
    assert len(got) == 1


def test_h1_and_h4_aggregate():
    bars = minute_bars("2026-01-05 00:00", 60 * 8)
    assert len(resample(bars, Timeframe.H1)) == 8
    assert len(resample(bars, Timeframe.H4)) == 2


def test_daily_ny_and_jst_boundaries_differ():
    """同じ1分足から2通りの日足ができ、区切りが違う。"""
    bars = minute_bars("2026-01-05 00:00", 60 * 48)
    ny = resample(bars, Timeframe.D1_NY)
    jst = resample(bars, Timeframe.D1_JST)
    assert set(ny.index) != set(jst.index)


def test_daily_ny_boundary_is_22utc_in_winter():
    bars = minute_bars("2026-01-05 00:00", 60 * 48)
    ny = resample(bars, Timeframe.D1_NY)
    assert ny.index[0].hour == 22  # 冬時間の NY 17:00


def test_daily_jst_boundary_is_15utc():
    bars = minute_bars("2026-01-05 00:00", 60 * 48)
    jst = resample(bars, Timeframe.D1_JST)
    assert jst.index[0].hour == 15  # JST 00:00


def test_weekly_ny_and_jst_differ():
    """週足も2系統。同じ実装を共有しつつ、区切りは別になる。"""
    bars = minute_bars("2026-01-05 00:00", 60 * 24 * 12)
    ny = resample(bars, Timeframe.W1_NY)
    jst = resample(bars, Timeframe.W1_JST)
    assert not ny.empty and not jst.empty
    assert set(ny.index) != set(jst.index)


def test_weekly_groups_a_full_week_into_one_bar():
    # 月曜0:00Zから12日分。ちょうど1週間ぶんが確定し、残りは翌週の未確定分として落ちる
    bars = minute_bars("2026-01-05 00:00", 60 * 24 * 12)
    jst = resample(bars, Timeframe.W1_JST)
    assert len(jst) >= 1
    assert (jst.index.to_series().diff().dropna() >= pd.Timedelta(days=6)).all()


def test_resample_is_deterministic():
    """同じ入力から必ず同じ出力。1分足から再生成すれば同じものができる。"""
    bars = minute_bars("2026-01-05 00:00", 100)
    pd.testing.assert_frame_equal(
        resample(bars, Timeframe.M15), resample(bars, Timeframe.M15)
    )


def test_rejects_m1_as_target():
    with pytest.raises(ValueError, match="1m"):
        resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M1)


# --- ここから先はブリーフに無い追加テスト。理由は task-7-report.md に記載 ---


def test_rejects_naive_index():
    """全時刻UTC tz-aware というグローバル制約を resample 自身も守っていることを確認する。"""
    naive = minute_bars("2026-01-05 00:00", 5)
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        resample(naive, Timeframe.M5)


def test_empty_input_returns_empty_frame_with_consistent_columns():
    """Lake.load は未取得シンボルに対して0行のフレームを返しうる。列の並びは非空の場合と揃える。"""
    empty = minute_bars("2026-01-05 00:00", 0)
    got = resample(empty, Timeframe.M5)
    assert got.empty
    non_empty_columns = list(resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M5).columns)
    assert list(got.columns) == non_empty_columns
    assert got.index.name == "open_time"


def test_daily_drops_when_data_covers_only_one_trading_day():
    """可変長分岐版の『確定足しか出さない』テスト。取引日が1つしか無ければ確定させようがない。"""
    bars = minute_bars("2026-01-05 00:00", 60 * 5)  # 月曜の5時間ぶんだけ。NY・JSTどちらの取引日も1つだけ
    assert resample(bars, Timeframe.D1_NY).empty
    assert resample(bars, Timeframe.D1_JST).empty


def test_weekly_jst_does_not_bleed_monday_into_previous_week():
    """W-SUN でグループ化する際、月曜の分足を前週に混ぜてしまうバグの回帰テスト。

    trading_day_label は「区切り時刻のローカル真夜中」をUTC表現で返す。JSTはUTCより
    進んでいるため、素朴に tz_localize(None) で日付を読むと月曜のラベルが日曜(UTC表現の
    日付)に見えてしまい、W-SUN 週境界をまたいで前週に混ざる。ローカルタイムゾーンに
    変換してから日付を読まないと再発する。
    """
    bars = minute_bars("2026-01-05 00:00", 60 * 24 * 12)  # 月曜2026-01-05 00:00Zから12日
    jst = resample(bars, Timeframe.W1_JST)
    assert len(jst) == 1
    # 週の開始は月曜 JST 00:00(= 2026-01-04 15:00 UTC)そのもの。前週に混ざっていれば
    # ここは1日ずれるか、月曜分のデータが週内から欠落して bid_open が変わってしまう。
    assert jst.index[0] == pd.Timestamp("2026-01-04 15:00", tz="UTC")
    assert jst.iloc[0]["bid_open"] == pytest.approx(150.00)  # 系列全体の最初の1分足
    # close_time は次の月曜 JST 00:00(= 2026-01-11 15:00 UTC)。1週間まるごと入っている証拠
    assert jst.iloc[0]["close_time"] == pd.Timestamp("2026-01-11 15:00", tz="UTC")


def test_weekly_ny_boundary_is_sunday_1700_ny_local():
    """NY基準の週足の開始は日曜17:00 America/New_York(=FXウィークの開始)に一致する。"""
    bars = minute_bars("2026-01-05 00:00", 60 * 24 * 12)
    ny = resample(bars, Timeframe.W1_NY)
    assert len(ny) == 1
    assert ny.index[0] == pd.Timestamp("2026-01-04 22:00", tz="UTC")  # 日曜17:00 EST
    assert ny.iloc[0]["close_time"] == pd.Timestamp("2026-01-11 22:00", tz="UTC")


def _market_hours_minute_bars(start: str, end: str) -> pd.DataFrame:
    """市場クローズ中はバーが存在しない、実運用に近い1分足。

    tests/helpers.py の minute_bars は市場時間を意識せず連続生成するため、週末の
    クローズ・オープンをまたぐ完結性判定(is_market_open 依存)は検証できない。
    このテスト専用に、その週末ギャップを再現したフィクスチャを用意する。
    """
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    idx = idx[is_market_open(idx).to_numpy()]
    n = len(idx)
    body: dict[str, object] = {
        "close_time": idx + pd.Timedelta(minutes=1),
        "bid_open": [150.0 + i * 0.01 for i in range(n)],
        "bid_high": [150.5 + i * 0.01 for i in range(n)],
        "bid_low": [149.5 + i * 0.01 for i in range(n)],
        "bid_close": [150.2 + i * 0.01 for i in range(n)],
        "volume": [10.0] * n,
    }
    for field in ("open", "high", "low", "close"):
        body[f"ask_{field}"] = [v + 0.02 for v in body[f"bid_{field}"]]
    return pd.DataFrame(body, index=idx).rename_axis("open_time")


def test_h4_keeps_short_bucket_at_weekly_close_and_reopen():
    """固定長分岐版の回帰テスト。週末で市場が閉まっている分は本数不足にならない。

    4時間足のバケット境界(00,04,...,20 UTC)は NY 17:00 クローズ(冬は22:00 UTC)と
    ずれているため、金曜最後の[20:00,24:00)バケットは実際には120分しか存在しない。
    それでも「その期間は完結している」(土日は最初から存在しない分であり、欠損ではない)。
    本数を固定の期待値と比較するだけの実装だとここを弾いてしまう。
    """
    bars = _market_hours_minute_bars("2026-01-09 00:00", "2026-01-13 00:00")  # 金〜火
    h4 = resample(bars, Timeframe.H4)

    friday_close_bucket = pd.Timestamp("2026-01-09 20:00", tz="UTC")
    sunday_open_bucket = pd.Timestamp("2026-01-11 20:00", tz="UTC")
    saturday_bucket = pd.Timestamp("2026-01-10 00:00", tz="UTC")

    assert friday_close_bucket in h4.index  # 120分しかないが完結している
    assert sunday_open_bucket in h4.index   # 同じく120分だが完結している
    assert saturday_bucket not in h4.index  # 土曜はそもそもバーが1本も無い

    row = h4.loc[friday_close_bucket]
    assert row["close_time"] == pd.Timestamp("2026-01-10 00:00", tz="UTC")


def test_resample_output_matches_validate_bars_contract_after_reset_index():
    """Task 11 は resample の返り値をそのまま Lake.save に渡す。Lake.save は内部で
    validate_bars を呼ぶので、その入り口である validate_bars(...) を実際に通して確認する。

    resample の返り値は Lake.load と同じ open_time-index 形なので、BAR_COLUMNS の列を
    要求する validate_bars にそのままは渡せない(open_time が列でなくindexにある)。
    reset_index() で列に戻せば通る、という契約を仮定ではなく実行して確かめる。
    """
    daily_source = minute_bars("2026-01-05 00:00", 60 * 48)
    weekly_source = minute_bars("2026-01-05 00:00", 60 * 24 * 12)
    cases = [
        (Timeframe.M5, minute_bars("2026-01-05 00:00", 20)),
        (Timeframe.H4, minute_bars("2026-01-05 00:00", 60 * 16)),
        (Timeframe.D1_NY, daily_source),
        (Timeframe.D1_JST, daily_source),
        (Timeframe.W1_NY, weekly_source),
        (Timeframe.W1_JST, weekly_source),
    ]
    for timeframe, source in cases:
        out = resample(source, timeframe)
        assert not out.empty, f"{timeframe} のテストフィクスチャが空。テスト自体を見直す"

        # index形のままでは open_time 列が無く弾かれる(Lake.load と同じ形である証拠)
        with pytest.raises(ValueError, match="open_time"):
            validate_bars(out, timeframe)

        # reset_index すれば Lake.save が要求する形になり、素通りする
        validated = validate_bars(out.reset_index(), timeframe)
        assert len(validated) == len(out)
