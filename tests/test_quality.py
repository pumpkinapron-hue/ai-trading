import json

import numpy as np
import pandas as pd
import pytest

from aitrading.quality import check
from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe, is_market_open

from tests.helpers import minute_bars


def _flat_bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    """任意の index に対して、価格が完全に一定なバーを作る。

    ギャップ検出だけを見たいテストで使う。値を一定にしておけば、
    スプレッド・ジャンプ系の判定が誤って発火する心配をせずに済む。
    """
    n = len(index)
    body: dict[str, object] = {
        "close_time": index + pd.Timedelta(minutes=1),
        "bid_open": [150.0] * n,
        "bid_high": [150.05] * n,
        "bid_low": [149.95] * n,
        "bid_close": [150.0] * n,
        "ask_open": [150.02] * n,
        "ask_high": [150.07] * n,
        "ask_low": [149.97] * n,
        "ask_close": [150.02] * n,
        "volume": [10.0] * n,
    }
    return pd.DataFrame(body, index=index).rename_axis("open_time")


# --- 基本 ---


def test_clean_data_has_no_gaps():
    # 月曜 00:00 UTC から4時間。市場は開いている。
    report = check(minute_bars("2026-01-05 00:00", 240), "USDJPY", Timeframe.M1)
    assert report.gaps == []
    assert report.actual_bars == 240
    assert report.expected_bars == 240


def test_detects_missing_bars():
    bars = minute_bars("2026-01-05 00:00", 240)
    bars = bars.drop(bars.index[100:130])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert len(report.gaps) == 1
    assert report.longest_gap_minutes == pytest.approx(30.0)
    assert report.actual_bars == 210
    # 「期待本数 = 実本数 + 欠損分」。この等式は _detect_gaps と expected_bars の
    # 計算経路が食い違っていないことの直接証拠になる。
    assert report.expected_bars == 240


# --- 週末クローズと欠損の区別（このモジュールの核心） ---
#
# 2026-01-09は金曜、01-10は土曜、01-11は日曜、01-12は月曜
# （test_timeutil.test_market_closed_on_weekend と同じ基準週）。
# 市場は 金曜 NY17:00=UTC22:00 に閉まり、日曜 NY17:00=UTC22:00 に開く（冬季・EST）。


def test_weekend_is_not_counted_as_a_gap():
    """週末クローズを欠損と数えると、品質チェックが毎週偽陽性を出す。

    草案のフィクスチャは金曜20:00UTCから120分（→22:00UTC=クローズちょうどまで）と、
    月曜00:00UTCから120分だった。しかし市場が再開するのは日曜22:00UTC(NY17:00)で
    あり、月曜00:00UTCではない。その間の日曜22:00〜24:00UTCの120分は
    「市場が開いているのにバーが無い」区間になり、正しく実装された check() では
    gaps が空にならない（実測: is_market_open で120/120分が開いている）。
    意図（週末そのものは欠損に数えない）を保ったまま、2本目のチャンクを
    市場再開ちょうど（日曜22:00UTC）から始まるように直した。
    """
    friday = minute_bars("2026-01-09 20:00", 120)  # 〜金曜22:00UTC(クローズ)まで
    sunday_reopen = minute_bars("2026-01-11 22:00", 120)  # 日曜22:00UTC(オープン)から
    bars = pd.concat([friday, sunday_reopen])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.gaps == []
    assert report.actual_bars == 240
    # 週末ぶんは母数からも除かれる。除かなければ expected が実本数より
    # 常に大きくなり、毎週「本数が足りない」という偽陽性になる。
    assert report.expected_bars == 240


def test_weekend_reopen_gap_is_still_detected_as_missing_data():
    """`test_weekend_is_not_counted_as_a_gap` と対になるテスト。

    週末の「市場が閉まっている」部分は無視してよいが、日曜22:00UTCの再開後に
    データが無ければ、それは正真正銘の欠損として検出されなければならない。
    これは草案の元のフィクスチャ（月曜00:00UTCから2本目が始まる）と同じ状況で、
    日曜22:00〜24:00UTCの120分がそのまま欠損になる。
    """
    friday = minute_bars("2026-01-09 20:00", 120)  # 〜金曜22:00UTCまで
    monday = minute_bars("2026-01-12 00:00", 120)  # 月曜00:00UTCから（日曜分が抜けている）
    bars = pd.concat([friday, monday])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert len(report.gaps) == 1
    assert report.gaps[0]["missing_bars"] == 120
    assert report.longest_gap_minutes == pytest.approx(120.0)
    assert report.actual_bars == 240
    assert report.expected_bars == 360


def test_multiple_gaps_and_weekends_over_a_month():
    """月をまたぐ規模で、週末クローズ（自然な穴）と実データ欠損（複数箇所）が
    混在していても正しく区別できることを確認する。ギャップ検出をベクトル化した
    実装（診断済みの穴だけをループする）の回帰検査を兼ねる。
    """
    full_index = pd.date_range(
        "2026-01-05", "2026-02-02", freq="1min", tz="UTC", inclusive="left"
    )
    open_index = full_index[is_market_open(full_index).to_numpy()]
    n = len(open_index)

    # 互いに十分離れた5箇所に7分の穴を開ける（間隔は月の1/5ぶんあるので重ならない）
    hole_starts = [int(n * frac) for frac in (0.1, 0.3, 0.5, 0.7, 0.9)]
    drop = np.zeros(n, dtype=bool)
    for h in hole_starts:
        drop[h : h + 7] = True
    kept_index = open_index[~drop]

    report = check(_flat_bars(kept_index), "USDJPY", Timeframe.M1)

    assert report.actual_bars == len(kept_index)
    assert len(report.gaps) == 5
    assert all(g["missing_bars"] == 7 for g in report.gaps)
    assert report.expected_bars == n
    assert report.duplicate_count == 0
    assert report.bad_spread_count == 0
    assert report.wide_spread_count == 0


# --- スプレッド異常 ---


def test_detects_zero_and_negative_spread():
    bars = minute_bars("2026-01-05 00:00", 60)
    bars.loc[bars.index[5], "ask_close"] = bars.loc[bars.index[5], "bid_close"]
    bars.loc[bars.index[6], "ask_close"] = bars.loc[bars.index[6], "bid_close"] - 0.01
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.bad_spread_count == 2


def test_wide_spread_quantile_flags_nothing_when_spread_is_constant():
    """スプレッドが全部同じ値なら、分位点しきい値もその値になり、
    「しきい値を超える」行は1つも無い(境界の等号を > で判定しているため)。
    分位点による外れ値検出が、一様なデータに対して誤検出しないことの確認。
    """
    bars = minute_bars("2026-01-05 00:00", 1000)  # ask-bid スプレッドは常に0.02
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.wide_spread_count == 0


def test_wide_spread_detects_a_genuine_outlier():
    bars = minute_bars("2026-01-05 00:00", 2000)  # 通常スプレッドは一様に0.02
    bars.loc[bars.index[500], "ask_close"] += 5.0  # 1本だけ突出して広い
    report = check(bars, "USDJPY", Timeframe.M1, wide_spread_quantile=0.999)
    assert report.wide_spread_count == 1


# --- 価格ジャンプ ---


def test_detects_price_jump():
    bars = minute_bars("2026-01-05 00:00", 240)
    bars.loc[bars.index[120], "bid_close"] += 50.0
    bars.loc[bars.index[120], "ask_close"] += 50.0
    report = check(bars, "USDJPY", Timeframe.M1)
    # 120本目のジャンプは「120本目に入る差分」と「121本目に入る差分」の
    # 両方に表れる(120本目のcloseだけが動くため、前後どちらの隣接差分も
    # 大きくなる)。ATRのウォームアップ(先頭14本)は十分過ぎている位置なので
    # 両方とも検出される。
    assert report.price_jump_count == 2


def test_price_jump_during_atr_warmup_is_not_detected():
    """`rolling(14, min_periods=14)` なので先頭13本はATRがNaNになり、
    その間に起きたジャンプは検出できない。これは意図的な既知の制約
    (quality.py の _price_jump_count のdocstring参照)。5本目(<13)で起こす
    ジャンプは検出されないことを確認する。
    """
    bars = minute_bars("2026-01-05 00:00", 40)
    bars.loc[bars.index[5], "bid_close"] += 50.0
    bars.loc[bars.index[5], "ask_close"] += 50.0
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.price_jump_count == 0


def test_price_jump_after_atr_warmup_is_detected():
    """`test_price_jump_during_atr_warmup_is_not_detected` と対照実験。
    同じ大きさのジャンプでも、ATRが有効になった後(20本目 >= 13)なら検出される。
    """
    bars = minute_bars("2026-01-05 00:00", 40)
    bars.loc[bars.index[20], "bid_close"] += 50.0
    bars.loc[bars.index[20], "ask_close"] += 50.0
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.price_jump_count == 2


def test_jump_atr_multiple_parameter_is_actually_used():
    """`jump_atr_multiple` がしきい値の計算式に実際に配線されていることの確認。

    既存のテスト(test_detects_price_jump など)は既定値(10.0)でしか呼んでおらず、
    もし実装がこの引数を無視して10.0をハードコードしていても既定値のテストは
    区別できずに通ってしまう(実際に変異検査で確認した)。ここでは同じジャンプに
    対して明示的に大きい倍率を渡し、しきい値が本当に動くことを見る。
    """
    bars = minute_bars("2026-01-05 00:00", 240)
    bars.loc[bars.index[120], "bid_close"] += 50.0
    bars.loc[bars.index[120], "ask_close"] += 50.0

    default_report = check(bars, "USDJPY", Timeframe.M1)
    assert default_report.price_jump_count == 2  # 既定値(10.0)なら検出される

    loose_report = check(bars, "USDJPY", Timeframe.M1, jump_atr_multiple=100.0)
    assert loose_report.price_jump_count == 0  # 倍率を上げれば検出されなくなる


def test_jump_atr_multiple_sets_the_threshold_precisely():
    """`jump_atr_multiple` の値そのものがしきい値の倍率になっていることを、
    境界のすぐ内側・外側にジャンプ幅を置いて厳密に確認する。

    `minute_bars` フィクスチャは bid_high-bid_low が常に1.0の定数なので、
    ウォームアップ後のATRは常にちょうど1.0になる。しきい値は
    `1.0 * jump_atr_multiple` に一致するはずなので、ジャンプ幅8.0は
    倍率10(既定)なら検出されず、倍率5になれば検出される。
    """
    bars = minute_bars("2026-01-05 00:00", 40)
    bars.loc[bars.index[20], "bid_close"] += 8.0
    bars.loc[bars.index[20], "ask_close"] += 8.0

    strict_report = check(bars, "USDJPY", Timeframe.M1, jump_atr_multiple=10.0)
    assert strict_report.price_jump_count == 0  # しきい値10.0 > ジャンプ約8.0

    loose_report = check(bars, "USDJPY", Timeframe.M1, jump_atr_multiple=5.0)
    assert loose_report.price_jump_count == 2  # しきい値5.0 < ジャンプ約8.0


def test_jump_atr_multiple_is_multiplicative_not_additive():
    """`jump_atr_multiple` はATRへの加算オフセットではなく倍率でなければならない。

    ATRがちょうど1.0の場合、`atr * multiple` と `atr + multiple` は近い値になり
    (例: 倍率10なら10.0 と 11.0)、上の2つのテストのジャンプ幅では区別できない
    (実際に変異検査で確認した: 乗算を加算に変える変異はどのテストにも
    検知されなかった)。ここでは ATR を 4.0 にずらし、乗算なら閾値40.0・
    加算なら閾値14.0と大きく開くようにしたうえで、その中間のジャンプ幅(25.0)
    で判定させる。加算になっていれば誤って検出してしまう。
    """
    bars = minute_bars("2026-01-05 00:00", 40)
    bars["bid_high"] = bars["bid_low"] + 4.0  # true_range を定数4.0にする(ATR=4.0)
    bars.loc[bars.index[20], "bid_close"] += 25.0
    bars.loc[bars.index[20], "ask_close"] += 25.0

    report = check(bars, "USDJPY", Timeframe.M1, jump_atr_multiple=10.0)
    # 乗算なら閾値 = 4.0*10.0 = 40.0 > ジャンプ約25.0 → 検出されない
    assert report.price_jump_count == 0


# --- 重複 ---


def test_detects_duplicates():
    bars = minute_bars("2026-01-05 00:00", 60)
    bars = pd.concat([bars, bars.iloc[[10]]])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.duplicate_count == 1
    assert report.actual_bars == 60  # 重複ぶんは1本に畳まれる


# --- tz-aware の強制 ---


def test_check_rejects_naive_index():
    """このリポジトリの規約: naive な時刻は ValueError。timeutil.ensure_utc と
    同じ規約に揃える(他のモジュールはすべてこの規約に従っている)。
    """
    bars = minute_bars("2026-01-05 00:00", 10)
    bars.index = bars.index.tz_localize(None)
    with pytest.raises(ValueError, match="tz-aware"):
        check(bars, "USDJPY", Timeframe.M1)


def test_check_converts_non_utc_tz_aware_index():
    """naiveでなければ、UTC以外のtzでも受け付けて内部でUTCに揃える
    (ensure_utcの既存の契約と同じ)。"""
    bars = minute_bars("2026-01-05 00:00", 10)
    bars.index = bars.index.tz_convert("Asia/Tokyo")
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.actual_bars == 10
    assert report.gaps == []


# --- 日足・週足(可変長)は非対応であることを明示する ---


@pytest.mark.parametrize(
    "timeframe",
    [Timeframe.D1_NY, Timeframe.D1_JST, Timeframe.W1_NY, Timeframe.W1_JST],
)
def test_variable_length_timeframe_raises(timeframe):
    """`timeframe.delta` が None(日足・週足)の場合、黙って間違った本数を返すより
    明示的に ValueError にする。日境界の暦計算(夏時間で23/25時間になる区切り)を
    ここで簡易的に再実装すると、まさに timeutil に集約する前に起きていた
    「区切りが1時間ずれる」先読みバグと同じ構造の間違いを繰り返しかねない。
    """
    bars = minute_bars("2026-01-05 00:00", 10)
    with pytest.raises(ValueError, match="可変長"):
        check(bars, "USDJPY", timeframe)


# --- to_dict / Meta 連携 ---


def test_to_dict_is_json_serializable():
    report = check(minute_bars("2026-01-05 00:00", 60), "USDJPY", Timeframe.M1)
    json.dumps(report.to_dict(), default=str)


def test_to_dict_round_trips_through_meta(tmp_path):
    """`to_dict()` が実際に Meta.record_quality / latest_quality を往復することを
    仮定ではなく実際に確認する。ギャップを含む(=ネストしたlist[dict]を含む)
    レポートで試す。
    """
    bars = minute_bars("2026-01-05 00:00", 240)
    bars = bars.drop(bars.index[100:130])
    report = check(bars, "USDJPY", Timeframe.M1)

    meta = Meta(tmp_path / "meta.db")
    meta.record_quality("USDJPY", Timeframe.M1, report.to_dict())
    got = meta.latest_quality("USDJPY", Timeframe.M1)

    assert got == report.to_dict()
    assert got["actual_bars"] == 210
    assert len(got["gaps"]) == 1
    assert got["gaps"][0]["missing_bars"] == 30
