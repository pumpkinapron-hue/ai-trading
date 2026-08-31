import json

import numpy as np
import pandas as pd
import pytest

from aitrading.bars import resample
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


# ============================================================================
# レビュー(task-8-review.md)で見つかった欠陥の回帰テスト
# ============================================================================


# --- C1: 要求レンジを渡さないと、端がまるごと落ちた欠損が見えない ---


def test_truncated_fetch_looks_perfect_without_the_requested_range():
    """要求レンジを渡さない場合の既定の挙動を固定する。

    母数が観測データ自身の端から作られるため、末尾が丸ごと落ちていても
    「完璧」になる。これは仕様（後方互換）だが、危険なので明示的に固定して
    おき、下の2本と対で「だから渡せ」を読めるようにする。
    """
    half = minute_bars("2026-01-05 00:00", 120)  # 本来は240本ほしかった
    report = check(half, "USDJPY", Timeframe.M1)
    assert report.expected_bars == report.actual_bars == 120
    assert report.gaps == []


def test_requested_range_reveals_a_truncated_tail():
    """取得が途中で切れた（末尾が丸ごと無い）ことを検出できる。

    実運用で最も起きやすい壊れ方（レートリミット・ソース側の履歴不足・
    チャンクループの中断）なのに、要求レンジが無いと痕跡が残らない。
    """
    half = minute_bars("2026-01-05 00:00", 120)
    report = check(
        half,
        "USDJPY",
        Timeframe.M1,
        expected_start=pd.Timestamp("2026-01-05 00:00", tz="UTC"),
        expected_end=pd.Timestamp("2026-01-05 04:00", tz="UTC"),
    )
    assert report.expected_bars == 240
    assert report.actual_bars == 120
    assert len(report.gaps) == 1
    assert report.gaps[0]["missing_bars"] == 120
    assert report.gaps[0]["from"] == "2026-01-05 02:00:00+00:00"
    assert report.gaps[0]["to"] == "2026-01-05 03:59:00+00:00"


def test_requested_range_reveals_a_truncated_head():
    """先頭が丸ごと無い場合も同じ。隣接バーの間隔だけを見る実装では原理的に見えない。"""
    tail = minute_bars("2026-01-05 02:00", 120)
    report = check(
        tail,
        "USDJPY",
        Timeframe.M1,
        expected_start=pd.Timestamp("2026-01-05 00:00", tz="UTC"),
        expected_end=pd.Timestamp("2026-01-05 04:00", tz="UTC"),
    )
    assert report.expected_bars == 240
    assert len(report.gaps) == 1
    assert report.gaps[0]["from"] == "2026-01-05 00:00:00+00:00"
    assert report.gaps[0]["missing_bars"] == 120


def test_requested_range_still_excludes_the_weekend():
    """要求レンジを渡しても、その中の週末クローズは母数に入らない。"""
    friday = minute_bars("2026-01-09 20:00", 120)
    report = check(
        friday,
        "USDJPY",
        Timeframe.M1,
        expected_start=pd.Timestamp("2026-01-09 20:00", tz="UTC"),
        expected_end=pd.Timestamp("2026-01-11 22:00", tz="UTC"),  # 日曜の再開ちょうどまで
    )
    assert report.expected_bars == 120
    assert report.gaps == []


def test_requested_range_rejects_naive_bounds():
    bars = minute_bars("2026-01-05 00:00", 10)
    with pytest.raises(ValueError, match="tz-aware"):
        check(bars, "USDJPY", Timeframe.M1, expected_start=pd.Timestamp("2026-01-05"))


# --- C2: 週明けの窓開けを価格ジャンプに数えない ---


def _gapped_bars(gap_pips: float) -> pd.DataFrame:
    """金曜クローズまでと日曜の再開後をつなぎ、その境目に窓開けを作る。

    合成データの連続系列では絶対に再現できない状況。実データでは毎週起きる。
    """
    friday = minute_bars("2026-01-09 20:00", 120)  # 〜金曜22:00Z(クローズ)
    sunday = minute_bars("2026-01-11 22:00", 120)  # 日曜22:00Z(オープン)〜
    for column in ("bid_open", "bid_high", "bid_low", "bid_close",
                   "ask_open", "ask_high", "ask_low", "ask_close"):
        sunday[column] = sunday[column] + gap_pips * 0.01
    return pd.concat([friday, sunday])


@pytest.mark.parametrize("gap_pips", [5.0, 20.0, 50.0, 200.0])
def test_weekend_price_gap_is_not_a_price_jump(gap_pips):
    """週明けの窓開けは価格ジャンプに数えない。

    数えると、10年ぶんのレポートで price_jump_count の大半が週末になり、
    実質「週末カウンタ」になる。週明けバーのATR窓は金曜クローズ直前の
    十数分（週で最も流動性が枯れた時間帯）だけで構成されるため、分子が
    週で最大になる瞬間に分母が週で最小になる。実測では数pipsの窓開けで
    毎週100%発火していた。
    """
    report = check(_gapped_bars(gap_pips), "USDJPY", Timeframe.M1)
    assert report.price_jump_count == 0


def test_data_gap_is_not_a_price_jump_either():
    """平日のデータ欠損を跨いだ価格変化も同じ理由で数えない。"""
    first = minute_bars("2026-01-05 00:00", 60)
    second = minute_bars("2026-01-05 03:00", 60)
    for column in ("bid_open", "bid_high", "bid_low", "bid_close",
                   "ask_open", "ask_high", "ask_low", "ask_close"):
        second[column] = second[column] + 50.0
    report = check(pd.concat([first, second]), "USDJPY", Timeframe.M1)
    assert report.price_jump_count == 0
    assert len(report.gaps) == 1  # 欠損としては、ちゃんと出ている


def test_price_jump_inside_contiguous_bars_is_still_detected():
    """C2の修正が、本来検出すべきジャンプまで消していないことの対照実験。"""
    bars = minute_bars("2026-01-05 00:00", 240)
    bars.loc[bars.index[120], "bid_close"] += 50.0
    bars.loc[bars.index[120], "ask_close"] += 50.0
    assert check(bars, "USDJPY", Timeframe.M1).price_jump_count == 2


# --- I2: バーの長さを考慮した母数（4時間足で expected < actual にならない） ---


@pytest.mark.parametrize(
    "timeframe", [Timeframe.M5, Timeframe.M15, Timeframe.H1, Timeframe.H4]
)
def test_expected_never_undercounts_intact_higher_timeframes(timeframe):
    """無傷のデータで expected_bars == actual_bars になること。

    市場が開いているかを open_time の1点だけで判定すると、4時間足の
    [20:00,24:00) バケットは日曜の再開(22:00Z)がバケットの内側に落ちるため
    実在するバーが母数から外れ、expected < actual になる（毎週末、確実に）。
    モジュール自身の不変条件が反転するので、ここで固定する。
    """
    minutes = pd.date_range("2026-01-09", "2026-01-14", freq="1min", tz="UTC",
                            inclusive="left")
    minutes = minutes[is_market_open(minutes).to_numpy()]
    higher = resample(_flat_bars(minutes), timeframe)
    report = check(higher, "USDJPY", timeframe)
    assert report.expected_bars == report.actual_bars == len(higher)
    assert report.gaps == []


# --- I3: wide_spread_quantile が実際に配線されているか ---


def _varied_spread_bars(n: int = 1000) -> pd.DataFrame:
    """スプレッドが単調に広がっていくバー。分位点を動かすと答えが変わる。"""
    bars = minute_bars("2026-01-05 00:00", n)
    bars["ask_close"] = bars["bid_close"] + np.linspace(0.01, 1.0, n)
    return bars


def test_wide_spread_quantile_parameter_is_actually_used():
    """`wide_spread_quantile` を無視して 0.999 固定にする実装を弾く。

    既存のフィクスチャ（一定スプレッド＋外れ値1本）ではどの分位点でも
    答えが同じになるため、この引数が配線されていなくても検出できなかった。
    """
    bars = _varied_spread_bars()
    counts = {
        q: check(bars, "USDJPY", Timeframe.M1, wide_spread_quantile=q).wide_spread_count
        for q in (0.5, 0.9, 0.99)
    }
    assert counts[0.5] > counts[0.9] > counts[0.99]
    assert counts[0.5] == pytest.approx(500, abs=2)


def test_wide_spread_threshold_is_reported():
    """本数だけでは「上位0.1%」を数え直しているだけで情報がほとんど無い。
    しきい値そのものを見れば、スプレッドの分布が普段と違うかを判断できる。"""
    bars = _varied_spread_bars()
    report = check(bars, "USDJPY", Timeframe.M1, wide_spread_quantile=0.5)
    assert report.wide_spread_threshold == pytest.approx(0.505, abs=0.01)


def test_wide_spread_threshold_ignores_non_positive_spreads():
    """分位点の基準は正のスプレッドだけから作る（0以下は bad_spread_count の担当）。
    含めてしまうと、壊れた行が多いほどしきい値が下がって外れ値が隠れる。"""
    bars = _varied_spread_bars(100)
    bars.loc[bars.index[:50], "ask_close"] = bars.loc[bars.index[:50], "bid_close"]
    report = check(bars, "USDJPY", Timeframe.M1, wide_spread_quantile=0.5)
    assert report.bad_spread_count == 50
    # 残った50本（0.5〜1.0付近）の中央値。0を含めていれば大きく下振れする
    assert report.wide_spread_threshold > 0.4


# --- I4: longest_gap_minutes が本当に「最長」か ---


def test_longest_gap_is_the_maximum_not_the_sum_or_the_minimum():
    """穴が1つしか無いテストでは max と sum と min の区別がつかない。"""
    bars = minute_bars("2026-01-05 00:00", 600)
    bars = bars.drop(bars.index[400:460])  # 60分の穴
    bars = bars.drop(bars.index[100:105])  # 5分の穴
    report = check(bars, "USDJPY", Timeframe.M1)
    assert sorted(g["missing_bars"] for g in report.gaps) == [5, 60]
    assert report.longest_gap_minutes == pytest.approx(60.0)  # sum=65, min=5


# --- I5 / M12 / M16: ジャンプ判定の各要素を固定する ---


def test_threshold_is_atr_times_multiple_using_high_minus_low():
    """しきい値が「(high-low)の14本平均 × 倍率」ちょうどであることを固定する。

    minute_bars の値幅は常に1.0なのでATRは1.0、既定倍率10でしきい値は10.0。
    本来の True Range（|high-前close| を含む）に変えると、ジャンプ本自身のTRが
    ATRに混ざってしきい値が上がり、10.5 は検出されなくなる。どちらが良いかとは
    別に、いまどちらで判定しているかをテストで明示しておく。
    """
    def count(jump: float) -> int:
        bars = minute_bars("2026-01-05 00:00", 100)
        bars.loc[bars.index[50], "bid_close"] += jump
        bars.loc[bars.index[50], "ask_close"] += jump
        return check(bars, "USDJPY", Timeframe.M1).price_jump_count

    assert count(10.5) == 2  # しきい値 10.0 を超える
    assert count(9.5) == 0   # 超えない


def test_jump_uses_the_mid_price_not_bid_alone():
    """判定価格は (bid+ask)/2。ask だけが動いたケースで区別できる。"""
    bars = minute_bars("2026-01-05 00:00", 100)
    bars.loc[bars.index[50], "ask_close"] += 30.0  # mid は +15 動く
    assert check(bars, "USDJPY", Timeframe.M1).price_jump_count == 2


# --- I6: 値が食い違う重複を、無害な重複と区別する ---


def test_identical_duplicate_is_counted_but_not_flagged_as_conflicting():
    """同じ値の重複は再取得で普通に起きる。異常ではない。"""
    bars = minute_bars("2026-01-05 00:00", 60)
    bars = pd.concat([bars, bars.iloc[[10]]])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.duplicate_count == 1
    assert report.conflicting_duplicate_count == 0


def test_conflicting_duplicate_is_reported_separately():
    """同じ時刻に違う値がある＝データソース側が過去を書き換えた疑い。

    `Lake._merge_year` は同じ状況を ValueError にする。こちらは報告するのが
    仕事なので投げないが、黙って最初の1本を残して終わりにはしない
    （それだと異常値がレポートのどこにも現れない）。
    """
    bars = minute_bars("2026-01-05 00:00", 60)
    conflicting = bars.iloc[[10]].copy()
    conflicting["bid_close"] = 999.0
    report = check(pd.concat([bars, conflicting]), "USDJPY", Timeframe.M1)
    assert report.duplicate_count == 1
    assert report.conflicting_duplicate_count == 1


# --- 入力の並び順・退化した入力 ---


def test_unsorted_input_does_not_change_the_result():
    """`check()` の契約に「ソート済みであること」は書かれていない。

    並び順に効くのは隣接判定（`index.diff() == step`）で、逆順だと差分が負に
    なって全行が「隣接していない」扱いになる。欠損だけを含むフィクスチャでは
    両者の答えが偶然一致してしまうので、必ずジャンプも入れておくこと。
    """
    bars = minute_bars("2026-01-05 00:00", 120)
    bars = bars.drop(bars.index[50:60])
    bars.loc[bars.index[80], "bid_close"] += 50.0
    bars.loc[bars.index[80], "ask_close"] += 50.0

    sorted_report = check(bars, "USDJPY", Timeframe.M1)
    assert sorted_report.price_jump_count == 2  # 並び順の影響が出る値であることの担保
    assert check(bars.iloc[::-1], "USDJPY", Timeframe.M1).to_dict() == (
        sorted_report.to_dict()
    )


@pytest.mark.parametrize("n", [0, 1])
def test_degenerate_input_does_not_crash(n):
    """0行・1行でも例外を出さない（取得が完全に失敗した場合の経路）。"""
    report = check(minute_bars("2026-01-05 00:00", n), "USDJPY", Timeframe.M1)
    assert report.actual_bars == n
    assert report.expected_bars == n
    assert report.gaps == []


# --- レポートの識別子 ---


def test_report_carries_the_symbol_and_timeframe_it_was_given():
    """`symbol` / `timeframe` を assert しているテストが1本も無かった。
    `to_dict()` の往復テストは自己整合の比較なので、両方が固定値でも通ってしまう。"""
    report = check(minute_bars("2026-01-05 00:00", 60), "EURUSD", Timeframe.M5)
    assert report.symbol == "EURUSD"
    assert report.timeframe == "5m"
