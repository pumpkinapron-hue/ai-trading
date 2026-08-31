"""全指標へのトランケーション不変性検査（先読み防止 第2層）。

入力の末尾を切り落としても、残った部分の出力が1つも変わらないこと。
先読みしている指標はこれで必ず落ちる。

このファイルの本体は `assert_truncation_invariant` と、それが本当に
先読みを検出できることを確かめる `test_detector_catches_a_deliberate_lookahead`
である。検査そのものが甘いと、他の全指標の合格は何の保証にもならない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aitrading.indicators import INDICATORS
from aitrading.indicators.core import vwap

#: 既定の切り落とし量。B: 複数の量でも検証する（下の
#: test_indicator_does_not_look_ahead_at_various_truncations）ので、
#: この値自体に特別な意味はない。
TRUNCATE = 10


def assert_same_prefix(full, truncated) -> None:
    """先頭 len(truncated) 本の値が一致することだけを見る（長さは見ない）。

    長さの検査は呼び出し側（`assert_truncation_invariant`）の責務にしてある。
    ここで長さも検査してしまうと、この関数を「検出器の動作確認」
    （test_detector_catches_a_deliberate_lookahead）にも使う際、
    値のズレそのものを見たいのに長さの都合で話がややこしくなるため。

    check_exact=True にしている（B: 既定の rtol=1e-5 程度の許容誤差だと、
    影響の小さい先読みを見逃すことを実測で確認した——例えば vwap を
    「累積和」から「日合計」に変えるバグは、日の末尾1本だけを切り落とす
    ケースだと日平均への影響が小さすぎて既定の許容誤差の中に埋もれて
    検出できなかった）。先読みが無い純粋な因果計算は、末尾を切っても
    残りはビット単位で不変になるはず（rolling/ewm は未来の要素を一切
    読まないので、配列を後ろに伸ばしても過去の計算結果は変わらない）。
    実際、登録済み9指標すべてで check_exact=True でも誤検出は出ない
    ことを複数の切り落とし量で確認済み。厳密比較にして損はない。
    """
    head = full.iloc[: len(truncated)]
    if isinstance(full, pd.DataFrame):
        pd.testing.assert_frame_equal(head, truncated, check_exact=True)
    else:
        pd.testing.assert_series_equal(head, truncated, check_exact=True)


def assert_truncation_invariant(fn, bars: pd.DataFrame, truncate: int) -> None:
    """指標関数 `fn` がトランケーション不変であることを検査する。

    C: `full.iloc[:len(truncated)]` と比べるだけの prefix 比較は、
    「truncated 自体の長さがおかしい」ケースを構造的に見逃す
    （len(truncated) に自分で合わせてから比較するので、極端には長さ0を
    返す実装でも prefix 比較そのものは通ってしまう）。そのため、
    出力の長さが入力の長さと一致することを prefix 比較の前に別途固定する。
    """
    truncated_bars = bars.iloc[:-truncate]
    full = fn(bars)
    truncated = fn(truncated_bars)
    assert len(truncated) == len(truncated_bars), (
        f"出力の長さ({len(truncated)})が入力の長さ({len(truncated_bars)})と一致しない。"
        " 長さの合っていない出力は prefix 比較だけでは検出できない"
    )
    assert_same_prefix(full, truncated)


def test_registry_is_not_empty():
    assert INDICATORS, "指標が1つも登録されていない"


def test_registry_has_all_nine_phase0_indicators():
    assert set(INDICATORS) == {
        "sma", "ema", "rsi", "macd", "atr", "bbands", "vwap", "donchian", "hist_vol",
    }


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_look_ahead(name, sample_bars):
    assert_truncation_invariant(INDICATORS[name], sample_bars, TRUNCATE)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_preserves_index(name, sample_bars):
    result = INDICATORS[name](sample_bars)
    pd.testing.assert_index_equal(result.index, sample_bars.index)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_mutate_input(name, sample_bars):
    before = sample_bars.copy(deep=True)
    INDICATORS[name](sample_bars)
    pd.testing.assert_frame_equal(sample_bars, before)


# --- B: 切り落とし量を固定(10)にしていて、その量でだけ露見する先読みを
# 見逃していないか。
#
# 実測で分かったこと: mean/std 系（sma, bbands, hist_vol, ...)は、切り落とし
# 量が1本でも境界の値が実質確実に変わる（連続値の平均・標準偏差は窓構成が
# 1つ変わるだけでほぼ確実にビットレベルで変化するため）。
#
# ところが max/min 系（donchian）は違う。center=True の先読みを donchian に
# 仕込んで手動で変異検査したところ、TRUNCATE=10（固定）は素通りし、
# TRUNCATE=59 でようやく検出できた。境界window内の「本来見てはいけない
# 未来側」の値が、たまたまその window の最大値・最小値を更新しない限り
# 検出できないため（rolling.max/min は順序統計量で、window構成が変わっても
# 値そのものは変わらないことが多い）。しかも実測では乱数系列によって
# 検出できる切り落とし量がバラバラ（seed=1は3, seed=2は20と70, seed=42は
# 20/25/30/50など）で、seed=2 では [1,3,10,25,59] の5点ではどれも
# 検出できなかった。つまり少数の切り落とし量だけでは donchian 系
# （順序統計量ベース）の先読みを見逃しうる。
#
# そのため、1本刻みで1〜150本を全部試す（300本のうち150本残る計算になり、
# hist_volの60本窓にも十分な余裕がある。9指標×150通りでも実測1.3秒程度）。
_TRUNCATE_SWEEP = range(1, 151)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_look_ahead_at_various_truncations(name, sample_bars):
    fn = INDICATORS[name]
    for truncate in _TRUNCATE_SWEEP:
        assert_truncation_invariant(fn, sample_bars, truncate)


# --- H: 既定パラメータでしか呼んでいないと、非既定パラメータでだけ
# 先読みする実装を見逃す可能性がある。レジストリ経由(常に既定値)ではなく
# core.py の関数を直接、非既定のパラメータで呼ぶ。切り落とし量も
# 上と同じ理由で複数試す。
_NONDEFAULT_KWARGS = {
    "sma": {"period": 7},
    "ema": {"period": 7},
    "rsi": {"period": 7},
    "macd": {"fast": 5, "slow": 13, "signal": 4},
    "atr": {"period": 7},
    "bbands": {"period": 10, "num_std": 1.5},
    "donchian": {"period": 10},
    "hist_vol": {"period": 20},
    # vwap にはパラメータが無い
}


@pytest.mark.parametrize("name", sorted(_NONDEFAULT_KWARGS))
def test_indicator_does_not_look_ahead_with_nondefault_params(name, sample_bars):
    fn = INDICATORS[name]
    kwargs = _NONDEFAULT_KWARGS[name]
    for truncate in _TRUNCATE_SWEEP:
        assert_truncation_invariant(lambda bars: fn(bars, **kwargs), sample_bars, truncate)


# --- D: vwap は NY の取引日境界で groupby する。sample_bars は300分
# （NYの1取引日に収まる）なので、groupby が複数グループに分かれる状況を
# 一度も経験しないままトランケーション不変性を「合格」してしまう。
# 日境界をまたぐ multi_day_bars（日1=1320本, 日2=680本, 全2000本）で、
# 日をまたいだ切り落としでも別途確認する。
@pytest.mark.parametrize(
    "truncate",
    [
        1,     # 日2内の最小の切り落とし
        10,    # 他のテストの既定値と揃える
        300,   # 日2の途中まで大きく切り落とす（両日とも残る）
        680,   # 切り落とし後の末尾がちょうど日1の最終バーに一致する境界ケース
        681,   # 境界を1本越えて日1の内側に食い込む
        700,   # 日1の内側にまとまって食い込む
        1319,  # 日1の先頭近くまで深く切り落とす
    ],
)
def test_vwap_truncation_invariant_across_multiple_ny_days(multi_day_bars, truncate):
    assert_truncation_invariant(vwap, multi_day_bars, truncate)


# --- A: 検出器そのものの検査。ここが不完全だと他の全指標の合格が
# 無意味になる。設計文書が名指しする3形態（center=True のローリング窓、
# 将来のバーを見た正規化、全期間統計を使った標準化）に加えて、
# shift(-1) / bfill / 逆順累積和も検出できることを確認する。
def _cheat_center_window(bars: pd.DataFrame) -> pd.Series:
    """中央寄せの窓は未来のバーを見る。"""
    return bars["bid_close"].rolling(5, center=True, min_periods=1).mean()


def _cheat_shift_future(bars: pd.DataFrame) -> pd.Series:
    """1本先の値をそのまま使う。"""
    return bars["bid_close"].shift(-1)


def _cheat_global_normalize(bars: pd.DataFrame) -> pd.Series:
    """全期間の平均で正規化する。末尾を切ると平均そのものが変わる。"""
    close = bars["bid_close"]
    return close / close.mean()


def _cheat_global_zscore(bars: pd.DataFrame) -> pd.Series:
    """全期間の平均・標準偏差で標準化する。"""
    close = bars["bid_close"]
    return (close - close.mean()) / close.std()


def _cheat_bfill_from_future(bars: pd.DataFrame) -> pd.Series:
    """欠損を後ろ（未来）の値で埋める。

    境界のすぐ内側に意図的に欠損を作る。TRUNCATE本を切り落とすと、
    full側ではその欠損が「切り落とされる範囲の値」で埋まるが、truncated側は
    埋める先の値ごと存在しないため NaN のまま残る。
    """
    close = bars["bid_close"].copy()
    close.iloc[-(TRUNCATE + 5) : -TRUNCATE] = np.nan
    return close.bfill()


def _cheat_reverse_cumsum(bars: pd.DataFrame) -> pd.Series:
    """逆順に累積和を取る。各時点の値が自分より後ろの全データに依存する。"""
    return bars["bid_close"].iloc[::-1].cumsum().iloc[::-1]


_CHEATERS = {
    "center=True のローリング窓": _cheat_center_window,
    "shift(-1) で1本先を見る": _cheat_shift_future,
    "全期間平均で正規化": _cheat_global_normalize,
    "全期間統計でzスコア化": _cheat_global_zscore,
    "bfill で未来の値を後ろ埋め": _cheat_bfill_from_future,
    "逆順累積和": _cheat_reverse_cumsum,
}


@pytest.mark.parametrize("label", sorted(_CHEATERS))
def test_detector_catches_a_deliberate_lookahead(label, sample_bars):
    """検査そのものが機能していることを確かめる。"""
    fn = _CHEATERS[label]
    with pytest.raises(AssertionError):
        assert_same_prefix(fn(sample_bars), fn(sample_bars.iloc[:-TRUNCATE]))


# --- C: 長さチェック自体の検査（変異検査）。
# 出力を常に空にする実装は、prefix比較だけ（長さチェック無し）だと
# full.iloc[:0] も空になるため検出できない。
def test_length_check_catches_an_empty_output(sample_bars):
    def always_empty(bars: pd.DataFrame) -> pd.Series:
        return bars["bid_close"].iloc[0:0]

    with pytest.raises(AssertionError):
        assert_truncation_invariant(always_empty, sample_bars, TRUNCATE)


def test_length_check_catches_a_constant_length_output(sample_bars):
    """入力の長さに関わらず固定長を返す実装（長さがたまたま一致しない限り）も、
    長さチェックで検出できることを確認する。"""

    def fixed_length_100(bars: pd.DataFrame) -> pd.Series:
        return bars["bid_close"].iloc[:100]

    with pytest.raises(AssertionError):
        assert_truncation_invariant(fixed_length_100, sample_bars, TRUNCATE)
