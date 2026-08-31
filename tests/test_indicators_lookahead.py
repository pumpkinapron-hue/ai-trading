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
# そのため、1本刻みで「取りうる全ての接頭辞長」を試す。
#
# 上限を途中で打ち切ってはいけない。この検査は fn(bars)[:L] と fn(bars[:L]) を
# 比べるもので、L = len(bars) - truncate。切り落とし量の上限を150に固定すると
# 検査される L は 150〜299 だけになり、**先読みの影響が index 150 より前で
# 完結している場合は構造的に検出できない**（full 側と truncated 側が同じ計算を
# するため、差が出ない）。実測で素通りしたもの:
#   - rolling(20).mean().bfill()  ウォームアップのNaNを未来から埋める
#     （bar 0 が bar 19 を見る）→ truncate=281 でしか露見しない
#   - 先頭100本だけ center=True
#   - interpolate() での欠損埋め → truncate=294
#
# 全域スイープ（L=1..len-1）は「時点tの出力は bars[:t+1] だけの関数である」の
# 正確な言い換えになっている。9指標すべてで偽陽性ゼロ、コストは 1.4秒→2.8秒。
def truncation_sweep(bars: pd.DataFrame) -> range:
    """取りうる全ての切り落とし量。フィクスチャ長から決める（固定値にしない）。"""
    return range(1, len(bars))


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_look_ahead_at_various_truncations(name, sample_bars):
    fn = INDICATORS[name]
    for truncate in truncation_sweep(sample_bars):
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
    for truncate in truncation_sweep(sample_bars):
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


def _cheat_warmup_bfill(bars: pd.DataFrame) -> pd.Series:
    """ウォームアップのNaNを未来の値で埋める。bar 0 が bar 19 を見ている。

    影響が先頭19本の中で完結するので、切り落とし量の上限を打ち切った
    スイープでは**構造的に検出できない**（full側とtruncated側が同じ計算をする）。
    全域スイープでのみ露見する。Critical 1 の回帰テスト。
    """
    return bars["bid_close"].rolling(20, min_periods=20).mean().bfill()


def _cheat_resample_ffill(bars: pd.DataFrame) -> pd.Series:
    """上位足の特徴量。進行中の5分足の平均は未来の1分足を含む。

    実務で最も出やすい形の先読み。切り落とし量が5の倍数だと、境界が
    ちょうど5分足の区切りに揃うため差が出ない。**固定 TRUNCATE=10 は
    その盲点の真上に乗っている。** スイープの密度を保つ根拠。
    """
    close = bars["bid_close"]
    return close.resample("5min").mean().reindex(close.index, method="ffill").rename("x")


def _cheat_ulp_blend(bars: pd.DataFrame) -> pd.Series:
    """1本先の値を相対1e-13だけ混ぜる。既定の許容誤差(rtol~1e-5)には埋もれる。

    `assert_same_prefix` の `check_exact=True` を保つ根拠。先読みが無い純粋な
    因果計算は末尾を切ってもビット単位で不変になるので、厳密比較にして損はない。
    """
    close = bars["bid_close"]
    base = close.rolling(20, min_periods=20).mean()
    return (base + (close.shift(-1).fillna(0) - base.fillna(0)) * 1e-13).rename("x")


_CHEATERS = {
    "center=True のローリング窓": _cheat_center_window,
    "shift(-1) で1本先を見る": _cheat_shift_future,
    "全期間平均で正規化": _cheat_global_normalize,
    "全期間統計でzスコア化": _cheat_global_zscore,
    "bfill で未来の値を後ろ埋め": _cheat_bfill_from_future,
    "逆順累積和": _cheat_reverse_cumsum,
    "ウォームアップのNaNを未来から埋める": _cheat_warmup_bfill,
    "上位足へのresample+ffill": _cheat_resample_ffill,
    "1本先を相対1e-13だけ混ぜる": _cheat_ulp_blend,
}


def _first_truncation_that_catches(fn, bars: pd.DataFrame) -> int | None:
    """`fn` の先読みを最初に検出できた切り落とし量。検出できなければ None。"""
    for truncate in truncation_sweep(bars):
        try:
            assert_truncation_invariant(fn, bars, truncate)
        except AssertionError:
            return truncate
    return None


@pytest.mark.parametrize("label", sorted(_CHEATERS))
def test_detector_catches_a_deliberate_lookahead(label, sample_bars):
    """検査そのものが機能していることを確かめる。

    本番の指標が通るのと**同じ経路**（`assert_truncation_invariant` ＋
    全域スイープ）でチーターを流す。自己検証だけ別経路にすると、
    本番側の経路が弱っても自己検証が緑のままになる。
    """
    caught_at = _first_truncation_that_catches(_CHEATERS[label], sample_bars)
    assert caught_at is not None, f"検出器が「{label}」を素通りさせた"


def test_check_exact_is_load_bearing(sample_bars):
    """`assert_same_prefix` の `check_exact=True` が効いていることを固定する。

    これが無いと、既定の許容誤差(rtol~1e-5)に埋もれる程度の先読みを見逃す。
    将来「厳しすぎる」と言って外されたら、このテストが落ちる。
    """
    full = _cheat_ulp_blend(sample_bars)
    truncated = _cheat_ulp_blend(sample_bars.iloc[:-TRUNCATE])
    head = full.iloc[: len(truncated)]

    # 既定の許容誤差では見逃す
    pd.testing.assert_series_equal(head, truncated)
    # 厳密比較なら捕まる
    with pytest.raises(AssertionError):
        assert_same_prefix(full, truncated)


def test_sweep_density_is_load_bearing(sample_bars):
    """全域スイープが効いていることを固定する。

    `resample('5min')` の先読みは、切り落とし量が5の倍数だと境界が
    5分足の区切りに揃って差が出ない。固定 TRUNCATE=10 はその盲点の真上。
    将来スイープを粗くしたり範囲を狭めたりしたら、このテストが落ちる。
    """
    cheat = _cheat_resample_ffill

    # 固定10本では素通りする（＝固定値1点に頼ってはいけない証拠）
    assert_truncation_invariant(cheat, sample_bars, TRUNCATE)
    for blind in (5, 15, 20, 25):
        assert_truncation_invariant(cheat, sample_bars, blind)

    # スイープなら捕まる
    assert _first_truncation_that_catches(cheat, sample_bars) is not None


def test_full_sweep_is_load_bearing(sample_bars):
    """スイープの上限を打ち切ってはいけないことを固定する。

    ウォームアップ区間で完結する先読みは、切り落とし量が浅いうちは
    full 側と truncated 側が同じ計算をするので差が出ない。上限を
    len(bars)-1 まで伸ばして初めて露見する。
    """
    caught_at = _first_truncation_that_catches(_cheat_warmup_bfill, sample_bars)
    assert caught_at is not None
    # 打ち切ったスイープ（旧実装の range(1,151)）では見逃していた
    assert caught_at > 150, (
        f"このチーターは truncate={caught_at} で捕まった。"
        " 打ち切りスイープの回帰テストとして機能させるには150より大きい必要がある"
    )


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


# --- 退化した出力（全NaN・定数）への歯止め。
#
# レジストリ由来の自動検査（先読み・index保存・非破壊）は、値が意味を持つことを
# 一切要求しない。実測で、全NaN／全ゼロ／定数を返す実装はこれらを全部通過した。
# 設計意図は「値のテストを書き忘れても先読み検査だけは走る」ことなので仕様どおり
# ではあるが、「新しい指標を足して壊れて全NaNを返す」ケースが緑のまま入る。
# レジストリ側に1本足して塞ぐ。
def _columns_of(result):
    return [result] if isinstance(result, pd.Series) else [result[c] for c in result.columns]


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_produces_meaningful_values(name, sample_bars):
    """ウォームアップ後に有限値があり、かつ定数でないこと。

    sample_bars はランダムウォークなので、9指標のいずれも定数にはならない。
    将来「定数が正しい」指標（定数価格に対する hist_vol=0 など）を足すときは、
    そのときに除外リストを作ること――黙って通す穴のままにしない。
    """
    for column in _columns_of(INDICATORS[name](sample_bars)):
        finite = column[np.isfinite(column.to_numpy(dtype="float64"))]
        assert not finite.empty, f"{name}.{column.name} が有限値を1つも返していない"
        assert finite.nunique() > 1, f"{name}.{column.name} が定数を返している"
