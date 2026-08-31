"""期待値スキャン（`edge_scan`）のテスト。

計画の草案（`docs/plans/2026-08-24-phase0-implementation.md` の Task 10）を
そのまま写していない。指摘された A〜H を実測して検証し、直した内容の詳細は
`src/aitrading/edge_scan.py` のモジュール docstring と
`.superpowers/sdd/2026-08-24-phase0-implementation/task-10-report.md` を参照。

このファイルの構成:
- 基本の集計（草案の9テストを踏襲、値は変更なし）
- G: `n_signals` と `HorizonStats.n` の食い違い
- group_by の検査（Session Enum の文字列化の落とし穴を含む）
- 入力検証（direction / pip / horizon）
- A: 重複サンプルと信頼区間（最重要。効いていることを複数の角度で固定する）
- B: ロック期間解除の監査記録の抜け穴
- D: NaN を含む結果の JSON 化
- E: pip の既定値（USD/JPY 前提）
- F: win_rate のちょうど0の扱い
- H: signal の reindex/dtype
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aitrading.config import load_settings
from aitrading.edge_scan import scan, scan_period
from aitrading.storage.meta import Meta


@pytest.fixture
def rising_bars():
    """1分ごとに mid が +1pip ずつ上がる、スプレッド2pip固定のバー。"""
    n = 200
    index = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    mid_price = 150.00 + np.arange(n) * 0.01
    body = {"close_time": index + pd.Timedelta(minutes=1), "volume": [100.0] * n}
    for f in ("open", "high", "low", "close"):
        body[f"bid_{f}"] = mid_price - 0.01
        body[f"ask_{f}"] = mid_price + 0.01
    return pd.DataFrame(body, index=index).rename_axis("open_time")


def _random_walk_bars(
    seed: int, n: int, sigma: float = 0.02, spread: float = 0.02, start: str = "2026-01-05 00:00"
) -> pd.DataFrame:
    """真の優位性が0（mid-to-mid の期待リターンが厳密にゼロ）のランダムウォーク。

    検証A（重複サンプルと信頼区間）の再現用。`sigma`/`spread` は
    `tests/conftest.py` の `sample_bars` と同スケール。
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    mid_price = 150.0 + np.cumsum(rng.normal(0, sigma, n))
    body = {"close_time": index + pd.Timedelta(minutes=1), "volume": [100.0] * n}
    for f in ("open", "high", "low", "close"):
        body[f"bid_{f}"] = mid_price - spread / 2
        body[f"ask_{f}"] = mid_price + spread / 2
    return pd.DataFrame(body, index=index).rename_axis("open_time")


# --- 基本の集計 ---


def test_computes_mean_return_per_horizon(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10, 20, 30]] = True
    result = scan(rising_bars, signal, horizons=(5,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.n == 3
    # 5分で mid が 5pip 上がる
    assert stats.mean_pips == pytest.approx(5.0, abs=1e-6)
    assert stats.win_rate == pytest.approx(1.0)


def test_spread_is_deducted(rising_bars):
    """生のリターンで見ると、ほとんどの指標が有望に見えてしまう。"""
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True
    gross = scan(rising_bars, signal, horizons=(5,), deduct_spread=False).horizons[0]
    net = scan(rising_bars, signal, horizons=(5,), deduct_spread=True).horizons[0]
    # 買いはAsk、決済はBid。往復で2pip分不利になる
    assert net.mean_pips == pytest.approx(gross.mean_pips - 2.0, abs=1e-6)


def test_short_direction_flips_sign(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True
    result = scan(rising_bars, signal, horizons=(5,), direction="short", deduct_spread=False)
    assert result.horizons[0].mean_pips == pytest.approx(-5.0, abs=1e-6)


def test_signals_without_enough_future_bars_are_dropped(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[len(rising_bars) - 2]] = True
    result = scan(rising_bars, signal, horizons=(60,), deduct_spread=False)
    assert result.horizons[0].n == 0
    assert result.horizons[0].n_eff == 0


def test_mean_median_win_rate_std_are_computed_correctly_with_varied_values():
    """rising_bars は線形な系列なので、シグナル位置ごとのリターンが全部同じ値
    になり、mean と median が偶然一致してしまう（取り違えの変異を検出できない）。

    ばらつきのある値で、mean/median/win_rate/std がそれぞれ別々の正しい値に
    なることを固定する。手作りの価格系列で horizon=1 のリターンを
    [1, 2, 3, 4, 100] pips に固定し、期待値は実際に numpy で計算して検証済み
    （手計算の丸め誤りを持ち込まないため）。
    """
    mid = [150.00, 150.01, 150.01, 150.03, 150.03, 150.06, 150.06, 150.10, 150.10, 151.10]
    n = len(mid)
    index = pd.date_range("2026-01-05", periods=n, freq="1min", tz="UTC")
    body = {"close_time": index + pd.Timedelta(minutes=1), "volume": [1.0] * n}
    for f in ("open", "high", "low", "close"):
        body[f"bid_{f}"] = [m - 0.005 for m in mid]
        body[f"ask_{f}"] = [m + 0.005 for m in mid]
    bars = pd.DataFrame(body, index=index).rename_axis("open_time")

    signal = pd.Series(False, index=bars.index)
    signal.iloc[[0, 2, 4, 6, 8]] = True
    result = scan(bars, signal, horizons=(1,), deduct_spread=False)
    stats = result.horizons[0]

    assert stats.n == 5
    assert stats.mean_pips == pytest.approx(22.0)
    assert stats.median_pips == pytest.approx(3.0)
    assert stats.mean_pips != stats.median_pips
    assert stats.win_rate == pytest.approx(1.0)
    assert stats.std_pips == pytest.approx(43.617656975128774, abs=1e-6)


# --- G: n_signals と HorizonStats.n の食い違い ---


def test_n_signals_stays_constant_while_horizon_n_shrinks(rising_bars):
    """n_signals はシグナルが立った本数そのもの。HorizonStats.n は、そのうち
    horizon 本先のバーが実在して実際に集計できた本数——horizon が長いほど
    末尾付近のシグナルが軒並み落ち、両者の差が広がる。読み違えると
    「n_signals件を全部集計した」と誤解しかねない。
    """
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[list(range(len(rising_bars) - 10, len(rising_bars)))] = True
    result = scan(rising_bars, signal, horizons=(1, 60), deduct_spread=False)

    assert result.n_signals == 10
    n_by_horizon = {s.horizon: s.n for s in result.horizons}
    # horizon=1: 末尾1本（最後のバー自身）だけ先が無い -> 9本集計できる
    assert n_by_horizon[1] == 9
    # horizon=60: 直近10本はどれも60本先が無い -> 0本
    assert n_by_horizon[60] == 0
    assert n_by_horizon[60] < result.n_signals
    assert n_by_horizon[1] < result.n_signals


# --- group_by ---


def test_group_by_session_splits_results(rising_bars):
    signal = pd.Series(True, index=rising_bars.index)
    result = scan(rising_bars, signal, horizons=(5,), group_by="session")
    assert result.by_group, "セッション別の集計が空"
    assert all(isinstance(k, str) for k in result.by_group)


def test_group_by_session_labels_are_plain_values(rising_bars):
    """`Session` は `class Session(str, Enum)`。`str(Session.TOKYO)` は
    `"Session.TOKYO"`（Enumの既定の__str__）になるが、メンバー自身がstrの
    インスタンスでもあるため、pandasの `.astype(str)` は `__str__` を呼ばず
    値のペイロード（`"TOKYO"`）を素通しする（実測確認済み）。ここが崩れて
    `"Session.TOKYO"` 形式に戻ると、ダッシュボードのラベル表示やフィルタが
    壊れる。
    """
    signal = pd.Series(True, index=rising_bars.index)
    result = scan(rising_bars, signal, horizons=(5,), group_by="session")
    assert set(result.by_group).issubset(
        {"TOKYO", "LONDON", "NEWYORK", "LDN_NY_OVERLAP", "OFF"}
    )
    assert not any(k.startswith("Session.") for k in result.by_group)


def test_group_by_rejects_unsupported_value(rising_bars):
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(ValueError, match="層別"):
        scan(rising_bars, signal, horizons=(5,), group_by="hour")


# --- 入力検証 ---


def test_scan_rejects_unknown_direction(rising_bars):
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(ValueError, match="方向"):
        scan(rising_bars, signal, direction="both")


def test_scan_rejects_nonpositive_pip(rising_bars):
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(ValueError, match="pip"):
        scan(rising_bars, signal, pip=0.0)
    with pytest.raises(ValueError, match="pip"):
        scan(rising_bars, signal, pip=-0.01)


def test_scan_rejects_nonpositive_horizon(rising_bars):
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(ValueError, match="horizon"):
        scan(rising_bars, signal, horizons=(0,))
    with pytest.raises(ValueError, match="horizon"):
        scan(rising_bars, signal, horizons=(5, -1))


# --- A: 重複サンプルと信頼区間（最重要） ---
#
# ci95_pips を素の n で `1.96 * std / sqrt(n)` にすると、シグナルが密に出る
# ケースで重複した（≒同じ情報しか持たない）サンプルを独立扱いしてしまい、
# 信頼区間が実際よりはるかに狭くなる。実測（ランダムウォーク、horizon=60、
# 乱数シードを変えて数百回試行）では、本来5%程度であるべき「0を跨がない
# （＝有意に見える）」頻度が70〜82%に達した。詳細は
# `src/aitrading/edge_scan.py` の `_effective_n` docstring と
# task-10-report.md 参照。


def test_effective_n_is_less_than_n_when_signals_overlap_densely(rising_bars):
    """rising_bars（200本）に毎足シグナル、horizon=60。隣接シグナルの
    horizon本先までのリターン窓はほぼ全て重なるので、n_eff は n より
    はっきり小さくなるはず。"""
    signal = pd.Series(True, index=rising_bars.index)
    result = scan(rising_bars, signal, horizons=(60,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.n == 140
    assert stats.n_eff == 3  # 実測値。直感的には n/horizon = 140/60 ≈ 2.33 に近い
    assert stats.n_eff < stats.n


def test_effective_n_equals_n_when_signals_do_not_overlap(rising_bars):
    """シグナル同士が horizon 本以上離れていれば、リターン窓は重ならないので
    n_eff は n から不必要に縮められないはず（`n/horizon` 固定の補正だと
    ここが壊れて過度に保守的になる。`_effective_n` docstring 参照）。
    """
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[::10] = True  # horizon=5 に対して常に10本以上離れている
    result = scan(rising_bars, signal, horizons=(5,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.n == 20
    assert stats.n_eff == 20


def test_ci_half_width_is_derived_from_effective_n(rising_bars):
    """ci95_pips の半幅が、実装の中で n ではなく n_eff から計算されている
    ことを、モジュールが返した std_pips/n_eff から検算して固定する。
    この式を n_eff から n に戻す変異はこのテストで検出できる。
    """
    signal = pd.Series(True, index=rising_bars.index)
    result = scan(rising_bars, signal, horizons=(60,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.n_eff < stats.n  # 前提: このデータで重複が起きている

    lo, hi = stats.ci95_pips
    expected_half = 1.96 * stats.std_pips / np.sqrt(stats.n_eff)
    assert (hi - lo) / 2 == pytest.approx(expected_half, rel=1e-9)

    naive_half = 1.96 * stats.std_pips / np.sqrt(stats.n)
    assert expected_half > naive_half * 3, "n_effによる補正の効果が小さすぎる"


def test_effective_n_cluster_boundary_is_inclusive(rising_bars):
    """`_effective_n` はシグナル間隔が「horizon本以上」でクラスタを区切る
    （`>=`）。変異検査で判明: 間隔がちょうど horizon のときにクラスタを
    区切るか区切らないか（`>=` か `>` か）は、多くの入力で最終的な n_eff の
    値を偶然変えない（クラスタをまたいでも `floor(span/horizon)` の計算が
    同じ答えに収束するケースが大半）。この境界を見分けるには、間隔の一部が
    ちょうど horizon で、かつ他のクラスタの幅が horizon の倍数でない
    （＝端数を持つ）組み合わせが要る。

    horizon=10, シグナル位置 [0, 7, 17, 24]（間隔 [7, 10, 7]）:
    - `>=`（正しい実装）: 間隔10のところだけクラスタが切れる ->
      {0,7}(幅7) と {17,24}(幅7) の2クラスタ -> n_eff = (7//10+1)+(7//10+1) = 2
    - `>`（間隔がちょうどhorizonでは切れない、という変異）: どこも切れず
      1クラスタ {0,7,17,24}(幅24) -> n_eff = 24//10+1 = 3

    実際に `>` へ変異させて実行し、この期待値どおりに 2 から 3 へ変わって
    テストが落ちることを確認済み（task-10-report.md 変異検査参照）。
    """
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[0, 7, 17, 24]] = True
    result = scan(rising_bars, signal, horizons=(10,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.n == 4
    assert stats.n_eff == 2


def test_naive_formula_would_have_falsely_flagged_noise_as_significant():
    """具体的な乱数シード（seed=0, n=400, horizon=60）で、生の n を使う
    信頼区間は0を跨がない（＝誤って「有意」と判定する）が、実装（n_eff補正
    あり）は0を跨ぐ（＝正しく「わからない」と言う）ことを固定する。真の
    優位性が0と分かっているデータでの実例（task-10-report.md 検証A）。
    """
    bars = _random_walk_bars(seed=0, n=400)
    signal = pd.Series(True, index=bars.index)
    result = scan(bars, signal, horizons=(60,), deduct_spread=False)
    stats = result.horizons[0]

    assert stats.n == 340
    assert stats.n_eff == 6

    lo, hi = stats.ci95_pips
    assert lo <= 0.0 <= hi, "n_eff補正後のCIは0を含むはず（有意とは言えない）"

    naive_half = 1.96 * stats.std_pips / np.sqrt(stats.n)
    naive_lo, naive_hi = stats.mean_pips - naive_half, stats.mean_pips + naive_half
    assert not (naive_lo <= 0.0 <= naive_hi), (
        "生の n を使うと0を跨がないはず（ノイズを「有意な優位性」と誤判定する"
        "実例。この対比が壊れたらフィクスチャ自体を見直すこと）"
    )


def test_ci_false_positive_rate_is_roughly_calibrated_across_seeds():
    """優位性ゼロのランダムウォークに毎足シグナルを立てて horizon=60 で scan
    したとき、95%CIが0を除外する（＝「有意」に見える）頻度は本来5%程度で
    あるべき統計量。実測（task-10-report.md）では seed 60本で 10.0% 程度に
    収まっている。ここでは緩めの上限（30%）だけを固定し、「n_eff補正を外して
    生のnに戻す」退行（実測70〜82%）を確実に検出できるようにする——閾値を
    5%近くまで詰めると、乱数の巡り合わせだけで時々failするテストになる。
    """
    n_seeds = 60
    false_positive = 0
    for seed in range(n_seeds):
        bars = _random_walk_bars(seed=seed + 900, n=500)
        signal = pd.Series(True, index=bars.index)
        result = scan(bars, signal, horizons=(60,), deduct_spread=False)
        lo, hi = result.horizons[0].ci95_pips
        if not (lo <= 0.0 <= hi):
            false_positive += 1

    rate = false_positive / n_seeds
    assert rate < 0.30, (
        f"誤検出率が高すぎる({rate:.1%})。ci95_pipsの計算がn_effを使わず"
        "生のnを使う式に戻っていないか確認すること"
        "（本来5〜10%程度、生のnを使うと実測70%超）"
    )


# --- B: ロック期間解除の監査記録 ---


def test_scan_period_refuses_locked_period_without_reason(rising_bars):
    settings = load_settings()
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(PermissionError, match="oos"):
        scan_period(rising_bars, signal, settings, "oos")


def test_scan_period_refuses_locked_period_without_meta(rising_bars):
    """unlock_reason だけ渡して meta を省略すると、理由さえ書けば監査記録
    （meta.db）が残らないままロック期間を覗ける抜け穴になる。「解除は記録に
    残る」という設計（設計文書 §8 (3)）の前提を守るため、両方揃わなければ
    拒否する。
    """
    settings = load_settings()
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(ValueError, match="meta"):
        scan_period(
            rising_bars, signal, settings, "oos", unlock_reason="理由はあるがmetaが無い"
        )


def test_scan_period_records_unlock(tmp_path, rising_bars):
    settings = load_settings()
    meta = Meta(tmp_path / "meta.db")
    signal = pd.Series(True, index=rising_bars.index)
    scan_period(
        rising_bars, signal, settings, "oos",
        meta=meta, unlock_reason="戦略v1.2の最終確認",
    )
    unlocks = meta.oos_unlocks()
    assert len(unlocks) == 1
    assert unlocks[0]["reason"] == "戦略v1.2の最終確認"


def test_scan_period_on_training_needs_no_reason(rising_bars):
    settings = load_settings()
    signal = pd.Series(True, index=rising_bars.index)
    result = scan_period(rising_bars, signal, settings, "training")
    # rising_bars は2026年なので training(〜2021) には入らない
    assert result.n_signals == 0


def test_scan_period_does_not_record_for_unlocked_period_even_with_meta(tmp_path, rising_bars):
    """ロックされていない期間は unlock_reason 不要で通る。meta を渡しても
    そもそも「解除」していないので record_oos_unlock は呼ばれないはず
    （解除していない集計まで記録すると、監査ログの意味が薄まる）。
    """
    settings = load_settings()
    meta = Meta(tmp_path / "meta.db")
    signal = pd.Series(True, index=rising_bars.index)
    scan_period(rising_bars, signal, settings, "training", meta=meta)
    assert meta.oos_unlocks() == []


# --- D: NaN を含む結果の JSON 化 ---


def test_to_dict_is_json_serializable(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True
    json.dumps(scan(rising_bars, signal, horizons=(5,)).to_dict(), default=str)


def test_to_dict_converts_nan_to_none_for_strict_json(rising_bars):
    """n=0（集計できたサンプルが無い）のとき mean_pips 等が NaN になる。
    Python の `json.dumps` は既定で NaN を非標準の `NaN` トークンとして
    素通しする（`allow_nan=True` が既定。RFC 8259 上は不正で、多くのJS実装の
    `JSON.parse` は読めない）。`to_dict()` は None に変換していて、
    json.dumps の出力に `NaN` という文字列が出ないことを確認する。
    """
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[len(rising_bars) - 1]] = True  # horizon分の先が無い -> n=0
    result = scan(rising_bars, signal, horizons=(60,))
    as_dict = result.to_dict()

    assert as_dict["horizons"][0]["mean_pips"] is None
    assert as_dict["horizons"][0]["ci95_pips"] == (None, None)

    dumped = json.dumps(as_dict)  # default=str が無くても素で通るはず
    assert "NaN" not in dumped

    round_tripped = json.loads(dumped)
    assert round_tripped["horizons"][0]["mean_pips"] is None
    assert round_tripped["horizons"][0]["ci95_pips"] == [None, None]


# --- E: pip の既定値（USD/JPY 前提） ---


def test_default_pip_is_usdjpy_scale(rising_bars):
    """既定 pip=0.01（`DEFAULT_PIP_USDJPY`）は rising_bars のような円建て
    価格に対して正しい pips 換算になる。EUR/USD 等の非JPYペア（1pip=0.0001）
    に使う場合は呼び出し側が pip= を明示しなければならない
    （`DEFAULT_PIP_USDJPY` のdocstring参照）。誤って他ペア用の粒度
    （0.0001）で呼ぶと、同じ価格変化が100倍のpipsに換算されてしまう
    ——桁がずれる実害をここで明示しておく。
    """
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True

    default_result = scan(rising_bars, signal, horizons=(5,), deduct_spread=False)
    explicit_result = scan(
        rising_bars, signal, horizons=(5,), deduct_spread=False, pip=0.01
    )
    assert default_result.horizons[0].mean_pips == pytest.approx(5.0)
    assert explicit_result.horizons[0].mean_pips == pytest.approx(5.0)

    wrong_pip_result = scan(
        rising_bars, signal, horizons=(5,), deduct_spread=False, pip=0.0001
    )
    assert wrong_pip_result.horizons[0].mean_pips == pytest.approx(500.0)


# --- F: win_rate のちょうど0の扱い ---


def test_win_rate_counts_exact_zero_as_not_a_win():
    """gross=0（ちょうど利益ゼロ）は勝ちに数えない——損益がプラスであることを
    「勝ち」の定義にする以上、建値どんとんは負け側に含めるのが妥当という判断
    （`_stats` のコメント参照）。ミッドが完全に横ばいの区間に対して
    deduct_spread=False で scan すると、全リターンがちょうど0になり、
    win_rate は0になるはず。
    """
    n = 20
    index = pd.date_range("2026-01-05", periods=n, freq="1min", tz="UTC")
    body = {"close_time": index + pd.Timedelta(minutes=1), "volume": [1.0] * n}
    for f in ("open", "high", "low", "close"):
        body[f"bid_{f}"] = [149.99] * n
        body[f"ask_{f}"] = [150.01] * n
    bars = pd.DataFrame(body, index=index).rename_axis("open_time")

    signal = pd.Series(True, index=bars.index)
    result = scan(bars, signal, horizons=(5,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.mean_pips == pytest.approx(0.0, abs=1e-9)
    assert stats.win_rate == pytest.approx(0.0), "ちょうど0は勝ちに数えないはず"


# --- H: signal の reindex/dtype ---


def test_partial_signal_index_is_aligned_and_treated_as_bool(rising_bars):
    """signal が bars の一部の時刻しかカバーしない（例えば指標のウォームアップ
    分だけ先頭が欠けている）場合でも、reindex で補われた分は False 扱いに
    なることを確認する。pandas 3 では `Series[bool].reindex(...)` の後の
    `.fillna(False)` が bool dtype に戻らず object dtype のまま残る
    （実測確認済み。`-W error` でも警告は出ない）ので、実装側の
    `.astype(bool)` が効いていないと後段の `.where(signal)` が正しく
    マスクとして働かない可能性がある。
    """
    partial_index = rising_bars.index[50:]  # 先頭50本を持たない signal
    signal = pd.Series(True, index=partial_index)
    result = scan(rising_bars, signal, horizons=(5,), deduct_spread=False)
    # reindexで補われた先頭50本はFalse扱いになるので、シグナル数は
    # partial_indexの長さと一致するはず
    assert result.n_signals == len(partial_index) == 150
