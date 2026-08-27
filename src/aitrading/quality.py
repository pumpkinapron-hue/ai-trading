"""データ品質チェック。取得直後に走らせて meta.db に記録する。

週末クローズを欠損と区別することが要点。区別しないと毎週末に偽陽性が出て、
アラートとして機能しなくなる。日足・週足は夏時間で長さが変わる可変長なので
ここでは非対応（`timeframe.delta` が None の場合は ValueError）。日境界の暦計算は
`timeutil` に集約する方針（bars.py と同じ）で、ここで簡易的に再実装すると
Task 7 で実際に起きた「区切りが1時間ずれて先読みになる」バグと同じ構造の
間違いを繰り返しかねない。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from aitrading.timeutil import Timeframe, ensure_utc, is_market_open


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    expected_bars: int
    actual_bars: int
    duplicate_count: int
    bad_spread_count: int
    wide_spread_count: int
    price_jump_count: int
    longest_gap_minutes: float
    gaps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    *,
    jump_atr_multiple: float = 10.0,
    wide_spread_quantile: float = 0.999,
) -> QualityReport:
    """バー列の品質を検査する。index は `open_time`（tz-aware）。

    欠損は「市場が開いている時間」だけを母数にする（`is_market_open`）。
    週末クローズをそのまま欠損に数えると、毎週偽陽性が出てアラートとして
    機能しなくなる。
    """
    step = timeframe.delta
    if step is None:
        raise ValueError(
            f"{timeframe} は可変長（日足・週足）なので品質チェック非対応。"
            "夏時間で1本の長さが23/25時間に変わる日境界の暦計算をここで簡易的に"
            "再実装すると、区切りが1時間ずれて先読みになるおそれがある"
            "（timeutil に集約する方針そのものが崩れる）。"
        )

    # naive は ValueError、UTC以外の tz-aware は UTC に変換――timeutil 全体の規約
    # (ensure_utc) にそのまま乗る。ここで自前の tz 判定は書かない。
    index = ensure_utc(pd.DatetimeIndex(bars.index))
    bars = bars.set_axis(index, axis=0)

    bars, duplicate_count = _dedupe(bars)
    index = pd.DatetimeIndex(bars.index)

    gaps = _detect_gaps(index, step)
    longest_gap = max((g["minutes"] for g in gaps), default=0.0)
    expected_bars = _expected_bars(index, step)

    spread = bars["ask_close"] - bars["bid_close"]
    bad_spread_count = int((spread <= 0).sum())
    wide_spread_count = _wide_spread_count(spread, wide_spread_quantile)
    price_jump_count = _price_jump_count(bars, jump_atr_multiple)

    return QualityReport(
        symbol=symbol,
        timeframe=timeframe.value,
        expected_bars=expected_bars,
        actual_bars=len(bars),
        duplicate_count=duplicate_count,
        bad_spread_count=bad_spread_count,
        wide_spread_count=wide_spread_count,
        price_jump_count=price_jump_count,
        longest_gap_minutes=float(longest_gap),
        gaps=gaps,
    )


def _dedupe(bars: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """index の重複を畳む。複数回現れる時刻は最初の1本だけを残す。

    `Index.duplicated()` は（`Series` ではなく）ndarray を返す。`.loc` に
    そのまま渡すと index のラベル重複に関係なく常に位置で効く。もし代わりに
    `pd.Series` のブールマスクを渡していたら `.loc` はラベル整列を試み、
    重複ラベルがある入力では意図通りに動かない――ここは実際に確認した
    （報告の「変異検査/性能」節を参照）。
    """
    index = pd.DatetimeIndex(bars.index)
    duplicated = index.duplicated(keep="first")
    duplicate_count = int(duplicated.sum())
    deduped = bars.loc[~duplicated].sort_index()
    return deduped, duplicate_count


def _detect_gaps(index: pd.DatetimeIndex, step: pd.Timedelta) -> list[dict]:
    """市場が開いている時間だけを対象に欠損区間を検出する。

    全行を Python でループすると、Task 11 で扱う10年分の1分足（市場が
    開いている時間だけで約370万行）で遅すぎる。実データで隣接バーの間隔が
    `step` を超える行（＝穴の候補）はごく少数なので、まず pandas の
    ベクトル化演算で候補行だけを絞り込み、その「診断済みの穴」だけを
    Python でループして `is_market_open` に投げる。

    穴の長さは delta（カレンダー上の間隔）そのものではなく、その区間で
    実際に `is_market_open` が真を返す本数で決める。週末クローズの直前・
    直後に穴が接していても（delta が週末を含んで大きくなっていても）、
    週末ぶんの閉場時間は missing_bars に混ざらない。
    """
    if len(index) < 2:
        return []

    deltas = index.to_series().diff()
    # 先頭行は前が無い(NaT)。NaT との比較は False になるが、pandas の
    # バージョン間でこの暗黙の挙動に頼りたくないので明示的に潰しておく。
    #
    # `>` を `>=` に変えないこと。正しさは壊れない(delta == step の行は
    # start > last になり、下のループで空スパン=missing 0 として握りつぶされる
    # ので既存テストは何も検知しない)が、実データでは delta == step の行が
    # 大多数を占めるため、ほぼ全行を再び Python でループすることになり、
    # このベクトル化の意味が失われる(実測: 3か月分・約9.4万行で約1.3秒。
    # 本来は候補ゼロで即 return のはずの入力)。Task 11 の10年規模(約370万行)
    # では数十秒に劣化する計算になる。
    gap_mask = (deltas > step).fillna(False).to_numpy()
    if not gap_mask.any():
        return []

    ends = index[gap_mask]
    span_deltas = deltas.to_numpy()[gap_mask]

    gaps: list[dict] = []
    for end, raw_delta in zip(ends, span_deltas):
        delta = pd.Timedelta(raw_delta)
        start = end - delta + step
        last = end - step
        span = pd.date_range(start, last, freq=step)
        missing = int(is_market_open(span).sum()) if len(span) else 0
        if missing == 0:
            continue  # 穴の中身が週末クローズだけ。実データ欠損ではない
        gaps.append(
            {
                "from": str(start),
                "to": str(last),
                "missing_bars": missing,
                "minutes": missing * step.total_seconds() / 60.0,
            }
        )
    return gaps


def _expected_bars(index: pd.DatetimeIndex, step: pd.Timedelta) -> int:
    """観測された範囲のうち、市場が開いていたはずの本数。

    週末クローズぶんを母数からも除く。除かないと expected が実本数より
    常に大きくなり、毎週「本数が足りない」という偽陽性になる。
    """
    if len(index) < 2:
        return len(index)
    full = pd.date_range(index.min(), index.max(), freq=step)
    return int(is_market_open(full).sum())


def _wide_spread_count(spread: pd.Series, quantile: float) -> int:
    """分位点を超えて広いスプレッドの本数。

    分位点の基準は正のスプレッドだけから作る（0以下は `bad_spread_count`
    で別に数えており、分位点の基準を歪めたくない）。しきい値との比較は
    `>`（等号を含まない）なので、スプレッドが全部同じ値のデータでは
    しきい値そのものがその値になり、「超える」行は1つも出ない
    （test_wide_spread_quantile_flags_nothing_when_spread_is_constant で固定）。
    """
    positive = spread[spread > 0]
    if positive.empty:
        return 0
    threshold = positive.quantile(quantile)
    if not np.isfinite(threshold):
        return 0
    return int((spread > threshold).sum())


def _price_jump_count(bars: pd.DataFrame, jump_atr_multiple: float) -> int:
    """ATR（真の値幅の14本移動平均）の `jump_atr_multiple` 倍を超える価格変化の本数。

    既知の制約: `rolling(14, min_periods=14)` なので先頭13本はATRがNaNになり、
    その間に起きたジャンプは検出できない（NaNとの比較は常にFalseなので、
    黙って見逃す）。ウォームアップ前だけ別の値幅推定に差し替える手もあるが、
    その代替の妥当性を検証できていない段階で導入すると「検出できているように
    見えて実は違う基準で判定している」状態になりかねない。ここでは
    「まだ判定できない」を正直に保つ方を選んだ——
    test_price_jump_during_atr_warmup_is_not_detected がこの挙動を固定している。
    """
    mid = (bars["bid_close"] + bars["ask_close"]) / 2.0
    true_range = (bars["bid_high"] - bars["bid_low"]).abs()
    atr = true_range.rolling(14, min_periods=14).mean()
    jump = mid.diff().abs()
    return int((jump > atr * jump_atr_multiple).sum())
