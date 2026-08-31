"""データ品質チェック。取得直後に走らせて meta.db に記録する。

**市場カレンダーを知らない判定経路を作らないこと**が、このモジュールの一貫した要点。
週末クローズを欠損と数えれば毎週偽陽性が出て、アラートとして誰も見なくなる。
同じことが価格ジャンプ側でも起きるので（週明けの窓開け）、判定はすべて
「直前のバーと時間的に隣接しているか」を通す。

日足・週足は夏時間で長さが変わる可変長なのでここでは非対応
（`timeframe.delta` が None の場合は `ValueError`）。日境界の暦計算は
`timeutil` に集約する方針（bars.py と同じ）で、ここで簡易的に再実装すると
Task 7 で実際に起きた「区切りが1時間ずれて先読みになる」バグと同じ構造の
間違いを繰り返しかねない。

既知の制約: `is_market_open` は曜日しか見ないので、祝日クローズ（12/25・1/1・
グッドフライデーなど）は「市場が開いているのにバーが無い」＝欠損として報告される。
10年で20件前後になり、1件あたり1440分なので `longest_gap_minutes` がそれで飽和する。
**この値を単独の見出し数値に使わないこと。** 祝日表をどこから持ってくるかは
外部依存の判断なので Phase 0 では入れていない。
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
    conflicting_duplicate_count: int
    bad_spread_count: int
    wide_spread_count: int
    wide_spread_threshold: float
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
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
    jump_atr_multiple: float = 10.0,
    wide_spread_quantile: float = 0.999,
) -> QualityReport:
    """バー列の品質を検査する。index は `open_time`（tz-aware）。

    `expected_start` / `expected_end` は「要求した範囲」を半開区間 `[start, end)` で
    渡すもので、`BarSource.fetch` の契約と同じ形。**省略すると、観測されたデータ自身の
    端が母数になるため、先頭や末尾がまるごと落ちている欠損は定義上ゼロになる**
    （取得が途中で切れたときこそ「問題なし」と報告されてしまう）。取得側は
    要求範囲を知っているので、必ず渡すこと。
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

    bars, duplicate_count, conflicting_count = _dedupe(bars)
    index = pd.DatetimeIndex(bars.index)

    expected_times = _expected_times(index, step, expected_start, expected_end)
    gaps = _detect_gaps(index, expected_times, step)
    longest_gap = max((g["minutes"] for g in gaps), default=0.0)

    spread = bars["ask_close"] - bars["bid_close"]
    bad_spread_count = int((spread <= 0).sum())
    wide_spread_count, wide_spread_threshold = _wide_spread(spread, wide_spread_quantile)
    price_jump_count = _price_jump_count(bars, step, jump_atr_multiple)

    return QualityReport(
        symbol=symbol,
        timeframe=timeframe.value,
        expected_bars=len(expected_times),
        actual_bars=len(bars),
        duplicate_count=duplicate_count,
        conflicting_duplicate_count=conflicting_count,
        bad_spread_count=bad_spread_count,
        wide_spread_count=wide_spread_count,
        wide_spread_threshold=wide_spread_threshold,
        price_jump_count=price_jump_count,
        longest_gap_minutes=float(longest_gap),
        gaps=gaps,
    )


def _dedupe(bars: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """index の重複を畳む。複数回現れる時刻は最初の1本だけを残す。

    「同じ値の重複」（再取得で起きる無害なもの）と「違う値の重複」
    （データソース側が過去を書き換えた疑い）は品質レポートとして別物なので、
    別カウンタにする。`Lake._merge_year` は後者を `ValueError` にするが、
    こちらは報告するのが仕事なので、見つけた異常を投げて報告不能にはしない。

    `Index.duplicated()` は（`Series` ではなく）ndarray を返す。`.loc` に
    そのまま渡すと index のラベル重複に関係なく常に位置で効く。もし代わりに
    `pd.Series` のブールマスクを渡していたら `.loc` はラベル整列を試み、
    重複ラベルがある入力では意図通りに動かない。
    """
    index = pd.DatetimeIndex(bars.index)
    duplicated = index.duplicated(keep="first")
    duplicate_count = int(duplicated.sum())

    conflicting_count = 0
    if duplicate_count:
        # 同じ時刻の行がすべて同一内容かどうか。値が食い違う時刻の数を数える。
        conflicting_count = sum(
            1
            for _, group in bars.groupby(level=0, sort=False)
            if len(group) > 1 and not _all_rows_equal(group)
        )

    deduped = bars.loc[~duplicated].sort_index()
    return deduped, duplicate_count, int(conflicting_count)


def _all_rows_equal(frame: pd.DataFrame) -> bool:
    """同じ時刻に属する複数行が、全列で完全に同じ値か。"""
    return bool(frame.eq(frame.iloc[0], axis=1).all().all())


def _expected_times(
    index: pd.DatetimeIndex,
    step: pd.Timedelta,
    expected_start: pd.Timestamp | None,
    expected_end: pd.Timestamp | None,
) -> pd.DatetimeIndex:
    """市場が開いていた「あるはずのバーの時刻」。欠損検出と母数の唯一の出どころ。

    欠損の検出と母数の計算を別々の経路で書くと、両者が食い違ったときに
    `expected = actual + 欠損` の不変条件が黙って壊れる。ここを1つにして、
    欠損は「あるはずの時刻のうち実在しないもの」と定義する。
    """
    if expected_start is not None:
        start = ensure_utc(pd.DatetimeIndex([expected_start]))[0]
    elif len(index):
        start = index.min()
    else:
        return index[:0]

    if expected_end is not None:
        end = ensure_utc(pd.DatetimeIndex([expected_end]))[0]
    elif len(index):
        # 省略時は観測された最後のバーの終わり。半開区間なので step を足す。
        end = index.max() + step
    else:
        return index[:0]

    grid = pd.date_range(start, end, freq=step, inclusive="left")
    if len(grid) == 0:
        return grid
    return grid[_market_open_during(grid, step).to_numpy()]


def _market_open_during(times: pd.DatetimeIndex, step: pd.Timedelta) -> pd.Series:
    """バー `[t, t+step)` の間に市場が開いている瞬間があるか。

    `open_time` の1点だけで判定すると、バーが長い時間軸で壊れる。たとえば4時間足の
    `[20:00, 24:00)` バケットは、日曜の再開（NY17:00 = 冬22:00Z / 夏21:00Z）が
    バケットの内側に落ちるため、実在するバーが母数から外れて
    `expected_bars < actual_bars` になる（毎週末、確実に起きる）。

    両端だけ見れば足りるのは、FX市場が「日曜NY17:00〜金曜NY17:00の連続した1本の
    開場」だからで、開場も閉場も4時間足より遥かに長い。区間が丸ごと開場の内側に
    収まることはあっても、開場が区間の内側に収まりきることはない。
    """
    opened = is_market_open(times)
    if step > pd.Timedelta(minutes=1):
        tail = is_market_open(times + step - pd.Timedelta(minutes=1))
        opened = opened | tail.to_numpy()
    return opened


def _detect_gaps(
    index: pd.DatetimeIndex, expected_times: pd.DatetimeIndex, step: pd.Timedelta
) -> list[dict]:
    """あるはずの時刻のうち実在しないものを、連続した区間にまとめる。

    要求範囲を渡していれば、先頭・末尾がまるごと落ちている欠損もここに出る
    （隣接バーの間隔だけを見る実装だと、端の欠損は原理的に見えない）。
    """
    missing = expected_times.difference(index)
    if len(missing) == 0:
        return []

    # 連続した欠損を1区間にまとめる。step ちょうど離れていれば同じ区間。
    # 整数（asi8）で比較しないこと。pandas 3 の datetime64 は分解能が [us] なので、
    # ナノ秒前提の step.value と突き合わせると全件が「不連続」になる（実際に踏んだ）。
    continues = (missing.to_series().diff() == step).to_numpy()
    breaks = np.flatnonzero(~continues[1:]) + 1
    runs = np.split(np.arange(len(missing)), breaks)

    return [
        {
            "from": str(missing[run[0]]),
            "to": str(missing[run[-1]]),
            "missing_bars": len(run),
            "minutes": len(run) * step.total_seconds() / 60.0,
        }
        for run in runs
    ]


def _wide_spread(spread: pd.Series, quantile: float) -> tuple[int, float]:
    """分位点を超えて広いスプレッドの本数と、そのしきい値。

    分位点は定義上「上位 (1-q)」を切るので、本数そのものはデータが綺麗でも汚くても
    ほぼ `n × (1-q)` になり、それ単体では情報をほとんど持たない。しきい値の値のほうが
    「スプレッドの分布がいつもと違うか」を見るのに使えるので、一緒に報告する。

    分位点の基準は正のスプレッドだけから作る（0以下は `bad_spread_count` で別に
    数えており、分位点の基準を歪めたくない）。比較は `>`（等号を含まない）なので、
    スプレッドが全部同じ値のデータでは「超える」行は1つも出ない。
    """
    positive = spread[spread > 0]
    if positive.empty:
        return 0, 0.0
    threshold = positive.quantile(quantile)
    if not np.isfinite(threshold):
        return 0, 0.0
    return int((spread > threshold).sum()), float(threshold)


def _price_jump_count(
    bars: pd.DataFrame, step: pd.Timedelta, jump_atr_multiple: float
) -> int:
    """ATR（値幅の14本移動平均）の `jump_atr_multiple` 倍を超える価格変化の本数。

    **直前のバーと時間的に隣接している行だけを判定する。** 隣接していない行の
    `mid.diff()` は「バーからバーへの変化」ではなく「穴を跨いだ変化」で、
    週明けの窓開けがそのまま出る。しかも `rolling(14)` は行ベースなので、
    週明けバーのATR窓は金曜クローズ直前の十数分――週で最も流動性が枯れた時間帯――
    だけで構成される。分子が週で最大になる瞬間に分母が週で最小になるため、
    実測では数pipsの窓開けで毎週100%発火した。これでは週末カウンタでしかない。

    ここを `true_range` の定義（`|high - 前close|` を含む本来のTR）で解こうとしないこと。
    実測では発火の分岐点が約3倍動くだけで、しかもそれが効く理由は「窓開けバー自身の
    TRが同じ行のATRに 1/14 だけ混ざる」という偶然の自己減衰にすぎない。
    ATRを `.shift(1)` する（異常検知としてはむしろ自然な変更）と丸ごと元に戻る。

    既知の制約: `rolling(14, min_periods=14)` なので先頭13本はATRがNaNになり、
    その間に起きたジャンプは検出できない（NaNとの比較は常にFalseなので黙って見逃す）。
    ウォームアップ前だけ別の値幅推定に差し替える手もあるが、その代替の妥当性を
    検証できていない段階で導入すると「検出できているように見えて実は違う基準で
    判定している」状態になりかねない。ここでは「まだ判定できない」を正直に保つ。
    """
    index = pd.DatetimeIndex(bars.index)
    mid = (bars["bid_close"] + bars["ask_close"]) / 2.0
    true_range = (bars["bid_high"] - bars["bid_low"]).abs()
    atr = true_range.rolling(14, min_periods=14).mean()
    jump = mid.diff().abs()
    adjacent = index.to_series().diff() == step
    return int(((jump > atr * jump_atr_multiple) & adjacent.to_numpy()).sum())
