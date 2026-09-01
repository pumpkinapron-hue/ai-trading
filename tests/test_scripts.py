"""scripts/fetch_data.py と scripts/build_bars.py のテスト。

`scripts/` はパッケージではない（`pyproject.toml` の `packages` は `src/aitrading`
のみ）ので、`sys.path` にディレクトリを足してから素のモジュールとして import する。
これは Task 11 の草案（docs/plans）が採用していたのと同じパターンで、テスト側にだけ
必要な配線（fetch_data.py / build_bars.py 自身の `sys.path` 操作とは別物。
concern G はそちら――CLIスクリプト自身が `src/` を通す操作――が不要という指摘であり、
テストがスクリプトを見つけるためのこの1行はそれとは別に必要）。
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_bars  # noqa: E402
import fetch_data  # noqa: E402

from aitrading.config import Period, Settings  # noqa: E402
from aitrading.datasource.base import validate_bars  # noqa: E402
from aitrading.quality import QualityReport  # noqa: E402
from aitrading.storage.lake import Lake  # noqa: E402
from aitrading.storage.meta import Meta  # noqa: E402
from aitrading.timeutil import Timeframe  # noqa: E402

from tests.helpers import minute_bars  # noqa: E402


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


# ============================================================================
# ダミーのデータソース（ネットワークに一切触れない）
# ============================================================================


class FakeSource:
    """要求された [start, end) ぶんの決定的な1分足を返すダミー。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def fetch(self, symbol, timeframe, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((symbol, timeframe, start, end))
        minutes = int((end - start).total_seconds() // 60)
        return minute_bars(str(start), max(minutes, 1)).reset_index()


class BrokenChunkSource:
    """指定した開始時刻のチャンクだけ、Ask が Bid を下回る壊れたバーを返すダミー。

    **本物の `BarSource` 実装と同じく、返す前に `validate_bars` を通す。**
    実装(`DukascopySource`)は `normalize()` の最終行で必ず `validate_bars` を
    呼ぶので、壊れたバーは `lake.save()` に届く前に `source.fetch()` の中で
    `ValueError` になる。ここを素通しするダブルにすると、**本番では絶対に
    通らない経路だけを検証することになり、隔離が実運用で一度も発火しない
    ことを見逃す**（実際にそうなっていた）。
    """

    def __init__(self, broken_start: pd.Timestamp):
        self.broken_start = pd.Timestamp(broken_start)
        self.calls: list[tuple] = []

    def fetch(self, symbol, timeframe, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((symbol, timeframe, start, end))
        minutes = int((end - start).total_seconds() // 60)
        df = minute_bars(str(start), max(minutes, 1)).reset_index()
        if start == self.broken_start:
            df.loc[0, "ask_close"] = df.loc[0, "bid_close"] - 0.10
        return validate_bars(df, timeframe)


class SometimesEmptySource:
    """指定した開始時刻のチャンクだけ0本を返すダミー（休場を模す）。"""

    def __init__(self, empty_start: pd.Timestamp):
        self.empty_start = pd.Timestamp(empty_start)
        self.calls: list[tuple] = []

    def fetch(self, symbol, timeframe, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        self.calls.append((symbol, timeframe, start, end))
        if start == self.empty_start:
            return minute_bars(str(start), 0).reset_index()
        minutes = int((end - start).total_seconds() // 60)
        return minute_bars(str(start), max(minutes, 1)).reset_index()


class TruncatedSource:
    """要求区間の一部（先頭30分）しか返さない、実際のデータ欠落を模したダミー。"""

    def fetch(self, symbol, timeframe, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        short_end = min(start + pd.Timedelta(minutes=30), end)
        minutes = int((short_end - start).total_seconds() // 60)
        return minute_bars(str(start), max(minutes, 0)).reset_index()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        symbol="USDJPY",
        data_start=ts("2026-01-05"),
        data_root=tmp_path,
        meta_db=tmp_path / "meta.db",
        periods={
            "training": Period("training", ts("2026-01-01"), ts("2026-12-31"), False)
        },
        models={},
    )


def _quality_rows(meta: Meta, symbol: str, timeframe: Timeframe) -> list[dict]:
    """quality_reports の全行を挿入順で読む（Meta には latest_quality しか無いため、
    隔離レコードのように「最新ではない行」を確かめたいテストはここから直接読む）。
    """
    conn = sqlite3.connect(meta.db_path)
    try:
        rows = conn.execute(
            "SELECT payload FROM quality_reports WHERE symbol = ? AND timeframe = ?"
            " ORDER BY rowid",
            (symbol, timeframe.value),
        ).fetchall()
    finally:
        conn.close()
    return [json.loads(r[0]) for r in rows]


# ============================================================================
# fetch(): 基本的な往復（ブリーフの草案が意図していた挙動）
# ============================================================================


def test_fetch_saves_bars_and_records_range(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    fetch_data.fetch(settings, source, lake, meta, start=ts("2026-01-05"), end=ts("2026-01-05 02:00"))
    got = lake.load("USDJPY", Timeframe.M1, as_of=ts("2030-01-01"))
    assert not got.empty
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [(ts("2026-01-05"), ts("2026-01-05 02:00"))]


def test_fetch_records_quality_report(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    fetch_data.fetch(settings, source, lake, meta, start=ts("2026-01-05"), end=ts("2026-01-05 02:00"))
    report = meta.latest_quality("USDJPY", Timeframe.M1)
    assert report is not None
    assert report["actual_bars"] == 120


def test_fetch_is_idempotent(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    window = dict(start=ts("2026-01-05"), end=ts("2026-01-05 02:00"))
    fetch_data.fetch(settings, source, lake, meta, **window)
    first = len(lake.load("USDJPY", Timeframe.M1, as_of=ts("2030-01-01")))
    fetch_data.fetch(settings, source, lake, meta, **window)
    second = len(lake.load("USDJPY", Timeframe.M1, as_of=ts("2030-01-01")))
    assert first == second


# ============================================================================
# A: 壊れたチャンクは隔離して、残りの取得を続ける
# ============================================================================


def test_fetch_quarantines_a_broken_chunk_and_continues(settings):
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    day1, day2, day3, day4 = (ts(f"2026-01-{d:02d}") for d in (5, 6, 7, 8))
    source = BrokenChunkSource(broken_start=day2)

    # 例外を出さずに最後まで走ることそのものが、このテストの主張。
    fetch_data.fetch(settings, source, lake, meta, start=day1, end=day4, chunk_days=1)

    stored = lake.load("USDJPY", Timeframe.M1, as_of=ts("2030-01-01"))
    # 壊れていない前後(day1〜day2, day3〜day4)は保存されている。
    assert ((stored.index >= day1) & (stored.index < day2)).sum() == 24 * 60
    assert ((stored.index >= day3) & (stored.index < day4)).sum() == 24 * 60
    # 壊れたチャンク(day2〜day3)の中身はレイクに一切入っていない
    # (=検証をすり抜けさせて部分的に保存した、ではないことの確認)。
    assert not ((stored.index >= day2) & (stored.index < day3)).any()

    # day2〜day3 が抜けているので、取得済み区間は連続した1本にはならず2本に分かれる。
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [(day1, day2), (day3, day4)]

    quarantined = [r for r in _quality_rows(meta, "USDJPY", Timeframe.M1) if r.get("status") == "quarantined"]
    assert len(quarantined) == 1
    assert quarantined[0]["chunk_start"] == str(day2)
    assert quarantined[0]["chunk_end"] == str(day3)
    assert "Ask" in quarantined[0]["error"]

    # 最終の品質サマリにも、隔離された分がちゃんと欠損として出ている
    # (「隔離したことにして中身は無かったことにする」ではなく、実際に無いことが
    # 品質レポート経由でも見える)。
    summary = meta.latest_quality("USDJPY", Timeframe.M1)
    assert "status" not in summary  # QualityReport.to_dict() であって隔離スタブではない
    assert any(g["missing_bars"] == 24 * 60 for g in summary["gaps"])


def test_fetch_does_not_swallow_errors_from_the_source_itself(settings):
    """`source.fetch()` 自体が上げる例外(ネットワーク断などを想定)は隔離の対象外。

    データの中身が壊れていることと、そもそも取得できないことは別の障害であり、
    後者まで握りつぶして続行すると、直っていない接続不良の下で永久に空振りし続ける。
    """

    class ExplodingSource:
        def fetch(self, symbol, timeframe, start, end):
            raise ConnectionError("模擬的な接続断")

    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    with pytest.raises(ConnectionError):
        fetch_data.fetch(
            settings, ExplodingSource(), lake, meta,
            start=ts("2026-01-05"), end=ts("2026-01-06"),
        )
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == []


# ============================================================================
# B: 途中から再開できる（meta.fetched_ranges を実際に読んでいる）
# ============================================================================


def test_fetch_resume_skips_a_fully_covered_range(settings):
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    window = dict(start=ts("2026-01-05"), end=ts("2026-01-10"), chunk_days=1)

    first_source = FakeSource()
    fetch_data.fetch(settings, first_source, lake, meta, **window)
    assert len(first_source.calls) == 5  # 5日ぶん、1日ずつ

    second_source = FakeSource()
    fetch_data.fetch(settings, second_source, lake, meta, **window)
    assert second_source.calls == []  # 全区間が既に取得済みなので、一度も呼ばれない


def test_fetch_resume_only_fetches_the_remaining_tail(settings):
    """クラッシュ後の再実行を模す: 前半だけ取得できた状態から、同じ(広い)要求区間で
    もう一度呼んでも、続きの区間だけを取得する(=最初からやり直しにならない)。
    """
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    start, mid, end = ts("2026-01-05"), ts("2026-01-07"), ts("2026-01-10")

    fetch_data.fetch(settings, FakeSource(), lake, meta, start=start, end=mid, chunk_days=1)

    resumed = FakeSource()
    fetch_data.fetch(settings, resumed, lake, meta, start=start, end=end, chunk_days=1)
    requested_starts = [call[2] for call in resumed.calls]
    assert requested_starts, "再開後に何も取得していない"
    assert min(requested_starts) == mid  # start〜mid を再取得していない
    assert max(call[3] for call in resumed.calls) == end


def test_fetch_resume_retries_a_previously_quarantined_gap(settings):
    """隔離された区間は「取得済み」に数えないので、次回の呼び出しでまた対象になる。

    ソース側が直っていれば今度は成功する――壊れたままなら、また隔離されるだけで
    残りを止めない(retryが無限ループや例外にならないことも合わせて確認する)。
    """
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    day1, day2, day3 = ts("2026-01-05"), ts("2026-01-06"), ts("2026-01-07")

    fetch_data.fetch(
        settings, BrokenChunkSource(broken_start=day1), lake, meta,
        start=day1, end=day3, chunk_days=1,
    )
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [(day2, day3)]

    # 2回目は直った(壊れていない)ソースで呼ぶ。day1〜day2 がまだ対象に入ること。
    retry_source = FakeSource()
    fetch_data.fetch(settings, retry_source, lake, meta, start=day1, end=day3, chunk_days=1)
    assert (day1, day2) in [(c[2], c[3]) for c in retry_source.calls]
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [(day1, day3)]


# ============================================================================
# C: 0本だったチャンク(週末・祝日)も取得済みとして記録する
# ============================================================================


def test_fetch_records_empty_chunks_so_they_are_not_retried(settings):
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    empty_day = ts("2026-01-10")
    source = SometimesEmptySource(empty_start=empty_day)
    fetch_data.fetch(settings, source, lake, meta, start=ts("2026-01-09"), end=ts("2026-01-11"), chunk_days=1)

    # 3日分が1本に連結されている: 0本のチャンクも record_fetch されて連続扱いに
    # なっている証拠(されていなければ 01-09〜01-10 と 01-10〜01-11 の間に切れ目が
    # でき、fetched_ranges は2本に分かれる)。
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [(ts("2026-01-09"), ts("2026-01-11"))]

    second_source = SometimesEmptySource(empty_start=empty_day)
    fetch_data.fetch(settings, second_source, lake, meta, start=ts("2026-01-09"), end=ts("2026-01-11"), chunk_days=1)
    assert second_source.calls == []  # 空だったチャンクへ再度問い合わせない


# ============================================================================
# D: 品質レポートに要求範囲(expected_start/expected_end)を渡している
# ============================================================================


def test_fetch_quality_report_reflects_the_full_requested_range(settings):
    """expected_start/expected_end を渡していなければ、取得が途中で切れても
    観測データ自身の端が母数になり「問題なし」に見える(quality.py 参照)。
    fetch() が要求範囲を渡していることを、この非対称(expected>actual)で確認する。
    """
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    start, end = ts("2026-01-05 00:00"), ts("2026-01-05 04:00")  # 240分要求

    fetch_data.fetch(settings, TruncatedSource(), lake, meta, start=start, end=end, chunk_days=30)

    report = meta.latest_quality("USDJPY", Timeframe.M1)
    assert report["expected_bars"] == 240
    assert report["actual_bars"] == 30
    assert len(report["gaps"]) == 1


def test_fetch_quality_load_is_scoped_to_the_requested_window(settings):
    """要求区間より前に別途取得済みのデータが、今回の actual_bars に混ざらないこと。

    混ざると actual_bars だけが expected_bars と無関係に膨らみ、
    「要求範囲に対してどれだけ揃っているか」を表さなくなる。
    """
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    # 要求区間より前のデータを先に用意しておく。
    fetch_data.fetch(settings, FakeSource(), lake, meta, start=ts("2025-01-01"), end=ts("2025-01-02"))

    start, end = ts("2026-01-05 00:00"), ts("2026-01-05 02:00")  # 120分
    fetch_data.fetch(settings, FakeSource(), lake, meta, start=start, end=end)

    report = meta.latest_quality("USDJPY", Timeframe.M1)
    assert report["expected_bars"] == 120
    assert report["actual_bars"] == 120  # 2025年分は含まれない


# ============================================================================
# E: 見出しの品質サマリが祝日規模の欠損で飽和しない
# ============================================================================


def _report_with_gaps(gaps: list[dict], actual: int = 100, expected: int = 100) -> QualityReport:
    return QualityReport(
        symbol="USDJPY",
        timeframe="1m",
        expected_bars=expected,
        actual_bars=actual,
        duplicate_count=0,
        conflicting_duplicate_count=0,
        bad_spread_count=0,
        wide_spread_count=0,
        wide_spread_threshold=0.0,
        price_jump_count=0,
        longest_gap_minutes=max((g["minutes"] for g in gaps), default=0.0),
        gaps=gaps,
    )


def test_quality_summary_does_not_headline_the_saturated_longest_gap():
    """longest_gap_minutes 自体は祝日規模(1440分)で頭打ちになりうる値。
    それをそのまま見出しに使わず、祝日規模を除いた最長を報告することを確認する。
    """
    gaps = [
        {"from": "a", "to": "b", "missing_bars": 1440, "minutes": 1440.0},  # 祝日規模
        {"from": "c", "to": "d", "missing_bars": 45, "minutes": 45.0},  # 本来注目すべき方
    ]
    summary = fetch_data._format_quality_summary(_report_with_gaps(gaps, actual=1000, expected=1000 + 1485))
    assert "最長 45" in summary
    # 素の longest_gap_minutes(1440)をそのまま「最長」として出す実装ならここが壊れる。
    assert "最長 1440" not in summary


def test_quality_summary_reports_holiday_scale_gap_count_separately():
    gaps = [{"from": "", "to": "", "missing_bars": 1440, "minutes": 1440.0} for _ in range(3)]
    summary = fetch_data._format_quality_summary(_report_with_gaps(gaps, actual=100, expected=100 + 3 * 1440))
    assert "祝日規模" in summary
    assert "3 件" in summary
    assert "それ以外 0 件" in summary


def test_quality_summary_all_gaps_small_reports_zero_holiday_scale():
    gaps = [{"from": "", "to": "", "missing_bars": 5, "minutes": 5.0}]
    summary = fetch_data._format_quality_summary(_report_with_gaps(gaps, actual=200, expected=205))
    assert "祝日規模[1440分以上] 0 件" in summary
    assert "最長 5" in summary


def test_quality_summary_handles_no_gaps():
    summary = fetch_data._format_quality_summary(_report_with_gaps([], actual=100, expected=100))
    assert "欠損 0 箇所" in summary
    assert "100/100" in summary
    assert "100.0%" in summary


# ============================================================================
# F: タイムゾーンの扱い
# ============================================================================


def test_fetch_rejects_naive_start(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    with pytest.raises(ValueError, match="tz-aware"):
        fetch_data.fetch(settings, source, lake, meta, start=pd.Timestamp("2026-01-05"))


def test_fetch_rejects_naive_end(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    with pytest.raises(ValueError, match="tz-aware"):
        fetch_data.fetch(
            settings, source, lake, meta,
            start=ts("2026-01-05"), end=pd.Timestamp("2026-01-06"),
        )


def test_fetch_rejects_non_positive_chunk_days(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    with pytest.raises(ValueError, match="chunk_days"):
        fetch_data.fetch(
            settings, source, lake, meta,
            start=ts("2026-01-05"), end=ts("2026-01-06"), chunk_days=0,
        )


def test_parse_cli_timestamp_localizes_bare_date_as_utc():
    assert fetch_data._parse_cli_timestamp("2026-01-05") == ts("2026-01-05")


def test_parse_cli_timestamp_converts_offset_aware_string_to_utc():
    got = fetch_data._parse_cli_timestamp("2026-01-05T00:00:00+09:00")
    assert got == ts("2026-01-04 15:00")
    assert got.tzinfo is not None


def test_build_rejects_naive_as_of(settings):
    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 10).reset_index())
    with pytest.raises(ValueError, match="tz-aware"):
        build_bars.build(settings, lake, as_of=pd.Timestamp("2026-01-05"))


# ============================================================================
# G: sys.path 操作は無い(退行防止)
# ============================================================================


def test_scripts_do_not_manipulate_sys_path():
    """aitrading の import に sys.path 操作は不要(uv sync の editable install で
    解決できることを実機で確認済み)。退行を防ぐための構造的なテスト。

    説明のコメント自体が「sys.path」という語を含みうるので、単純な部分文字列一致では
    誤検出する。実際に操作している呼び出し(insert/append)だけを見る。
    """
    for module in (fetch_data, build_bars):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "sys.path.insert" not in source, f"{module.__name__} に sys.path.insert が残っている"
        assert "sys.path.append" not in source, f"{module.__name__} に sys.path.append が残っている"


# ============================================================================
# H: build_bars の as_of
# ============================================================================


def test_build_bars_generates_all_timeframes(settings):
    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 60 * 48).reset_index())
    build_bars.build(settings, lake)
    as_of = ts("2030-01-01")
    for tf in (Timeframe.M5, Timeframe.H1, Timeframe.D1_NY, Timeframe.D1_JST):
        assert not lake.load("USDJPY", tf, as_of=as_of).empty, f"{tf} が生成されていない"


def test_build_as_of_excludes_bars_not_yet_closed_at_that_point(settings):
    """as_of が実際に build() の中で使われ、その時点でまだ確定していない期間を
    除外することを確認する。渡した値を無視して常に now() を使う実装だとここが壊れる
    (今日の日付は2026-09-01なので、now() ならこの2日分は全部確定して見えてしまう)。
    """
    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 60 * 48).reset_index())
    early_as_of = ts("2026-01-05 03:00")  # 2日分のうち先頭3時間ぶんだけ確定させる

    build_bars.build(settings, lake, timeframes=[Timeframe.H1], as_of=early_as_of)

    stored = lake.load("USDJPY", Timeframe.H1, as_of=ts("2030-01-01"))
    assert len(stored) == 3  # 00-01, 01-02, 02-03 の3本だけ
    assert stored["close_time"].max() <= early_as_of


def test_build_raises_a_clear_error_when_no_m1_data(settings):
    lake = Lake(settings.data_root)
    with pytest.raises(ValueError, match="1分足"):
        build_bars.build(settings, lake)


# ============================================================================
# I: 生成足の再現性テスト ―― 何を保証しているのかを明示し、独立した検算も足す
# ============================================================================


def test_generated_5m_round_trips_through_the_lake_without_drift(settings):
    """build() が内部で resample() を呼ぶ以上、「両者とも resample() を呼んでいる」
    という比較は resample() 自身の正しさまでは保証しない(それは tests/test_bars.py の
    責務で、実際に手計算した値やDST境界などで別途厳密に検証されている)。

    このテストが実際に保証しているのは、build() 固有の配線――
      1. Lake.load(M1) で読み戻した1分足(パスした元データそのものではなく、
         一度 Parquet に save/load を経由したもの)を resample に渡していること
      2. 生成した派生足を .reset_index() して Lake.save に渡し、それを再度
         Lake.load で読み戻しても値が変わらないこと(精度落ち・型変化・並び替えが
         無いこと)
      3. ループの中で正しい timeframe を resample に渡していること
    ――が壊れていないこと。`expected` 側は元データに一切 Lake を経由させずに
    直接 resample() を呼んで作るので、両者の差分は「Lake を2回(1分足の往復+
    派生足の往復)経由したこと」だけに起因する。

    index を落とさずに突き合わせるのは、行数だけ・1列だけの一致では「正しい行と
    正しい行が対応している」ことまでは確認できないため。固定長(M5)と可変長
    (D1_NY)の両方を見て、resample の2つの内部分岐(_fixed_length/_variable_length)
    のどちらでも build() の配線が同じように機能することを確認する。
    """
    from aitrading.bars import resample

    source_bars = minute_bars("2026-01-05 00:00", 60 * 48)  # 月火の2日分
    for timeframe in (Timeframe.M5, Timeframe.D1_NY):
        lake = Lake(settings.data_root / timeframe.value)  # 時間軸ごとに独立したレイク
        lake.save("USDJPY", Timeframe.M1, source_bars.reset_index())
        build_bars.build(settings, lake, timeframes=[timeframe])

        stored = lake.load("USDJPY", timeframe, as_of=ts("2030-01-01"))
        expected = resample(source_bars, timeframe)

        assert not stored.empty, f"{timeframe}: フィクスチャが小さすぎて確定足が無い"
        assert len(stored) == len(expected)
        # check_freq=False: resample() の出力 index は「等間隔である」という pandas の
        # メタデータキャッシュ(freq)を持つが、これは実データではなく、Parquet往復
        # (Lake.save/load)では保持されない。実機で確認済み: 値は完全一致するのに
        # freq だけ None 対 <5 * Minutes> で assert_frame_equal が落ちる。このテストが
        # 検出したいのは値のずれであって、この属性の有無ではない。
        pd.testing.assert_frame_equal(stored, expected, check_like=False, check_freq=False)


def test_generated_5m_matches_hand_computed_ohlc_independent_of_resample(settings):
    """resample() を一切呼ばずに求めた既知の値と突き合わせる、resample()の
    正しさに依存しない独立した検算。

    値は tests/test_bars.py::test_m5_aggregates_ohlc_correctly と同じ
    minute_bars(10本)から、resample()を経由せず手計算したもの
    (1本目=1分目〜5分目: open=1分目のbid_open=150.00, close=5分目のbid_close=150.24,
    high=5本のbid_highの最大=150.54, low=5本のbid_lowの最小=149.50, volume=10*5=50)。
    """
    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 10).reset_index())
    build_bars.build(settings, lake, timeframes=[Timeframe.M5])

    stored = lake.load("USDJPY", Timeframe.M5, as_of=ts("2030-01-01"))
    assert len(stored) == 2
    first = stored.iloc[0]
    assert first["bid_open"] == pytest.approx(150.00)
    assert first["bid_close"] == pytest.approx(150.24)
    assert first["bid_high"] == pytest.approx(150.54)
    assert first["bid_low"] == pytest.approx(149.50)
    assert first["volume"] == pytest.approx(50.0)


# ============================================================================
# _missing_ranges / _iter_chunks: 純粋な区間演算の単体テスト
# ============================================================================


def test_missing_ranges_with_no_coverage_is_the_whole_span():
    assert fetch_data._missing_ranges([], ts("2026-01-01"), ts("2026-01-10")) == [
        (ts("2026-01-01"), ts("2026-01-10"))
    ]


def test_missing_ranges_with_full_coverage_is_empty():
    covered = [(ts("2026-01-01"), ts("2026-01-10"))]
    assert fetch_data._missing_ranges(covered, ts("2026-01-01"), ts("2026-01-10")) == []


def test_missing_ranges_finds_a_gap_in_the_middle():
    covered = [(ts("2026-01-01"), ts("2026-01-03")), (ts("2026-01-05"), ts("2026-01-10"))]
    assert fetch_data._missing_ranges(covered, ts("2026-01-01"), ts("2026-01-10")) == [
        (ts("2026-01-03"), ts("2026-01-05"))
    ]


def test_missing_ranges_finds_only_the_tail_after_a_partial_prefix():
    covered = [(ts("2026-01-01"), ts("2026-01-05"))]
    assert fetch_data._missing_ranges(covered, ts("2026-01-01"), ts("2026-01-10")) == [
        (ts("2026-01-05"), ts("2026-01-10"))
    ]


def test_missing_ranges_ignores_coverage_outside_the_requested_span():
    covered = [(ts("2025-01-01"), ts("2025-06-01")), (ts("2027-01-01"), ts("2027-06-01"))]
    assert fetch_data._missing_ranges(covered, ts("2026-01-01"), ts("2026-01-10")) == [
        (ts("2026-01-01"), ts("2026-01-10"))
    ]


def test_missing_ranges_clips_partial_overlap_at_both_ends():
    covered = [(ts("2025-12-01"), ts("2026-01-15"))]  # 要求区間をはみ出して覆っている
    assert fetch_data._missing_ranges(covered, ts("2026-01-01"), ts("2026-01-10")) == []


def test_missing_ranges_handles_several_disjoint_gaps():
    covered = [
        (ts("2026-01-02"), ts("2026-01-03")),
        (ts("2026-01-04"), ts("2026-01-05")),
        (ts("2026-01-07"), ts("2026-01-08")),
    ]
    assert fetch_data._missing_ranges(covered, ts("2026-01-01"), ts("2026-01-10")) == [
        (ts("2026-01-01"), ts("2026-01-02")),
        (ts("2026-01-03"), ts("2026-01-04")),
        (ts("2026-01-05"), ts("2026-01-07")),
        (ts("2026-01-08"), ts("2026-01-10")),
    ]


def test_iter_chunks_splits_by_step():
    chunks = list(fetch_data._iter_chunks(ts("2026-01-01"), ts("2026-01-10"), pd.Timedelta(days=3)))
    assert chunks == [
        (ts("2026-01-01"), ts("2026-01-04")),
        (ts("2026-01-04"), ts("2026-01-07")),
        (ts("2026-01-07"), ts("2026-01-10")),
    ]


def test_iter_chunks_last_chunk_is_clipped_to_end():
    chunks = list(fetch_data._iter_chunks(ts("2026-01-01"), ts("2026-01-08"), pd.Timedelta(days=3)))
    assert chunks[-1] == (ts("2026-01-07"), ts("2026-01-08"))


def test_iter_chunks_empty_span_yields_nothing():
    assert list(fetch_data._iter_chunks(ts("2026-01-01"), ts("2026-01-01"), pd.Timedelta(days=3))) == []


# ============================================================================
# main(): CLI引数の配線(実ネットワーク・実 config/settings.toml には触れない)
# ============================================================================


def test_fetch_main_wires_cli_args_into_fetch(tmp_path, monkeypatch):
    calls = {}

    def fake_fetch(settings, source, lake, meta, *, start=None, end=None, chunk_days=30):
        calls.update(settings=settings, source=source, start=start, end=end, chunk_days=chunk_days)
        # 本物と同じ契約（FetchOutcome を返す）を守る。ダブルが本物より緩いと、
        # そのダブルでしか起きない経路だけを検証することになる（C1 の教訓）。
        return fetch_data.FetchOutcome(saved_chunks=1, quarantined_chunks=0)

    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(fetch_data, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(fetch_data, "fetch", fake_fetch)
    monkeypatch.setattr(fetch_data, "DukascopySource", lambda: "FAKE_SOURCE")
    monkeypatch.setattr(fetch_data, "Lake", lambda root: f"FAKE_LAKE:{root}")
    monkeypatch.setattr(fetch_data, "Meta", lambda db: f"FAKE_META:{db}")

    code = fetch_data.main(["--start", "2026-01-05", "--end", "2026-01-06", "--chunk-days", "7"])

    assert code == 0
    assert calls["start"] == ts("2026-01-05")
    assert calls["end"] == ts("2026-01-06")
    assert calls["chunk_days"] == 7
    assert calls["source"] == "FAKE_SOURCE"


def test_fetch_main_defaults_start_and_end_to_none(tmp_path, monkeypatch):
    """--start/--end を省略したら fetch() 自身の既定値(settings.data_start/現在時刻)
    に委ねる。main() 側で勝手な既定値を作らないことの確認。"""
    calls = {}

    def fake_fetch(settings, source, lake, meta, *, start=None, end=None, chunk_days=30):
        calls.update(start=start, end=end)
        # 本物と同じ契約（FetchOutcome を返す）を守る。ダブルが本物より緩いと、
        # そのダブルでしか起きない経路だけを検証することになる（C1 の教訓）。
        return fetch_data.FetchOutcome(saved_chunks=1, quarantined_chunks=0)

    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(fetch_data, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(fetch_data, "fetch", fake_fetch)
    monkeypatch.setattr(fetch_data, "DukascopySource", lambda: "FAKE_SOURCE")
    monkeypatch.setattr(fetch_data, "Lake", lambda root: "FAKE_LAKE")
    monkeypatch.setattr(fetch_data, "Meta", lambda db: "FAKE_META")

    code = fetch_data.main([])
    assert code == 0
    assert calls == {"start": None, "end": None}


def test_fetch_main_returns_nonzero_and_prints_message_on_value_error(tmp_path, monkeypatch, capsys):
    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(fetch_data, "load_settings", lambda: fake_settings)

    def raising_fetch(*a, **k):
        raise ValueError("わざとのエラー")

    monkeypatch.setattr(fetch_data, "fetch", raising_fetch)
    monkeypatch.setattr(fetch_data, "DukascopySource", lambda: "FAKE_SOURCE")
    monkeypatch.setattr(fetch_data, "Lake", lambda root: "FAKE_LAKE")
    monkeypatch.setattr(fetch_data, "Meta", lambda db: "FAKE_META")

    code = fetch_data.main([])
    assert code == 1
    assert "わざとのエラー" in capsys.readouterr().out


def test_build_main_wires_timeframe_args_into_build(tmp_path, monkeypatch):
    calls = {}

    def fake_build(settings, lake, *, timeframes=None, as_of=None):
        calls.update(settings=settings, lake=lake, timeframes=timeframes)

    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(build_bars, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(build_bars, "build", fake_build)
    monkeypatch.setattr(build_bars, "Lake", lambda root: f"FAKE_LAKE:{root}")

    code = build_bars.main(["--timeframe", "5m", "--timeframe", "1h"])

    assert code == 0
    assert calls["timeframes"] == [Timeframe.M5, Timeframe.H1]


def test_build_main_returns_nonzero_and_prints_message_on_value_error(tmp_path, monkeypatch, capsys):
    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(build_bars, "load_settings", lambda: fake_settings)

    def raising_build(*a, **k):
        raise ValueError("1分足が無い。先に scripts/fetch_data.py を実行すること")

    monkeypatch.setattr(build_bars, "build", raising_build)
    monkeypatch.setattr(build_bars, "Lake", lambda root: "FAKE_LAKE")

    code = build_bars.main([])
    assert code == 1
    assert "1分足が無い" in capsys.readouterr().out


# ============================================================================
# レビュー(task-11-review.md)で見つかった欠陥の回帰テスト
# ============================================================================


class ValidatingBrokenSource:
    """本物の BarSource と同じく、返す前に validate_bars を通すダミー。

    実装(`DukascopySource`)は `normalize()` の最終行で必ず `validate_bars` を
    呼ぶので、壊れたバーは `lake.save()` ではなく `source.fetch()` の中で
    ValueError になる。この違いが C1（隔離が本番で一度も発火しない）の原因だった。
    """

    def __init__(self, broken_starts: set[pd.Timestamp]):
        self.broken_starts = {pd.Timestamp(s) for s in broken_starts}

    def fetch(self, symbol, timeframe, start, end):
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        minutes = int((end - start).total_seconds() // 60)
        df = minute_bars(str(start), max(minutes, 1)).reset_index()
        if start in self.broken_starts:
            df.loc[0, "ask_close"] = df.loc[0, "bid_close"] - 0.10
        return validate_bars(df, timeframe)


def test_quarantine_works_when_the_source_validates_before_returning(settings):
    """C1: 壊れたバーが `source.fetch()` の中で ValueError になっても隔離する。

    本番の DukascopySource はこの経路を通る。`lake.save()` だけを囲んでいると
    隔離は一度も発火せず、しかも再開ロジックが正しく続きから始める結果、
    毎回同じ壊れたチャンクで死んでそれより後ろが未来永劫取得されない。
    """
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    day1, day2, day3, day4 = (ts(f"2026-01-{d:02d}") for d in (5, 6, 7, 8))
    source = ValidatingBrokenSource({day2})

    outcome = fetch_data.fetch(
        settings, source, lake, meta, start=day1, end=day4, chunk_days=1
    )

    assert outcome.saved_chunks == 2
    assert outcome.quarantined_chunks == 1
    stored = lake.load("USDJPY", Timeframe.M1, as_of=ts("2030-01-01"))
    assert ((stored.index >= day3) & (stored.index < day4)).sum() == 24 * 60, (
        "壊れたチャンクより後ろが取得できていない"
    )
    quarantined = [
        r for r in _quality_rows(meta, "USDJPY", Timeframe.M1)
        if r.get("status") == "quarantined"
    ]
    assert len(quarantined) == 1
    assert quarantined[0]["bar_count"] is None  # 手元にデータが無いので0本とは書かない


def test_fetch_reports_how_many_chunks_were_saved_and_quarantined(settings):
    """C3: 例外が出なくても「1本も取れていない」ことが呼び出し側に伝わる。"""
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)
    days = [ts(f"2026-01-{d:02d}") for d in (5, 6, 7)]
    source = ValidatingBrokenSource(set(days))

    outcome = fetch_data.fetch(
        settings, source, lake, meta, start=days[0], end=ts("2026-01-08"), chunk_days=1
    )
    assert outcome.saved_chunks == 0
    assert outcome.quarantined_chunks == 3
    assert not outcome.ok


def test_main_exits_nonzero_when_nothing_was_fetched(tmp_path, monkeypatch):
    """C3: 全チャンクが隔離されたのに終了コードが0だと、シェルやタスク
    スケジューラが「取得成功」と判断してしまう。"""
    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(fetch_data, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(fetch_data, "DukascopySource", lambda: None)
    monkeypatch.setattr(fetch_data, "Lake", lambda root: None)
    monkeypatch.setattr(fetch_data, "Meta", lambda db: None)
    monkeypatch.setattr(
        fetch_data, "fetch",
        lambda *a, **k: fetch_data.FetchOutcome(saved_chunks=0, quarantined_chunks=3),
    )
    assert fetch_data.main([]) == 1


def test_main_exits_nonzero_when_some_chunks_were_quarantined(tmp_path, monkeypatch):
    """一部だけ隔離された場合も、成功(0)にしてはいけない。"""
    fake_settings = Settings(
        symbol="USDJPY", data_start=ts("2026-01-01"), data_root=tmp_path,
        meta_db=tmp_path / "meta.db", periods={}, models={},
    )
    monkeypatch.setattr(fetch_data, "load_settings", lambda: fake_settings)
    monkeypatch.setattr(fetch_data, "DukascopySource", lambda: None)
    monkeypatch.setattr(fetch_data, "Lake", lambda root: None)
    monkeypatch.setattr(fetch_data, "Meta", lambda db: None)
    monkeypatch.setattr(
        fetch_data, "fetch",
        lambda *a, **k: fetch_data.FetchOutcome(saved_chunks=5, quarantined_chunks=1),
    )
    assert fetch_data.main([]) != 0


def test_build_recovers_after_a_hole_is_filled(settings):
    """C2: 1分足に内側の穴があるまま生成した派生足が、穴を埋めたあと
    作り直せること。

    派生足を既存と結合すると、同じ open_time の値が変わって Lake の値衝突
    検出に当たり、**以後その時間軸は何度実行しても生成できなくなる**
    （parquet を手で消すまで）。派生足は1分足の純粋な関数なので、毎回
    作り直すのが正しい。
    """
    lake = Lake(settings.data_root)
    full = minute_bars("2026-01-05 00:00", 60 * 24 * 4)
    hole = (full.index >= ts("2026-01-06")) & (full.index < ts("2026-01-07"))
    lake.save("USDJPY", Timeframe.M1, full.loc[~hole].reset_index())

    as_of = ts("2030-01-01")
    build_bars.build(settings, lake, timeframes=[Timeframe.D1_NY], as_of=as_of)
    partial = lake.load("USDJPY", Timeframe.D1_NY, as_of=as_of)
    assert not partial.empty

    # 穴が埋まる（隔離されたチャンクを取り直した状況）
    lake.save("USDJPY", Timeframe.M1, full.loc[hole].reset_index())
    build_bars.build(settings, lake, timeframes=[Timeframe.D1_NY], as_of=as_of)

    rebuilt = lake.load("USDJPY", Timeframe.D1_NY, as_of=as_of)
    assert not rebuilt.empty
    # 穴が埋まったぶん、同じ日足の出来高が増えている（作り直されている証拠）
    first_day = rebuilt.index[0]
    assert rebuilt.loc[first_day, "volume"] > partial.loc[first_day, "volume"]


def test_build_regenerates_rather_than_accumulating(settings):
    """派生足は毎回作り直す。1分足を減らしたら派生足も減る。"""
    lake = Lake(settings.data_root)
    full = minute_bars("2026-01-05 00:00", 60 * 24 * 4)
    lake.save("USDJPY", Timeframe.M1, full.reset_index())
    as_of = ts("2030-01-01")
    build_bars.build(settings, lake, timeframes=[Timeframe.H1], as_of=as_of)
    many = len(lake.load("USDJPY", Timeframe.H1, as_of=as_of))

    # 生成の材料を狭めて作り直す（as_of を前倒しする）
    build_bars.build(settings, lake, timeframes=[Timeframe.H1], as_of=ts("2026-01-06"))
    fewer = len(lake.load("USDJPY", Timeframe.H1, as_of=as_of))
    assert fewer < many, "古い生成物が残っている（作り直しではなく積み上げになっている）"


def test_lake_drop_refuses_to_delete_the_fetched_minute_bars(settings):
    """1分足は取得物。消したら再取得（数時間）でしか復元できない。"""
    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 60).reset_index())
    with pytest.raises(ValueError, match="取得物"):
        lake.drop("USDJPY", Timeframe.M1)
    assert lake.available_years("USDJPY", Timeframe.M1) == [2026]
