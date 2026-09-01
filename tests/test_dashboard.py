"""dashboard/app.py のテスト。

`dashboard/` はパッケージではない（`pyproject.toml` の `packages` は
`src/aitrading` のみ）ので、`sys.path` にディレクトリを足してから素のモジュール
として import する（`tests/test_scripts.py` が `scripts/` に対して採っているのと
同じ形）。

`main()` 自体はテストしない。Streamlit のスクリプト実行コンテキストが要るうえ、
判断ロジックは全て純関数に切り出してあるので、そちらを直接検証する。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import app  # noqa: E402

from aitrading.config import Period, Settings  # noqa: E402
from aitrading.quality import QualityReport, format_summary  # noqa: E402
from aitrading.storage.meta import Meta  # noqa: E402
from aitrading.timeutil import Timeframe  # noqa: E402

from tests.helpers import minute_bars  # noqa: E402


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


@pytest.fixture
def settings(tmp_path):
    """training は解放、oos はロック。実際の settings.toml と同じ構成。"""
    return Settings(
        symbol="USDJPY",
        data_start=ts("2026-01-05"),
        data_root=tmp_path,
        meta_db=tmp_path / "meta.db",
        periods={
            "training": Period("training", ts("2026-01-01"), ts("2026-06-30"), False),
            "oos": Period("oos", ts("2026-07-01"), ts("2026-12-31"), True),
        },
        models={},
    )


# ============================================================================
# チャート
# ============================================================================


def test_candlestick_figure_has_ohlc_trace():
    fig = app.candlestick_figure(minute_bars("2026-01-05 00:00", 60))
    assert any(trace.type == "candlestick" for trace in fig.data)


def test_overlays_are_added_as_lines():
    bars = minute_bars("2026-01-05 00:00", 60)
    overlay = bars["bid_close"].rolling(5, min_periods=5).mean().rename("sma")
    fig = app.candlestick_figure(bars, overlays={"sma": overlay})
    assert any(trace.type == "scatter" and trace.name == "sma" for trace in fig.data)


def test_chart_is_capped_so_the_browser_does_not_hang():
    """実データは10年で約370万本。そのまま Plotly に渡すとブラウザが固まる。"""
    bars = minute_bars("2026-01-05 00:00", app.MAX_CHART_BARS + 500)
    limited = app.limit_for_chart(bars)
    assert len(limited) == app.MAX_CHART_BARS
    # 直近側を残す（古い方を捨てる）。逆だとチャートが常に最初の数日で止まる
    assert limited.index[-1] == bars.index[-1]


def test_chart_limit_leaves_short_series_untouched():
    bars = minute_bars("2026-01-05 00:00", 10)
    pd.testing.assert_frame_equal(app.limit_for_chart(bars), bars)


# ============================================================================
# 品質タブ —— 隔離レコードと正規サマリの区別
# ============================================================================


def _summary_payload() -> dict:
    report = QualityReport(
        symbol="USDJPY", timeframe="1m", expected_bars=100, actual_bars=100,
        duplicate_count=0, conflicting_duplicate_count=0, bad_spread_count=0,
        wide_spread_count=0, wide_spread_threshold=0.02, price_jump_count=0,
        longest_gap_minutes=0.0, gaps=[],
    )
    return report.to_dict()


def _quarantine_payload(chunk_start: str) -> dict:
    return {
        "status": "quarantined",
        "chunk_start": chunk_start,
        "chunk_end": chunk_start,
        "bar_count": None,
        "error": "Ask が Bid を close で下回る行が 1 件ある",
    }


def test_quality_view_classifies_a_quarantine_record_without_crashing():
    """`latest_quality()` は隔離レコードを返しうる。

    全チャンクが隔離されると最終サマリが一度も書かれないので、最新レコードが
    隔離スタブになる。`report["actual_bars"]` を無条件に読むと KeyError になる
    （実測済みの不具合）。
    """
    view = app.quality_view(_quarantine_payload("2026-01-06 00:00:00+00:00"))
    assert view["kind"] == "quarantined"
    assert "Ask" in view["record"]["error"]


def test_quality_view_classifies_a_normal_summary():
    view = app.quality_view(_summary_payload())
    assert view["kind"] == "summary"
    assert view["record"].actual_bars == 100


def test_quality_view_handles_no_report_at_all():
    assert app.quality_view(None) == {"kind": "none"}


def test_quarantine_events_stay_visible_behind_a_later_summary():
    """一部だけ隔離された取得では、そのあと正規サマリが記録されるので
    `latest_quality()` からは隔離が見えなくなる。履歴からは見えること。"""
    history = [
        _quarantine_payload("2026-01-06 00:00:00+00:00"),
        _summary_payload(),
    ]
    assert app.quality_view(history[-1])["kind"] == "summary"
    quarantined = app.quarantine_records(history)
    assert len(quarantined) == 1
    assert quarantined[0]["chunk_start"] == "2026-01-06 00:00:00+00:00"


def test_quality_headline_does_not_saturate_on_holidays():
    """祝日クローズ（1440分）が見出しの最長欠損を占拠しないこと。

    `is_market_open` は曜日しか見ないので祝日は毎回1440分の欠損として出る。
    そのまま見出しにすると、本物の30分欠損が起きていても数字が1440のままになる。
    """
    report = QualityReport(
        symbol="USDJPY", timeframe="1m", expected_bars=10000, actual_bars=8530,
        duplicate_count=0, conflicting_duplicate_count=0, bad_spread_count=0,
        wide_spread_count=0, wide_spread_threshold=0.02, price_jump_count=0,
        longest_gap_minutes=1440.0,
        gaps=[
            {"from": "a", "to": "b", "missing_bars": 1440, "minutes": 1440.0},
            {"from": "c", "to": "d", "missing_bars": 30, "minutes": 30.0},
        ],
    )
    summary = format_summary(report)
    # 祝日規模とそれ以外を分けて数え、後者の最長（30分）を見出しにすること。
    # `"30" in summary` のような素朴な部分一致は空振りする（actual_bars=8530 に
    # "30" が含まれるため、閾値を壊しても通ってしまう。実際に変異検査で判明した）。
    assert "祝日規模[1440分以上] 1 件" in summary
    assert "それ以外 1 件・最長 30 分" in summary


# ============================================================================
# 指標からシグナルを作る
# ============================================================================


def test_signal_from_a_series_indicator():
    values = pd.Series([10.0, 20.0, 30.0], name="rsi")
    got = app.signal_from_indicator(values, None, "<", 25.0)
    assert got.tolist() == [True, True, False]


def test_signal_from_a_dataframe_indicator_requires_a_column():
    frame = pd.DataFrame({"macd": [1.0, -1.0], "signal": [0.0, 0.0]})
    assert app.signal_from_indicator(frame, "macd", ">", 0.0).tolist() == [True, False]
    with pytest.raises(ValueError, match="column"):
        app.signal_from_indicator(frame, None, ">", 0.0)


def test_signal_rejects_an_unknown_operator():
    with pytest.raises(ValueError, match="演算子"):
        app.signal_from_indicator(pd.Series([1.0]), None, "==", 1.0)


def test_indicator_result_columns_covers_both_shapes():
    assert app.indicator_result_columns(pd.Series([1.0], name="rsi")) == ["rsi"]
    assert app.indicator_result_columns(
        pd.DataFrame({"lower": [1.0], "upper": [2.0]})
    ) == ["lower", "upper"]


# ============================================================================
# 期待値スキャン —— ロック期間の扱い（このダッシュボードで最も重要）
# ============================================================================


def _scan_bars(n: int = 400) -> pd.DataFrame:
    """指標がウォームアップしきる長さのランダムウォーク。"""
    rng = np.random.default_rng(3)
    index = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    mid = 150.0 + np.cumsum(rng.normal(0, 0.01, n))
    body = {"close_time": index + pd.Timedelta(minutes=1), "volume": np.full(n, 100.0)}
    for field in ("open", "high", "low", "close"):
        body[f"bid_{field}"] = mid - 0.01
        body[f"ask_{field}"] = mid + 0.01
    return pd.DataFrame(body, index=index).rename_axis("open_time")


def test_scan_on_an_unlocked_period_needs_no_reason(settings):
    meta = Meta(settings.meta_db)
    result = app.scan_indicator_condition(
        _scan_bars(), settings, meta,
        indicator_name="rsi", column=None, op="<", threshold=70.0,
        period="training", horizons=(5,),
    )
    assert result.n_signals >= 0
    assert meta.oos_unlocks() == []


def test_scan_refuses_a_locked_period_without_a_reason(settings):
    """ダッシュボードは人間がうっかり OOS を見てしまう最有力の場所。

    「Training のみ」と画面に書くだけでは何も強制されない。ロックされた期間は
    `scan_period` 自身の拒否にそのまま乗せること。
    """
    meta = Meta(settings.meta_db)
    with pytest.raises(PermissionError, match="oos"):
        app.scan_indicator_condition(
            _scan_bars(), settings, meta,
            indicator_name="rsi", column=None, op="<", threshold=70.0,
            period="oos", horizons=(5,),
        )
    assert meta.oos_unlocks() == []


def test_scan_records_the_unlock_when_a_reason_is_given(settings):
    meta = Meta(settings.meta_db)
    app.scan_indicator_condition(
        _scan_bars(), settings, meta,
        indicator_name="rsi", column=None, op="<", threshold=70.0,
        period="oos", unlock_reason="戦略v1の最終確認", horizons=(5,),
    )
    unlocks = meta.oos_unlocks()
    assert len(unlocks) == 1
    assert unlocks[0]["reason"] == "戦略v1の最終確認"


def test_scan_refuses_a_meta_pointing_somewhere_else(settings, tmp_path):
    """使い捨てDBを渡せば監査を迂回できる、という抜け道が塞がっていること。"""
    throwaway = Meta(tmp_path / "throwaway.db")
    with pytest.raises(ValueError, match="宛先"):
        app.scan_indicator_condition(
            _scan_bars(), settings, throwaway,
            indicator_name="rsi", column=None, op="<", threshold=70.0,
            period="oos", unlock_reason="こっそり見る", horizons=(5,),
        )


def test_scan_rejects_an_unknown_indicator(settings):
    meta = Meta(settings.meta_db)
    with pytest.raises(ValueError, match="未知の指標"):
        app.scan_indicator_condition(
            _scan_bars(), settings, meta,
            indicator_name="存在しない", column=None, op="<", threshold=1.0,
            period="training",
        )


def test_edge_result_table_has_one_row_per_horizon(settings):
    meta = Meta(settings.meta_db)
    result = app.scan_indicator_condition(
        _scan_bars(), settings, meta,
        indicator_name="rsi", column=None, op="<", threshold=70.0,
        period="training", horizons=(1, 5, 15),
    )
    table = app.edge_result_table(result)
    assert len(table) == 3
    assert table["horizon"].tolist() == [1, 5, 15]
    # 重複補正した実効サンプル数が表に出ていること（n だけ見ると過信する）
    assert "n_eff" in table.columns


# ============================================================================
# データ読み込み
# ============================================================================


def test_load_bars_honours_the_as_of_cursor(settings, tmp_path):
    """as_of より後のバーは物理的に返らない（先読み防止 第4層）。"""
    from aitrading.storage.lake import Lake

    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 120).reset_index())
    got = app._load_bars(lake, "USDJPY", Timeframe.M1, ts("2026-01-05 00:30"))
    assert got.index.max() < ts("2026-01-05 00:30")


def test_settings_periods_drive_the_period_choices(settings):
    """期間は settings から取る。画面に固定文字列で並べない
    （settings.toml を変えたのに画面が古い期間を出す、という食い違いを避ける）。"""
    assert set(settings.periods) == {"training", "oos"}
    locked = {name for name, p in settings.periods.items() if p.locked}
    assert locked == {"oos"}
