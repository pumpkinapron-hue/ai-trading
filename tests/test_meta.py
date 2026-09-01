import sqlite3

import pandas as pd
import pytest

from aitrading.storage import meta as meta_module
from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe


@pytest.fixture
def meta(tmp_path):
    return Meta(tmp_path / "meta.db")


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_records_and_reads_fetch_range(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [
        (ts("2026-01-05"), ts("2026-01-06"))
    ]


def test_merges_adjacent_ranges(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-06"), ts("2026-01-07"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [
        (ts("2026-01-05"), ts("2026-01-07"))
    ]


def test_keeps_disjoint_ranges_separate(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-02-01"), ts("2026-02-02"))
    assert len(meta.fetched_ranges("USDJPY", Timeframe.M1)) == 2


def test_ranges_are_scoped_per_timeframe(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M5) == []


def test_latest_quality_returns_most_recent(meta):
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 100})
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 200})
    assert meta.latest_quality("USDJPY", Timeframe.M1)["actual_bars"] == 200


def test_latest_quality_none_when_absent(meta):
    assert meta.latest_quality("USDJPY", Timeframe.M1) is None


def test_oos_unlock_is_recorded(meta):
    meta.record_oos_unlock("oos", "戦略v1.2の最終確認")
    unlocks = meta.oos_unlocks()
    assert len(unlocks) == 1
    assert unlocks[0]["period"] == "oos"
    assert unlocks[0]["reason"] == "戦略v1.2の最終確認"
    assert unlocks[0]["at"] is not None


def test_reopening_same_db_keeps_data(tmp_path):
    path = tmp_path / "meta.db"
    Meta(path).record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert len(Meta(path).fetched_ranges("USDJPY", Timeframe.M1)) == 1


# --- 以下はブリーフの8テストではカバーされない挙動を確認する追加テスト ---


def test_record_fetch_rejects_naive_start(meta):
    """naive な start を素通りさせると、既存のtz-aware行と比較するときに
    fetched_ranges が `TypeError: Cannot compare tz-naive and tz-aware
    timestamps` で落ちる（マージのループが両方tz-awareだと仮定しているため）。
    「naive入力はValueError」という全体の規約はここにも適用する。
    """
    with pytest.raises(ValueError, match="tz-aware"):
        meta.record_fetch(
            "USDJPY", Timeframe.M1, pd.Timestamp("2026-01-05"), ts("2026-01-06")
        )


def test_record_fetch_rejects_naive_end(meta):
    with pytest.raises(ValueError, match="tz-aware"):
        meta.record_fetch(
            "USDJPY", Timeframe.M1, ts("2026-01-05"), pd.Timestamp("2026-01-06")
        )


def test_record_fetch_normalizes_non_utc_offset_before_sorting(meta):
    """start/endはUTCに正規化してから保存する。しないと、SQLite側の
    `ORDER BY start_at` はテキストの辞書順であって時刻の大小ではないため、
    オフセットの違う文字列が混ざると並び順が時系列と一致しなくなる。

    例えば "2026-01-05 00:00:00+09:00" と "2026-01-05 00:00:00+00:00" は、
    実際には前者が9時間早い時刻を指すが、辞書順では '9' > '0' なので後者が
    先に来てしまう。すると本来「先」に処理されるべき区間がマージのループで
    後回しになり、マージ結果の開始時刻が実際より遅く（＝取得済み範囲が
    実際より狭く）算出される——過去に取得済みの分を「未取得」と誤認しうる
    サイレントな不整合になる。

    ここでは JST 00:00〜09:00（= UTC 前日15:00〜当日00:00）と、それに隣接する
    UTC 00:00〜01:00 を渡す。正しく正規化されていれば1本の区間にマージされる。
    """
    jst_start = pd.Timestamp("2026-01-05 00:00", tz="Asia/Tokyo")  # = 2026-01-04 15:00 UTC
    jst_end = pd.Timestamp("2026-01-05 09:00", tz="Asia/Tokyo")  # = 2026-01-05 00:00 UTC
    meta.record_fetch("USDJPY", Timeframe.M1, jst_start, jst_end)
    meta.record_fetch(
        "USDJPY", Timeframe.M1, ts("2026-01-05 00:00"), ts("2026-01-05 01:00")
    )

    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [
        (ts("2026-01-04 15:00"), ts("2026-01-05 01:00"))
    ]


def test_merges_ranges_inserted_out_of_chronological_order(meta):
    """record_fetch はどんな順序で呼ばれてもよい。fetched_ranges 側が
    `ORDER BY start_at` で並べ直してからマージするので、後から呼ばれた方が
    時系列で先でも正しくマージされることを確認する。
    """
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-06"), ts("2026-01-07"))
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [
        (ts("2026-01-05"), ts("2026-01-07"))
    ]


def test_latest_quality_picks_last_insert_when_clock_ties(meta, monkeypatch):
    """created_at はウォールクロックの文字列なので、短時間に連続で呼ばれると
    同じ値になりうる（クロック分解能次第。Windowsに限った話ではないが、
    このプロジェクトの実行環境はWindows）。それでも「後から記録された方」が
    最新として返るべきで、latest_quality は rowid（このテーブルがINSERT専用
    である限り挿入順と一致する）で判定していることを確認する。

    `_now` を固定値にモンキーパッチして同着を確実に再現する
    ——実クロックの分解能に依存させると、たまたま値が違って偽陽性で
    通ってしまうテストになる。
    """
    monkeypatch.setattr(meta_module, "_now", lambda: "2026-01-01T00:00:00+00:00")
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 100})
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 200})
    assert meta.latest_quality("USDJPY", Timeframe.M1)["actual_bars"] == 200


def test_connection_is_closed_after_each_call(tmp_path, monkeypatch):
    """`with conn:` は sqlite3.Connection をコミット/ロールバックするだけで、
    接続そのものは閉じない（stdlibの挙動）。呼び出しのたびに新しい接続を
    開くこのクラスの作りで閉じ忘れると、呼び出しのたびに接続がリークする。
    Windowsでは開いたファイルハンドルが残ると、後続の操作を妨げうる。

    sqlite3.connect を、close() 呼び出しを記録するサブクラスをfactoryに
    差し込むラッパーに差し替えて確認する。
    """
    closed = []

    class TrackingConnection(sqlite3.Connection):
        def close(self):
            closed.append(True)
            super().close()

    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        kwargs.setdefault("factory", TrackingConnection)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", spy_connect)

    m = Meta(tmp_path / "meta.db")  # コンストラクタも最低1回は接続を使う
    assert closed, "コンストラクタでの接続が閉じられていない"

    closed.clear()
    m.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert closed == [True], "record_fetch のあとに接続が閉じられていない"


def test_record_oos_unlock_rejects_blank_reason(meta):
    """reasonが空文字列（または空白のみ）だと、記録の体をなさない。
    「必須」を型（strを受け取る）だけで担保すると空文字列で素通りしてしまい、
    「理由なしで覗ける」という抜け道になる。OOS解除の監査ログはこの抜け道を
    許してはいけない。
    """
    with pytest.raises(ValueError, match="reason"):
        meta.record_oos_unlock("oos", "")
    with pytest.raises(ValueError, match="reason"):
        meta.record_oos_unlock("oos", "   ")


def test_record_oos_unlock_rejects_blank_period(meta):
    with pytest.raises(ValueError, match="period"):
        meta.record_oos_unlock("", "戦略v1.2の最終確認")


def test_record_oos_unlock_rejected_call_leaves_no_record(meta):
    """拒否された呼び出しが、それでも部分的にレコードを残してしまわないこと
    （バリデーションがSQL実行より前に走っていることの確認）。
    """
    with pytest.raises(ValueError):
        meta.record_oos_unlock("oos", "")
    assert meta.oos_unlocks() == []


# --- quality_history: dashboard/app.py の品質タブが使う（Task 12） ---


def test_quality_history_returns_all_records_in_insertion_order(meta):
    """latest_quality は最新1件しか返さないため、途中の隔離レコードは
    正規の最終サマリが後から記録されると見えなくなる。quality_history は
    その全件を挿入順で返す。
    """
    meta.record_quality("USDJPY", Timeframe.M1, {"status": "quarantined", "chunk": 1})
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 100})
    history = meta.quality_history("USDJPY", Timeframe.M1)
    assert [r.get("status", r.get("actual_bars")) for r in history] == ["quarantined", 100]


def test_quality_history_empty_when_absent(meta):
    assert meta.quality_history("USDJPY", Timeframe.M1) == []


def test_quality_history_is_scoped_per_symbol_and_timeframe(meta):
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 1})
    meta.record_quality("USDJPY", Timeframe.M5, {"actual_bars": 2})
    meta.record_quality("EURUSD", Timeframe.M1, {"actual_bars": 3})
    assert meta.quality_history("USDJPY", Timeframe.M1) == [{"actual_bars": 1}]
