"""SQLite のメタデータ。取得済み区間・品質レポート・OOS解除ログ。

将来ここにトレードログ・戦略バージョン・AI判断ログ（仕様書§15/§16/§19）が乗る。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aitrading.timeutil import Timeframe, ensure_utc_timestamp

SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_ranges (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    start_at   TEXT NOT NULL,
    end_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quality_reports (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS oos_unlocks (
    period     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Meta:
    """取得済み区間・品質レポート・OOS解除ログの永続化。

    呼び出しのたびに接続を開いて閉じる。インスタンスとして接続を持ち回さない
    （SQLiteのファイルハンドルの寿命を、Metaインスタンスより短く、
    1回の呼び出しに閉じ込めておきたいため）。
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """接続を開き、正常終了ならコミット・例外ならロールバックし、
        どちらの場合でも必ず閉じる。

        sqlite3.Connection それ自体を `with conn:` として使うと、コミット/
        ロールバックはするが接続を閉じない（stdlibのドキュメント通りの挙動。
        「閉じる」と誤解しやすい落とし穴）。ここで毎回接続を開いて使い捨てる
        設計だと、閉じ忘れは呼び出しのたびに接続がリークすることを意味する。
        Windowsでは開いたままのファイルハンドルが後続の操作を妨げうる。
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def record_fetch(
        self, symbol: str, timeframe: Timeframe, start: pd.Timestamp, end: pd.Timestamp
    ) -> None:
        start = ensure_utc_timestamp(start, "start")
        end = ensure_utc_timestamp(end, "end")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fetch_ranges (symbol, timeframe, start_at, end_at)"
                " VALUES (?, ?, ?, ?)",
                (symbol, timeframe.value, str(start), str(end)),
            )

    def fetched_ranges(
        self, symbol: str, timeframe: Timeframe
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """取得済み区間。隣接・重複はマージして返す。

        マージしておくと「どこがまだ無いか」の差分計算が素直に書ける。

        DB側で `ORDER BY start_at` してから読むので、record_fetch がどんな
        順序で呼ばれてもマージは正しく動く。これは保存する文字列がすべて
        UTCに揃っている前提の上に成り立つ（_ensure_utc_timestamp参照）——
        揃っていないとテキストの辞書順ソートが時系列と一致しなくなる。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT start_at, end_at FROM fetch_ranges"
                " WHERE symbol = ? AND timeframe = ? ORDER BY start_at",
                (symbol, timeframe.value),
            ).fetchall()

        merged: list[list[pd.Timestamp]] = []
        for row in rows:
            start, end = pd.Timestamp(row["start_at"]), pd.Timestamp(row["end_at"])
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def record_quality(self, symbol: str, timeframe: Timeframe, report: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO quality_reports (symbol, timeframe, created_at, payload)"
                " VALUES (?, ?, ?, ?)",
                (symbol, timeframe.value, _now(), json.dumps(report, default=str)),
            )

    def latest_quality(self, symbol: str, timeframe: Timeframe) -> dict | None:
        """直近の品質レポート。

        rowid（挿入順）で並べる。created_at はウォールクロック由来の文字列
        なので、短時間に連続で呼ばれると同着になりうるうえ、クロック調整で
        後退することもありうる——created_atで並べていたら、その場合に
        「実際に後から挿入された方」を取り違える。rowidはこのテーブルが
        INSERT専用（DELETEしない）である限り常に挿入順と一致するので、
        created_atより頑丈な基準になる。
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM quality_reports"
                " WHERE symbol = ? AND timeframe = ?"
                " ORDER BY rowid DESC LIMIT 1",
                (symbol, timeframe.value),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def quality_history(self, symbol: str, timeframe: Timeframe) -> list[dict]:
        """品質レポートの全件を挿入順で返す（隔離レコードも含む）。

        `latest_quality()` は最新の1件しか返さない。壊れたチャンクが一部だけ
        あった取得（残りは成功した）では、その後に正規の最終サマリが記録される
        ため、隔離が起きた事実は「最新」からは見えなくなる（全チャンクが隔離
        された場合を除く）。ダッシュボードが「隔離が起きたこと自体」を、直近の
        レコードが正規サマリか隔離レコードかによらず見せられるよう、履歴を
        丸ごと返す口を `latest_quality()` とは別に用意する。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM quality_reports"
                " WHERE symbol = ? AND timeframe = ?"
                " ORDER BY rowid",
                (symbol, timeframe.value),
            ).fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def record_oos_unlock(self, period: str, reason: str) -> None:
        """ロック期間を覗いたことを記録に残す。

        一度OOSの結果を見てしまったら、そのOOSはもうOOSではない。後から
        「いつ・なぜ覗いたか」を追えないと、検証の信頼性が主張できない。

        reasonが空（空白のみ含む）だと記録の体をなさないので拒否する。
        型（strを受け取る）だけで「必須」を担保すると空文字列で素通り
        してしまい、「理由なしで覗ける」という抜け道になってしまう。
        """
        if not period.strip():
            raise ValueError("period は必須（空文字列・空白のみは不可）")
        if not reason.strip():
            raise ValueError("reason は必須（空文字列・空白のみは不可）。記録の意味がなくなる")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO oos_unlocks (period, reason, created_at) VALUES (?, ?, ?)",
                (period, reason, _now()),
            )

    def oos_unlocks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT period, reason, created_at FROM oos_unlocks ORDER BY rowid"
            ).fetchall()
        return [
            {"period": r["period"], "reason": r["reason"], "at": r["created_at"]}
            for r in rows
        ]
