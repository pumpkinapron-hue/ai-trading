"""データ取得CLI。

`data/` はGit管理外なので、別PCではこれを実行してレイクを再構築する。10年ぶん・
約370万本の1分足を一度に要求すると、失敗したときのやり直しが大きすぎる。そのため
`chunk_days` ごとに区切って取得し、`Meta.fetched_ranges()` から前回までの進捗を
読んで続きから再開する。

`validate_bars`（Task 3）は壊れたバー（NaN・Ask<Bid・重複・期間の重なりなど）を
握りつぶさず `ValueError` にする。そのため、10年ぶんの取得の途中で実データに1本でも
壊れたバーが混ざっていると、何も対策しなければ取得全体がそこで止まる。ここでは
「壊れたチャンクだけ隔離して `quality_reports` に記録し、残りの取得は続ける」ことで
それを避ける（隔離は黙って捨てることではない――記録が残るので、あとで人間が
チャンクの範囲とエラー内容を確認できる）。

aitrading パッケージの import に `sys.path` 操作は不要。`pyproject.toml`
(`[tool.hatch.build.targets.wheel] packages = ["src/aitrading"]`) を `uv sync` が
editable install するため、`uv run python scripts/fetch_data.py` のようにこの
ファイルを直接実行しても `import aitrading` は素通りする（実機で確認済み）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Iterator

import pandas as pd

from aitrading import quality
from aitrading.config import Settings, load_settings
from aitrading.datasource.base import BarSource
from aitrading.datasource.dukascopy import DukascopySource
from aitrading.storage.lake import Lake
from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe, ensure_utc_timestamp

def _parse_cli_timestamp(value: str) -> pd.Timestamp:
    """CLI引数の日付文字列をUTCのtz-awareにする。

    `pd.Timestamp(value, tz="UTC")` は、`value` がtzinfo付きの `datetime`/`Timestamp`
    *オブジェクト* だと「tzinfo持ちの入力とtzを同時に渡せない」で `ValueError` になる
    （pandas 3.0.5 で実機確認済み）。文字列の場合はこの衝突は起きず、`tz=` はむしろ
    「その文字列が指す時刻をUTCへ変換する」側として働く。ただし単なる文字列パースの
    暗黙動作に頼らず、ここでは常に同じ経路（naiveならlocalize、tz付きならconvert）で
    明示的に正規化する。日付だけの `--start 2026-01-05` という通常の使い方（=naive）を
    普通に通しつつ、稀にオフセット付き文字列が来ても例外にせず素直にUTCへ変換する。
    """
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _missing_ranges(
    covered: list[tuple[pd.Timestamp, pd.Timestamp]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """要求区間 `[start, end)` のうち、まだ取得済みでない部分区間を返す。

    `Meta.fetched_ranges()` は「どこまで取得済みか」を返すために作られているのに、
    素直に書くと `fetch()` は毎回 `settings.data_start` から始めてしまい、10年分の
    取得が途中で落ちたら次の実行が最初からやり直しになる。ここで要求区間と既取得区間
    （`covered`。`fetched_ranges()` の戻り値そのままで、開始時刻でソート済み・隣接区間は
    マージ済みという前提）の差分を取り、まだ無い部分だけを対象にする。

    `covered` は「隔離されたチャンクを含まない」――壊れたチャンクは `record_fetch` を
    呼ばないので、ここでは常に「未取得」として扱われる。次回の呼び出しでも同じ区間が
    ここに現れて再度取得を試みることになるが、それは意図している。壊れたデータは
    ソース側が直らない限り何度取得しても壊れたままなので、隔離を繰り返すだけで残りの
    取得は止まらない。逆に「一度隔離したら二度と取得しに行かない」設計にすると、
    ソース側が直っても人間が気づいて明示的に再取得しない限り永久に穴が残ることになる。
    """
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    for covered_start, covered_end in covered:
        if covered_end <= cursor or covered_start >= end:
            continue  # 要求区間と重ならない既取得区間は無関係
        if covered_start > cursor:
            gaps.append((cursor, min(covered_start, end)))
        cursor = max(cursor, min(covered_end, end))
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def _iter_chunks(
    start: pd.Timestamp, end: pd.Timestamp, step: pd.Timedelta
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """`[start, end)` を `step` ごとに区切る。最後のチャンクは `end` で切り詰める。"""
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + step, end)
        yield cursor, chunk_end
        cursor = chunk_end


@dataclass(frozen=True)
class FetchOutcome:
    """取得1回の結果。終了コードに落とすために使う。

    全チャンクが隔離されても例外は出ないので、戻り値を見ないと
    「1本も取れていないのに成功」になる。別PCで数時間かけて回す用途では、
    シェルやタスクスケジューラが $? を見て成否を判断する。
    """

    requested_chunks: int
    saved_chunks: int
    quarantined_chunks: int

    @property
    def ok(self) -> bool:
        """取りに行くべきものが全部取れたか。

        **「取りに行く区間が無かった」を失敗にしないこと。** 再開機能
        （`_missing_ranges`）があるので、取得済みの状態で再実行すると
        `saved_chunks == 0` になる。これは正常系（何もする必要が無かった）
        なのに、`saved_chunks > 0` を成功条件にすると失敗として返る。
        日次でタスクスケジューラに登録すると、追いついた翌日から毎晩
        「失敗」を報告し続けることになる――`quality.py` が言う
        「毎週偽陽性が出るとアラートとして誰も見なくなる」のと同じ壊れ方。
        """
        if self.requested_chunks == 0:
            return True
        return self.quarantined_chunks == 0 and self.saved_chunks > 0

    @property
    def exit_code(self) -> int:
        """`main()` が返す終了コード。判定式をここ1箇所に置く。"""
        if self.requested_chunks and self.saved_chunks == 0:
            return 1
        if self.quarantined_chunks:
            return 2
        return 0


def _quarantine(
    meta: Meta,
    symbol: str,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    bars: pd.DataFrame | None,
    error: Exception,
) -> None:
    """壊れたチャンクをレイクへ保存せず、`quality_reports` へ理由付きで記録する。

    「どのチャンクが・なぜ落ちたか」が後から分かることが目的なので、通常の品質
    レポート（`QualityReport.to_dict()`）とは別の形の dict にする――
    `expected_bars` や `gaps` など、検証を通っていないデータには意味を持たせられない
    フィールドを、あたかも算出できたかのように詰めて記録すると、あとで読む人を
    誤解させる。`status="quarantined"` を目印に、正規の品質サマリと区別できるように
    しておく。

    `bars` が `None` になるのは、`source.fetch()` 自身が返す前に `validate_bars` で
    落ちた場合（本番の `DukascopySource` はこちら）。手元にデータが無いので
    本数は記録しない――0本と書くと「空のチャンクだった」と読めてしまう。
    """
    meta.record_quality(
        symbol,
        Timeframe.M1,
        {
            "status": "quarantined",
            "chunk_start": str(chunk_start),
            "chunk_end": str(chunk_end),
            "bar_count": None if bars is None else int(len(bars)),
            "error": str(error),
        },
    )
    print(
        f"[隔離] {chunk_start:%Y-%m-%d %H:%M} 〜 {chunk_end:%Y-%m-%d %H:%M}: "
        f"検証エラーのため保存せず継続（{error}）"
    )


# 実体は quality.py（dashboard/app.py と共有）。ここに残しているのは呼び出し側
# （下の fetch() や tests/test_scripts.py）の互換のため――関数名を変えると
# 「取得直後にCLIへ出す要約」と「品質モジュール本体」が別物であるかのように
# 読めてしまうので、モジュールローカルな別名として残す。
_format_quality_summary = quality.format_summary


def fetch(
    settings: Settings,
    source: BarSource,
    lake: Lake,
    meta: Meta,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    chunk_days: int = 30,
) -> FetchOutcome:
    """1分足を分割して取得し、レイクへ保存して品質レポートを記録する。

    - 前回までの進捗（`meta.fetched_ranges`）を読み、まだ無い区間だけを取得する
      （`_missing_ranges`）。10年分の取得が途中で落ちても、次の実行は続きから。
    - 壊れたチャンク（`validate_bars` が `ValueError` にするもの）は隔離して
      `quality_reports` に記録し、残りのチャンクの取得は続ける。ここで
      `source.fetch()` 自体の例外（ネットワーク断など）は握りつぶさない――
      「データの中身が壊れている」と「取得そのものができない」は別の障害であり、
      後者まで隔離して続行すると、直っていない接続不良の下で延々と空振りし続ける。
    - 0本だったチャンク（週末・祝日）も `record_fetch` する。しないと、休場中の
      区間が「未取得」のまま残り、再実行のたびに取りに行くことになる。
    - 品質レポートには要求した範囲（`expected_start`/`expected_end`）を渡す。
      渡さないと、取得が途中で切れたときこそ観測データ自身の端が母数になり
      「問題なし」と報告されてしまう（quality.py 参照）。
    """
    symbol = settings.symbol
    start = ensure_utc_timestamp(
        start if start is not None else settings.data_start, "start"
    )
    end = ensure_utc_timestamp(
        end if end is not None else pd.Timestamp.now(tz="UTC"), "end"
    ).floor("min")
    if chunk_days <= 0:
        raise ValueError(f"chunk_days は正の整数であること: {chunk_days!r}")

    saved = quarantined = requested = 0
    step = pd.Timedelta(days=chunk_days)
    covered = meta.fetched_ranges(symbol, Timeframe.M1)
    gaps = _missing_ranges(covered, start, end)

    for gap_start, gap_end in gaps:
        for chunk_start, chunk_end in _iter_chunks(gap_start, gap_end, step):
            requested += 1
            # `source.fetch()` も try の中に入れること。本番の `DukascopySource` は
            # 返す前に自分で `validate_bars` を通す（`dukascopy.normalize()` の
            # 最終行）ので、**壊れたバーは `lake.save()` に届く前に
            # `source.fetch()` の中で ValueError になる。** save だけを囲んで
            # いると隔離が本番で一度も発火せず、しかも `_missing_ranges` が
            # 正しく続きから再開する結果、毎回同じ壊れたチャンクで死んで
            # それより後ろが未来永劫取得されない。
            #
            # 例外の種類で分ける: ValueError は「中身が壊れている」（隔離して先へ）、
            # それ以外（ConnectionError など）は「取得そのものができない」ので
            # 握りつぶさずに落とす。ネットワーク層が ValueError を投げないことは
            # `DukascopySource.fetch` を読めば確認できる（ValueError を出すのは
            # 未対応シンボル・未対応 timeframe・naive timestamp のみで、いずれも
            # チャンクに依らず全チャンクで失敗するため即座に全滅として現れる）。
            bars = None
            try:
                bars = source.fetch(symbol, Timeframe.M1, chunk_start, chunk_end)
                lake.save(symbol, Timeframe.M1, bars)
            except ValueError as exc:
                _quarantine(meta, symbol, chunk_start, chunk_end, bars, exc)
                quarantined += 1
                continue
            meta.record_fetch(symbol, Timeframe.M1, chunk_start, chunk_end)
            saved += 1
            print(f"{chunk_start:%Y-%m-%d} 〜 {chunk_end:%Y-%m-%d}: {len(bars)} 本")

    # actual_bars と expected_bars を同じ窓で比較できるよう、読み出しも要求区間に絞る
    # （絞らないと、この fetch() 呼び出しより前に別区間で取得済みのデータまで
    # actual_bars に混ざり、両者が食い違う窓を比較することになる）。
    stored = lake.load(symbol, Timeframe.M1, as_of=end, start=start)
    if stored.empty:
        print("品質: 保存されたバーが無い")
        return FetchOutcome(
            requested_chunks=requested, saved_chunks=saved, quarantined_chunks=quarantined
        )
    report = quality.check(
        stored, symbol, Timeframe.M1, expected_start=start, expected_end=end
    )
    meta.record_quality(symbol, Timeframe.M1, report.to_dict())
    print(_format_quality_summary(report))
    return FetchOutcome(
            requested_chunks=requested, saved_chunks=saved, quarantined_chunks=quarantined
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="USD/JPY の1分足を取得してレイクへ保存する")
    parser.add_argument("--start", help="開始日 (YYYY-MM-DD)。既定は settings.toml の値")
    parser.add_argument("--end", help="終了日 (YYYY-MM-DD)。既定は現在時刻")
    parser.add_argument("--chunk-days", type=int, default=30, help="1回の取得日数")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        outcome = fetch(
            settings,
            DukascopySource(),
            Lake(settings.data_root),
            Meta(settings.meta_db),
            start=_parse_cli_timestamp(args.start) if args.start else None,
            end=_parse_cli_timestamp(args.end) if args.end else None,
            chunk_days=args.chunk_days,
        )
    except ValueError as exc:
        print(f"エラー: {exc}")
        return 1

    if outcome.requested_chunks == 0:
        print("取得済み。新たに取りに行く区間は無かった")
    elif outcome.saved_chunks == 0:
        print("エラー: 1本も取得できなかった")
    elif outcome.quarantined_chunks:
        print(f"警告: {outcome.quarantined_chunks} チャンクを隔離した（未取得のまま残っている）")
    return outcome.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
