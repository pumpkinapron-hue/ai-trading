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
from aitrading.timeutil import Timeframe

#: 「祝日クローズ規模」とみなす欠損の長さ（分）。timeutil.is_market_open は曜日しか
#: 見ないため、祝日（12/25・1/1など）のクローズは丸1日=1440分の欠損として出続ける
#: （quality.py 冒頭の既知の制約を参照。10年で20件前後になる）。この定数は「その規模を
#: 下回る欠損＝市場が開いているはずなのにデータが無い、調査対象になりうる欠損」を
#: 見出しの数字から拾うためだけに使う表示上の閾値であり、祝日カレンダーの代用ではない
#: （祝日をどれが本物かまで判定する気なら quality.py 自身が既に「外部依存の判断なので
#: Phase 0 では入れていない」と明記している）。
_HOLIDAY_SCALE_MINUTES = 1440.0


def _ensure_utc_timestamp(value: pd.Timestamp, label: str) -> pd.Timestamp:
    """スカラーの Timestamp を tz-aware・UTCに揃える。naive は ValueError。

    `timeutil.ensure_utc` は `DatetimeIndex` 専用でスカラーには使えないため、ここに
    専用の変換を置く。`aitrading.storage.meta._ensure_utc_timestamp` /
    `aitrading.datasource.dukascopy._ensure_utc_timestamp` と同じ理由・同じ形
    （このリポジトリでスカラー版が必要になった場所は毎回この4行を個別に持っている）。
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{label} が tz-aware でない。UTCで渡すこと")
    return ts.tz_convert("UTC")


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

    saved_chunks: int
    quarantined_chunks: int

    @property
    def ok(self) -> bool:
        return self.quarantined_chunks == 0 and self.saved_chunks > 0


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


def _format_quality_summary(report: quality.QualityReport) -> str:
    """人間が見て意味のある要約を作る。

    `report.longest_gap_minutes` をそのまま見出しに使わないこと。`is_market_open` は
    曜日しか見ないため、祝日クローズは丸1日=1440分の欠損として毎回出続け、10年分では
    20件前後にもなる（quality.py 冒頭の既知の制約）。単独の見出し数値として使うと、
    「取得が途中で切れて数十分〜数時間分のデータが本当に抜けている」ときも
    「12/25の祝日が1440分の欠損として出ている」ときも同じ1440という値になり、
    どちらが起きているのか区別できない。

    ここでは欠損を「祝日規模（1440分以上）」とそれ以外に分け、後者――市場が開いて
    いるはずなのにデータが無い、調査対象になりうる欠損――の中で最長のものを報告する。
    件数（祝日規模の欠損がだいたい何件あるか）も添えて、10年で20件前後という
    見込みと桁が合っているかを人間が確認できるようにする。
    """
    ratio = report.actual_bars / report.expected_bars if report.expected_bars else 1.0
    holiday_scale = [g for g in report.gaps if g["minutes"] >= _HOLIDAY_SCALE_MINUTES]
    other = [g for g in report.gaps if g["minutes"] < _HOLIDAY_SCALE_MINUTES]
    longest_other = max((g["minutes"] for g in other), default=0.0)
    return (
        f"品質: {report.actual_bars}/{report.expected_bars} 本 ({ratio:.1%})、"
        f"欠損 {len(report.gaps)} 箇所"
        f"（祝日規模[{_HOLIDAY_SCALE_MINUTES:.0f}分以上] {len(holiday_scale)} 件、"
        f"それ以外 {len(other)} 件・最長 {longest_other:.0f} 分）"
    )


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
    start = _ensure_utc_timestamp(
        start if start is not None else settings.data_start, "start"
    )
    end = _ensure_utc_timestamp(
        end if end is not None else pd.Timestamp.now(tz="UTC"), "end"
    ).floor("min")
    if chunk_days <= 0:
        raise ValueError(f"chunk_days は正の整数であること: {chunk_days!r}")

    saved = quarantined = 0
    step = pd.Timedelta(days=chunk_days)
    covered = meta.fetched_ranges(symbol, Timeframe.M1)
    gaps = _missing_ranges(covered, start, end)

    for gap_start, gap_end in gaps:
        for chunk_start, chunk_end in _iter_chunks(gap_start, gap_end, step):
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
        return FetchOutcome(saved_chunks=saved, quarantined_chunks=quarantined)
    report = quality.check(
        stored, symbol, Timeframe.M1, expected_start=start, expected_end=end
    )
    meta.record_quality(symbol, Timeframe.M1, report.to_dict())
    print(_format_quality_summary(report))
    return FetchOutcome(saved_chunks=saved, quarantined_chunks=quarantined)


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

    if outcome.saved_chunks == 0:
        print("エラー: 1本も取得できなかった")
        return 1
    if outcome.quarantined_chunks:
        print(f"警告: {outcome.quarantined_chunks} チャンクを隔離した（未取得のまま残っている）")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
