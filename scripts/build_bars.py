"""上位足・日足2系統の生成CLI。

取得するのは1分足だけで、他の7時間軸（5分・15分・1時間・4時間・日足NY/JST・
週足NY/JST）はすべてここで作る生成物。レイクの1分足が変わっていなければ、
いつ再生成しても同じ結果になる。

aitrading パッケージの import に `sys.path` 操作は不要（scripts/fetch_data.py の
モジュールdocstring参照。editable install により素の import で解決できることを
実機で確認済み）。
"""

from __future__ import annotations

import argparse

import pandas as pd

from aitrading.bars import resample
from aitrading.config import Settings, load_settings
from aitrading.storage.lake import Lake
from aitrading.timeutil import Timeframe

DERIVED = [
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1_NY,
    Timeframe.D1_JST,
    Timeframe.W1_NY,
    Timeframe.W1_JST,
]


def _ensure_utc_timestamp(value: pd.Timestamp, label: str) -> pd.Timestamp:
    """スカラーの Timestamp を tz-aware・UTCに揃える。naive は ValueError。

    scripts/fetch_data.py の同名関数と同じ理由・同じ形（`timeutil.ensure_utc` は
    `DatetimeIndex` 専用でスカラーには使えない）。2つのCLIスクリプト間でモジュールを
    共有する仕組みが無く、`scripts/` はパッケージでもないため、ここでも複製している
    ――このリポジトリで既に2箇所（`aitrading.storage.meta` /
    `aitrading.datasource.dukascopy`）が同じ理由で同じ4行を個別に持っている。
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        raise ValueError(f"{label} が tz-aware でない。UTCで渡すこと")
    return ts.tz_convert("UTC")


def build(
    settings: Settings,
    lake: Lake,
    *,
    timeframes: list[Timeframe] | None = None,
    as_of: pd.Timestamp | None = None,
) -> None:
    """1分足から上位足・日足2系統を生成し、レイクへ保存する。

    `as_of` は「どの時点までの1分足を確定分とみなすか」。省略時は現在時刻
    （`pd.Timestamp.now(tz="UTC")`）。

    これは先読み防止 第4層（`Lake.load` の `as_of` カーソル）を無効化するものではない
    ――第4層が守るのは「バックテストエンジンが時点Tのシミュレーション中に未来の
    データを見てしまう」という設計ミスであり（docs/specs 5節）、その防御は各消費者が
    *自分の* `as_of` で `Lake.load` を呼ぶ時点で効く。ここで生成した派生足を、あとで
    別の消費者が `as_of=T`（T < ここでのas_of）で読めば、`close_time <= T` を満たさない
    バーは物理的に返らない。生成時にどの `as_of` を使ったかとは無関係に、読み出し側の
    `as_of` がそのつど効く。加えて `resample()` 自身も「元データの範囲に丸ごと収まって
    いない期間は確定足として出さない」（bars.py 参照）ため、まだ閉じていない期間の
    バーが混ざることもない。

    それでも `as_of` を呼び出し側から差し込めるようにしているのは別の理由――
    `pd.Timestamp.now(tz="UTC")` に関数内部で暗黙に依存させると、この関数は
    「呼ぶたびに違う値を見る」関数になり、テストが実行時の実時刻に依存してしまう
    （実際の壁時計に依存するテストは、実行するタイミング次第で意味が変わりかねない）。
    引数として渡せるようにすれば、テストは固定値で決定的に検証できる。CLIの
    `main()` は従来どおり既定値（現在時刻）のまま呼ぶので、通常運用の挙動は変わらない。
    """
    as_of = _ensure_utc_timestamp(
        as_of if as_of is not None else pd.Timestamp.now(tz="UTC"), "as_of"
    )
    source = lake.load(settings.symbol, Timeframe.M1, as_of=as_of)
    if source.empty:
        raise ValueError("1分足が無い。先に scripts/fetch_data.py を実行すること")

    for timeframe in timeframes or DERIVED:
        derived = resample(source, timeframe)
        if derived.empty:
            print(f"{timeframe.value}: 生成できるバーが無い")
            continue
        lake.save(settings.symbol, timeframe, derived.reset_index())
        print(f"{timeframe.value}: {len(derived)} 本")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1分足から上位足・日足2系統を生成する")
    parser.add_argument("--timeframe", action="append", help="生成する時間軸（複数可）")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        selected = [Timeframe(t) for t in args.timeframe] if args.timeframe else None
        build(settings, Lake(settings.data_root), timeframes=selected)
    except ValueError as exc:
        print(f"エラー: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
