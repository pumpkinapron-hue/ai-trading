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

from aitrading.bars import resample, source_coverage
from aitrading.config import Settings, load_settings
from aitrading.storage.lake import Lake
from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe, ensure_utc_timestamp

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


def build(
    settings: Settings,
    lake: Lake,
    *,
    timeframes: list[Timeframe] | None = None,
    as_of: pd.Timestamp | None = None,
    meta: Meta | None = None,
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
    as_of = ensure_utc_timestamp(
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
        # **派生足は毎回まるごと作り直す（既存と結合しない）。**
        #
        # 派生足は1分足の純粋な関数なので、蓄積する意味が無い。それどころか
        # 結合すると詰む: 1分足に内側の穴があるとその期間の上位足は欠けたまま
        # 保存され、あとで穴が埋まって同じコマンドを実行すると、同じ open_time の
        # 値が変わって `Lake._merge_year` が ValueError を投げる。以後その時間軸は
        # 何度実行しても生成できなくなり、parquet を手で消すまで復帰しない。
        # 内側の穴は絵空事ではなく、fetch_data.py が壊れたチャンクを隔離した
        # 跡がそのまま内側の穴になる。
        #
        # 作り直しなら、値が変わっても常に最新の1分足と整合し、詰まない。
        lake.drop(settings.symbol, timeframe)
        lake.save(settings.symbol, timeframe, derived.reset_index())

        # **派生足には、これまで品質検査が1つも走っていなかった。**
        # `quality.check()` は可変長（日足・週足）を拒否するので、汚染されうる
        # 時間軸ほど検査できない。`resample()` は期間の内側の穴を見ないため、
        # 隔離チャンクに接した日足は中身が1割でも「確定足」として保存される。
        # 充足率を出しておけば、少なくとも気づける。
        coverage = source_coverage(source, derived)
        report = {
            "timeframe": timeframe.value,
            "bars": len(derived),
            "min_source_coverage": float(coverage.min()),
            "median_source_coverage": float(coverage.median()),
            "bars_below_90pct": int((coverage < 0.90).sum()),
        }
        if meta is not None:
            meta.record_quality(settings.symbol, timeframe, report)
        print(
            f"{timeframe.value}: {len(derived)} 本"
            f"（元データ充足率 最低 {100 * coverage.min():.1f}% /"
            f" 90%未満 {report['bars_below_90pct']} 本）"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1分足から上位足・日足2系統を生成する")
    parser.add_argument("--timeframe", action="append", help="生成する時間軸（複数可）")
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        selected = [Timeframe(t) for t in args.timeframe] if args.timeframe else None
        build(
            settings,
            Lake(settings.data_root),
            timeframes=selected,
            meta=Meta(settings.meta_db),
        )
    except ValueError as exc:
        print(f"エラー: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
