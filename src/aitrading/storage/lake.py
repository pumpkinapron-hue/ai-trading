"""Parquet レイク。データアクセスはここに一本化する。

load() の as_of がキーワード必須引数なのは意図的。省略できると、
時点Tのシミュレーション中に未来のバーを読むコードが書けてしまう。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aitrading.datasource.base import BAR_COLUMNS, TIME_COLUMNS, validate_bars
from aitrading.timeutil import Timeframe


def _empty_bars() -> pd.DataFrame:
    """列・dtypeが validate_bars 後のデータと一致する、0行のフレーム。

    datetime64 の分解能は pandas のバージョンに依存する（例: このリポジトリの
    pandas 3.0.5 では date_range(tz="UTC") が [us] を返す）。ここを
    "datetime64[ns, UTC]" のように決め打ちすると、実データ（validate_bars→
    parquet 往復）の dtype とズレて、未取得シンボル（0件）のときだけ型の違う
    フレームが返る。date_range(periods=0, ...) で実データと同じ経路から
    型を借りることでこのズレを構造的に防ぐ。
    """
    empty_time = pd.date_range("1970-01-01", periods=0, tz="UTC")
    body = {
        column: empty_time if column in TIME_COLUMNS else pd.Series(dtype="float64")
        for column in BAR_COLUMNS
    }
    return pd.DataFrame(body)


def _merge_year(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """1年ぶんの既存データと新規バッチを、値の衝突を検出しながら結合する。

    同じ open_time が両方にあっても、他の列の値まで完全に一致していれば
    黙って一本化する（同じ範囲を同じ値で再取得するのは常に成功する必要が
    ある——save() が謳う「再取得の冪等性」はこの前提の上に成り立つ）。

    値が食い違う場合は ValueError にする。同じ open_time で違う値が来るのは
    データソース側で何かが変わったということであり、keep="last" で黙って
    上書きしてはいけない。どちらが正しいかはこの関数には判断できないため、
    判断を呼び出し側（人間）に投げ返す。

    戻り値はまだ validate_bars を通していない（呼び出し側の責務）。
    """
    combined = pd.concat([existing, new], ignore_index=True)

    duplicated = combined["open_time"].duplicated(keep=False)
    if duplicated.any():
        variants = combined.loc[duplicated].groupby("open_time").nunique()
        conflicts = variants.index[(variants > 1).any(axis=1)]
        if len(conflicts) > 0:
            raise ValueError(
                "open_time の値が既存データと衝突している"
                f"（データソース側の変更を疑う）: {list(conflicts[:5])}"
            )

    return combined.drop_duplicates(subset="open_time", keep="last")


class Lake:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, symbol: str, timeframe: Timeframe) -> Path:
        return self.root / "bars" / symbol / timeframe.value

    def _path(self, symbol: str, timeframe: Timeframe, year: int) -> Path:
        return self._dir(symbol, timeframe) / f"{year}.parquet"

    def available_years(self, symbol: str, timeframe: Timeframe) -> list[int]:
        directory = self._dir(symbol, timeframe)
        if not directory.exists():
            return []
        return sorted(int(p.stem) for p in directory.glob("*.parquet"))

    def drop(self, symbol: str, timeframe: Timeframe) -> int:
        """その銘柄・時間軸の保存済みデータを消す。消したファイル数を返す。

        **生成物（上位足・日足）を作り直すためのもの。** 1分足は取得物なので
        消せないようにしてある――消したら再取得（数時間）でしか復元できない。

        生成物を結合して積み上げると、元の1分足が変わったときに同じ open_time の
        値が変わり、`save` の値衝突検出で詰む。生成物は毎回作り直すのが正しい。
        """
        if timeframe is Timeframe.M1:
            raise ValueError(
                "1分足は取得物であって生成物ではない。drop してはいけない"
                "（消すと再取得でしか復元できない）"
            )
        removed = 0
        for year in self.available_years(symbol, timeframe):
            self._path(symbol, timeframe, year).unlink()
            removed += 1
        return removed

    def save(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        """年ごとに分割して保存する。既存があれば結合して重複を落とす。

        再取得が冪等になるので、途中で失敗しても同じコマンドを再実行できる
        ——ただしそれは値が変わっていない場合の話で、同じ open_time に違う
        値が来たら _merge_year が ValueError にする（黙って上書きしない）。

        全年ぶんをメモリ上で用意し終える（結合・重複除去・validate_bars）まで
        ディスクには一切書き込まない。複数年にまたがるバッチの一部の年だけ
        検証に失敗した場合に、既に検証を通った別の年のファイルだけ書き換わって
        しまうと、呼び出し側は例外を見てもどの年が書けたか分からず、レイクが
        部分的に矛盾した状態になる。
        """
        df = validate_bars(df, timeframe)
        if df.empty:
            return

        # Phase 1: 準備。結合・衝突検出・検証をすべてメモリ上で行う。
        # ここで例外が飛べば、ディスクにはまだ何も触れていない。
        prepared: dict[int, pd.DataFrame] = {}
        for year, group in df.groupby(df["open_time"].dt.year):
            year = int(year)
            path = self._path(symbol, timeframe, year)
            if path.exists():
                group = _merge_year(pd.read_parquet(path), group)
            prepared[year] = validate_bars(group, timeframe)

        # Phase 2: 書き込み。すべての年の準備が成功したあとにのみ実行する。
        directory = self._dir(symbol, timeframe)
        directory.mkdir(parents=True, exist_ok=True)
        for year, merged in prepared.items():
            merged.to_parquet(self._path(symbol, timeframe, year), index=False)

    def load(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        as_of: pd.Timestamp,
        start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """as_of 時点で確定しているバーだけを返す。

        close_time <= as_of が条件。形成中の足は返さない。as_of がキーワード
        必須引数なのは意図的（このファイルのモジュールdocstring参照）。
        """
        as_of = pd.Timestamp(as_of)
        if as_of.tz is None:
            raise ValueError("as_of は tz-aware で渡すこと")

        if start is not None:
            start = pd.Timestamp(start)
            if start.tz is None:
                raise ValueError("start は tz-aware で渡すこと")

        years = self.available_years(symbol, timeframe)
        if start is not None:
            years = [y for y in years if y >= start.year]
        years = [y for y in years if y <= as_of.year]

        frames = [pd.read_parquet(self._path(symbol, timeframe, y)) for y in years]
        df = pd.concat(frames, ignore_index=True) if frames else _empty_bars()

        df = df.loc[df["close_time"] <= as_of]
        if start is not None:
            df = df.loc[df["open_time"] >= start]

        return df.sort_values("open_time").set_index("open_time")
