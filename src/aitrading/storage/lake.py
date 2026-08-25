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

    def save(self, symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None:
        """年ごとに分割して保存する。既存があれば結合して重複を落とす。

        再取得が冪等になるので、途中で失敗しても同じコマンドを再実行できる。

        年ごとに既存データと結合したあとも validate_bars をもう一度通す。
        新しいバッチ単体が妥当でも、既存の1本と部分的に時間帯が重なるだけで
        結合後には重なりバーになる、といった壊れ方は結合前の検証だけでは
        見えない。重複除去は結合直後・再検証の前に行う（重複 open_time は
        validate_bars 自身が拒否するため、順序を逆にすると idempotent な
        再保存が常にエラーになってしまう）。
        """
        df = validate_bars(df, timeframe)
        if df.empty:
            return

        directory = self._dir(symbol, timeframe)
        directory.mkdir(parents=True, exist_ok=True)

        for year, group in df.groupby(df["open_time"].dt.year):
            path = self._path(symbol, timeframe, int(year))
            if path.exists():
                group = pd.concat([pd.read_parquet(path), group], ignore_index=True)
            merged = group.drop_duplicates(subset="open_time", keep="last")
            merged = validate_bars(merged, timeframe)
            merged.to_parquet(path, index=False)

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
