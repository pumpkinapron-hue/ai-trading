"""設定の読み込み。期間分割とモデルIDはコードに直書きしない。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from aitrading.timeutil import ensure_utc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = PROJECT_ROOT / "config" / "settings.toml"


@dataclass(frozen=True)
class Period:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    locked: bool


@dataclass(frozen=True)
class Settings:
    symbol: str
    data_start: pd.Timestamp
    data_root: Path
    meta_db: Path
    periods: dict[str, Period]
    models: dict[str, str]

    def period_for(self, name: str) -> Period:
        try:
            return self.periods[name]
        except KeyError:
            raise KeyError(f"未知の期間: {name!r}（{sorted(self.periods)}）") from None

    def slice_bars(
        self, df: pd.DataFrame, period: str, *, allow_locked: bool = False
    ) -> pd.DataFrame:
        """指定期間だけを切り出す。

        **`allow_locked` は「門」ではない。** ロック解除の判定と監査記録を持って
        いるのは `edge_scan.scan_period()` のほうで、こちらはその判定結果を
        受け取って切り出すだけ。ここで `PermissionError` を投げるのは
        「`scan_period` を経由せずに直接呼んだ人」への注意喚起であって、
        `allow_locked=True` と書けば通る。

        以前はこのエラーメッセージ自身が `allow_locked=True` の書き方を教えており、
        **抜け道の使用説明書になっていた**。判定が2箇所（`scan_period` と
        ここ）に分かれていて片方だけが記録を要求している状態なので、
        メッセージは `scan_period` へ誘導する。

        過学習対策は、人間が「ちょっとだけ覗く」のを防げないと機能しない。
        """
        index = ensure_utc(df.index)
        target = self.period_for(period)
        if target.locked and not allow_locked:
            raise PermissionError(
                f"期間 {target.name!r} はロックされている。"
                " 集計するには edge_scan.scan_period() を使い、解除理由と meta を渡すこと"
                "（解除は meta.db に記録される）"
            )
        # end は日付指定なのでその日の終わりまで含める
        end = target.end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return df.loc[(index >= target.start) & (index <= end)]


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def load_settings(path: Path | None = None) -> Settings:
    path = path or DEFAULT_SETTINGS
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    periods = {
        name: Period(
            name=name,
            start=_ts(body["start"]),
            end=_ts(body["end"]),
            locked=bool(body["locked"]),
        )
        for name, body in raw["periods"].items()
    }

    data = raw["data"]
    return Settings(
        symbol=data["symbol"],
        data_start=_ts(data["start"]),
        data_root=PROJECT_ROOT / data["root"],
        meta_db=PROJECT_ROOT / data["meta_db"],
        periods=periods,
        models=dict(raw["models"]),
    )
