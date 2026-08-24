"""時刻の規約。すべてUTCのtz-awareで扱い、市場ローカル時刻は都度変換する。

固定オフセット（「NYは+9時間」など）で書くと夏時間の切り替え週に1時間ずれるため、
必ず各市場のタイムゾーン名を経由する。
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

TOKYO_TZ = "Asia/Tokyo"
LONDON_TZ = "Europe/London"
NEWYORK_TZ = "America/New_York"

#: FXの1日の区切り（NYクローズ）
NY_CLOSE_HOUR = 17


class Timeframe(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1_NY = "1D_ny"
    D1_JST = "1D_jst"
    W1_NY = "1W_ny"
    W1_JST = "1W_jst"

    @property
    def delta(self) -> pd.Timedelta | None:
        """固定長の期間。日足・週足は夏時間で長さが変わるので None。"""
        fixed = {
            Timeframe.M1: pd.Timedelta(minutes=1),
            Timeframe.M5: pd.Timedelta(minutes=5),
            Timeframe.M15: pd.Timedelta(minutes=15),
            Timeframe.H1: pd.Timedelta(hours=1),
            Timeframe.H4: pd.Timedelta(hours=4),
        }
        return fixed.get(self)

    @property
    def convention(self) -> str | None:
        """日足・週足の日境界系統。"""
        if self in (Timeframe.D1_NY, Timeframe.W1_NY):
            return "ny"
        if self in (Timeframe.D1_JST, Timeframe.W1_JST):
            return "jst"
        return None


class Session(str, Enum):
    TOKYO = "TOKYO"
    LONDON = "LONDON"
    NEWYORK = "NEWYORK"
    LDN_NY_OVERLAP = "LDN_NY_OVERLAP"
    OFF = "OFF"


def ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """tz-aware であることを強制し、UTCに揃える。"""
    index = pd.DatetimeIndex(index)
    if index.tz is None:
        raise ValueError("naive な DatetimeIndex は受け付けない。tz-aware にすること")
    return index.tz_convert("UTC")


def _local_minutes(index: pd.DatetimeIndex, tz: str) -> pd.Series:
    """市場ローカル時刻の「0時からの経過分」。夏時間はtz変換が吸収する。"""
    local = index.tz_convert(tz)
    return pd.Series(local.hour * 60 + local.minute, index=index)


def session_labels(index: pd.DatetimeIndex) -> pd.Series:
    """各時刻にセッションのタグを付ける。ロンドンとNYが重なる帯は専用ラベル。"""
    index = ensure_utc(index)

    tokyo = _local_minutes(index, TOKYO_TZ).between(9 * 60, 17 * 60, inclusive="left")
    london = _local_minutes(index, LONDON_TZ).between(8 * 60, 16 * 60 + 30, inclusive="left")
    newyork = _local_minutes(index, NEWYORK_TZ).between(8 * 60, 17 * 60, inclusive="left")

    labels = pd.Series(Session.OFF, index=index, dtype=object)
    labels[tokyo] = Session.TOKYO
    labels[london] = Session.LONDON
    labels[newyork] = Session.NEWYORK
    labels[london & newyork] = Session.LDN_NY_OVERLAP
    return labels


def is_market_open(index: pd.DatetimeIndex) -> pd.Series:
    """FX市場が開いているか。日曜NY17:00オープン〜金曜NY17:00クローズ。

    週末を「欠損」と誤検出しないために要る（品質チェックが毎週偽陽性を出さないように）。
    """
    index = ensure_utc(index)
    local = index.tz_convert(NEWYORK_TZ)
    dow = pd.Series(local.dayofweek, index=index)  # 月=0 … 日=6
    minutes = _local_minutes(index, NEWYORK_TZ)
    close = NY_CLOSE_HOUR * 60

    opened = pd.Series(True, index=index)
    opened[dow == 5] = False                       # 土曜は終日クローズ
    opened[(dow == 6) & (minutes < close)] = False  # 日曜17:00前
    opened[(dow == 4) & (minutes >= close)] = False  # 金曜17:00以降
    return opened


def trading_day_start(index: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex:
    """各時刻が属する「取引日」の開始時刻（UTC）を返す。

    convention="ny"  : 17:00 America/New_York 区切り（夏時間に追従）
    convention="jst" : 00:00 Asia/Tokyo 区切り（夏時間なし）
    """
    index = ensure_utc(index)

    if convention == "ny":
        tz, offset = NEWYORK_TZ, pd.Timedelta(hours=NY_CLOSE_HOUR)
    elif convention == "jst":
        tz, offset = TOKYO_TZ, pd.Timedelta(0)
    else:
        raise ValueError(f"未知の日境界: {convention!r}（'ny' か 'jst'）")

    local = index.tz_convert(tz)
    # offset を引いてから日付を取ると、区切り時刻より前は前日に落ちる
    day = (local - offset).normalize()
    starts = day + offset
    return pd.DatetimeIndex(starts).tz_convert("UTC")


def trading_day_label(index: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex:
    """取引日の「暦日ラベル」（UTC正規化した日付）。

    NY基準では日曜17:00開始の足が慣習的に「月曜」の取引日なので、区切り時刻ぶん
    進めてから日付を取る。週足のグループ化でこれを使わないと、週の切れ目が1日ずれる。
    """
    start = trading_day_start(index, convention)
    if convention == "ny":
        tz, shift = NEWYORK_TZ, pd.Timedelta(hours=24 - NY_CLOSE_HOUR)
    else:
        tz, shift = TOKYO_TZ, pd.Timedelta(0)
    local = pd.DatetimeIndex(start).tz_convert(tz) + shift
    return pd.DatetimeIndex(local.normalize()).tz_convert("UTC")
