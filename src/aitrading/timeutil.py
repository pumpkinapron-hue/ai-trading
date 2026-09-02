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


def ensure_utc_timestamp(value, name: str = "timestamp") -> pd.Timestamp:
    """スカラーの時刻を tz-aware・UTC に揃える。`ensure_utc` のスカラー版。

    naive は `ValueError`、tz-aware だが UTC でないものは UTC へ**変換する**
    （検証だけして変換しない、をやらないこと）。同じ規約の実装がモジュールごとに
    散ると、`Timestamp.year` のようにローカルのタイムゾーンで答えが変わる操作を
    したときに、経路によって結果が違うという壊れ方をする——実際に `Lake.load` が
    そうなっていた（同じ瞬間を UTC で渡すと10本、NY表記で渡すと5本しか返らない）。

    この関数が無かったせいで、同等の実装が4コピー（dukascopy / meta /
    fetch_data / build_bars）とインライン2箇所に増えていた。
    """
    ts = pd.Timestamp(value)
    if ts.tz is None:
        raise ValueError(f"{name} は tz-aware で渡すこと（naive は受け付けない）")
    return ts.tz_convert("UTC")


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


def _convention_tz(convention: str) -> tuple[str, pd.Timedelta]:
    """日境界系統ごとの（タイムゾーン, 暦日00:00からの区切り時刻）。"""
    if convention == "ny":
        return NEWYORK_TZ, pd.Timedelta(hours=NY_CLOSE_HOUR)
    if convention == "jst":
        return TOKYO_TZ, pd.Timedelta(0)
    raise ValueError(f"未知の日境界: {convention!r}（'ny' か 'jst'）")


def _localize_boundary(wall: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    """壁時計の境界時刻をその市場の実時刻に戻す。

    NYの夏時間切り替えは日曜02:00ローカルで起きるので、区切りである17:00と
    暦日の00:00はどちらも「存在しない時刻」にも「二度ある時刻」にもならない。
    将来この前提が崩れる区切りを足したときに黙って通らないよう、既定の raise のままにする。
    """
    return pd.DatetimeIndex(wall).tz_localize(tz).tz_convert("UTC")


def trading_day_start(index: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex:
    """各時刻が属する「取引日」の開始時刻（UTC）を返す。

    convention="ny"  : 17:00 America/New_York 区切り（夏時間に追従）
    convention="jst" : 00:00 Asia/Tokyo 区切り（夏時間なし）
    """
    index = ensure_utc(index)

    tz, offset = _convention_tz(convention)

    # 壁時計（naive ローカル）で計算する。tz-aware のまま絶対時間で ±offset すると、
    # 夏時間の切り替え日（23時間 / 25時間）で normalize() との往復がズレて
    # 区切りが1時間ずれる。壁時計なら「17:00」は常に「17:00」。
    wall = index.tz_convert(tz).tz_localize(None)
    # offset を引いてから日付を取ると、区切り時刻より前は前日に落ちる
    day = (wall - offset).normalize()
    return _localize_boundary(day + offset, tz)


def trading_day_label(index: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex:
    """取引日の「暦日ラベル」（UTC正規化した日付）。

    NY基準では日曜17:00開始の足が慣習的に「月曜」の取引日なので、区切り時刻ぶん
    進めてから日付を取る。週足のグループ化でこれを使わないと、週の切れ目が1日ずれる。
    """
    tz, offset = _convention_tz(convention)
    start = trading_day_start(index, convention)
    # 区切りが 00:00 でない系統（NY）だけ、区切り時刻ぶん進めてから暦日を読む。
    # JST は offset=0 なので進めない（剰余を取らないと24時間ぶん進んで1日ずれる）。
    day = pd.Timedelta(hours=24)
    shift = (day - offset) % day
    # ここも壁時計で行う（理由は trading_day_start と同じ）。
    wall = pd.DatetimeIndex(start).tz_convert(tz).tz_localize(None)
    return _localize_boundary((wall + shift).normalize(), tz)


def local_trading_date(index: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex:
    """各時刻が属する取引日の「市場ローカルの暦日」（naive）。

    `trading_day_label` と同じ日を指すが、UTCの瞬間ではなく暦日そのものを返す。
    UTC表現のまま日付を読むと、JSTでは月曜（= 日曜15:00Z）が日曜に見えて
    週のグループ化が1日ずれる。日付として扱うときは必ずこちらを使う。
    """
    tz, _ = _convention_tz(convention)
    label = pd.DatetimeIndex(trading_day_label(index, convention))
    return label.tz_convert(tz).tz_localize(None)


def trading_period_start(local_dates: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex:
    """市場ローカルの暦日（naive）→ その取引日が始まる時刻（UTC）。

    `local_trading_date` の逆写像。NY基準では「月曜の取引日」は日曜17:00に始まる。
    週足のように暦から境界を決めたいとき（観測されたデータの端ではなく）に使う。

    グループ化（どの足に入れるか）と境界計算（いつ始まりいつ終わるか）を同じ
    モジュールの表裏として置く。別々の場所で別々に計算すると夏時間の日に食い違い、
    足の中身と close_time がズレて先読みになる（実際に起きた）。
    """
    tz, offset = _convention_tz(convention)
    day = pd.Timedelta(hours=24)
    wall = pd.DatetimeIndex(local_dates).normalize() - ((day - offset) % day)
    return _localize_boundary(wall, tz)
