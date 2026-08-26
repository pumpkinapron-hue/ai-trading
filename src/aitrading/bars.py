"""1分足から上位足・日足2系統を生成する。

取得するのは1分足だけで、他はすべてここで作る生成物。
`close_time` を「その期間が終わった時刻」に置くのが要点で、
ここを誤るとマルチタイムフレーム解析が先読みになる（先読み防止 第3層）。

期間が確定しているかの判定は、時間軸によらず
「その期間が元データの範囲に丸ごと収まっているか」の一点で決める。
本数で判定すると、市場が閉まっている時間を含むバケット（金曜NYクローズ後や
日曜オープン前）を「欠損」と誤認して確定足を捨ててしまう。

日境界の暦計算（どの取引日・どの週に属するか、その期間がいつ始まるか）は
すべて `timeutil` に置く。グループ化と境界計算を別々に持つと夏時間の日に
食い違い、足の中身と `close_time` がズレて先読みになる。
"""

from __future__ import annotations

import pandas as pd

from aitrading.timeutil import (
    Timeframe,
    ensure_utc,
    local_trading_date,
    trading_period_start,
)

_AGGREGATION = {
    "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last",
    "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last",
    "volume": "sum",
}

#: 返り値の列順。0行のときも非0行のときもこれで揃える。
_OUTPUT_COLUMNS = ["close_time", *_AGGREGATION]

_WEEKLY = (Timeframe.W1_NY, Timeframe.W1_JST)


def resample(bars_1m: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """1分足を上位足に集約する。入力・返り値ともに index は `open_time`（UTC）。"""
    if timeframe is Timeframe.M1:
        raise ValueError("1m はデータソースから取得するもので、生成対象ではない")

    index = ensure_utc(pd.DatetimeIndex(bars_1m.index))
    if bars_1m.empty:
        return _empty_output()

    source = bars_1m.set_axis(index, axis=0).sort_index()
    data_start = source.index.min()
    data_end = ensure_utc(pd.DatetimeIndex(source["close_time"])).max()

    if timeframe.convention is None:
        out = _fixed_length(source, timeframe.delta)
    else:
        out = _variable_length(source, timeframe)

    # 期間が元データの範囲に丸ごと収まっていなければ、その足はまだ確定していない。
    # 右端だけ見ると、途中から始まるデータの先頭が「5分のうち2分」でも確定足として
    # 出てしまう。あとで前を埋めて再生成すると同じ open_time の値が変わり、
    # Lake.save の値衝突検出に引っかかる（＝静かに間違うのではなく壊れる）。
    out = out.loc[(out.index >= data_start) & (out["close_time"] <= data_end)]
    return out.loc[:, _OUTPUT_COLUMNS].rename_axis("open_time")


def _fixed_length(source: pd.DataFrame, delta: pd.Timedelta) -> pd.DataFrame:
    """固定長（5分〜4時間）。バケットの端は暦どおりで、夏時間の影響を受けない。"""
    out = source.resample(delta, label="left", closed="left").agg(_AGGREGATION)
    # 1本も存在しないバケット（土曜など）は集約結果が NaN になる。欠損ではなく
    # 「そもそも市場が無い」ので、足として作らずに落とす。
    out = out.dropna(subset=["bid_open"])
    out["close_time"] = pd.DatetimeIndex(out.index) + delta
    return out


def _variable_length(source: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """日足・週足。夏時間で1本の長さが変わるので、暦のほうから境界を決める。"""
    convention = timeframe.convention
    dates = local_trading_date(source.index, convention)

    if timeframe in _WEEKLY:
        # 週の代表は「その取引日が属する週の月曜」。データに月曜が無くても
        # 境界がずれないよう、観測された取引日の最小値ではなく暦から決める。
        keys = dates - pd.to_timedelta(dates.dayofweek, unit="D")
        step = pd.Timedelta(days=7)
    else:
        keys = dates
        step = pd.Timedelta(days=1)

    out = source.groupby(pd.Index(keys, name="local_date"), sort=True).agg(_AGGREGATION)

    # keys は naive なローカル暦日なので、+1日 / +7日 は素直な加算でよい
    # （夏時間の伸縮は naive → UTC に戻すときに tz 側が吸収する）。
    boundaries = pd.DatetimeIndex(out.index)
    out.index = trading_period_start(boundaries, convention)
    out["close_time"] = trading_period_start(boundaries + step, convention)
    return out


def _empty_output() -> pd.DataFrame:
    """列・index名が非0行の返り値と一致する、0行のフレーム。

    dtype を決め打ちせず date_range(periods=0) から借りるのは `Lake._empty_bars`
    と同じ理由（pandas のバージョンで datetime64 の分解能が変わる）。
    """
    empty_time = pd.date_range("1970-01-01", periods=0, tz="UTC")
    body: dict[str, object] = {"close_time": empty_time}
    for column in _AGGREGATION:
        body[column] = pd.Series(dtype="float64")
    return pd.DataFrame(body, index=empty_time.rename("open_time"))
