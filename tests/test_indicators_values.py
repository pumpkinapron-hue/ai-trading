"""指標の正しさを既存ライブラリ（ta）の出力と照合する。

ここが狂うと以降の分析が全部崩れる。照合は開発時の1回で、以降は ta に
依存しない（ta は dev 依存にのみ入っている。pyproject.toml 参照）。

スライス位置について: RSI・ATR・MACDシグナルは、ta 側の実装がウォームアップの
種付け方法にわずかに異なる流儀を使っている（診断はこのファイル末尾のコメントと
task-9-report.md を参照）。どちらも Wilder 平滑化の正当な変種で、差は指数的に
減衰する。十分に収束した範囲だけを比較するため、他の指標より後ろのスライスを使う。
sma・ema・bbands・donchian は素直なローリング窓/ewmなので ta とビット単位で一致し、
rtol=1e-9 で締めている。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aitrading.indicators.core import atr, bbands, donchian, ema, hist_vol, macd, rsi, sma, vwap
from aitrading.indicators.registry import mid

ta = pytest.importorskip("ta")


def test_sma_matches_reference(sample_bars):
    from ta.trend import SMAIndicator

    close = mid(sample_bars, "close")
    expected = SMAIndicator(close, window=20).sma_indicator()
    np.testing.assert_allclose(
        sma(sample_bars).to_numpy()[19:], expected.to_numpy()[19:], rtol=1e-9
    )


def test_ema_matches_reference(sample_bars):
    from ta.trend import EMAIndicator

    close = mid(sample_bars, "close")
    expected = EMAIndicator(close, window=20).ema_indicator()
    np.testing.assert_allclose(
        ema(sample_bars).to_numpy()[19:], expected.to_numpy()[19:], rtol=1e-9
    )


def test_rsi_matches_reference(sample_bars):
    from ta.momentum import RSIIndicator

    close = mid(sample_bars, "close")
    expected = RSIIndicator(close, window=14).rsi()
    # G: ta の RSI も同じ Wilder 平滑化(alpha=1/window, adjust=False)だが、
    # diff() 先頭のNaNの扱いが違う（taは0扱いでewmを1本早く開始する）。
    # その1本分のズレが (1-1/14) の比で減衰しきる手前（cut<180あたり）だと
    # 数値が合わないことを実測で確認した。200本目以降なら複数の乱数系列で
    # 安定して rtol=1e-6 に収まる。
    np.testing.assert_allclose(
        rsi(sample_bars).to_numpy()[200:], expected.to_numpy()[200:], rtol=1e-6
    )


def test_macd_matches_reference(sample_bars):
    from ta.trend import MACD

    close = mid(sample_bars, "close")
    reference = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    got = macd(sample_bars)
    # macd線自体はta側もウォームアップNaNを引きずらないのでビット単位で一致する
    # （実測: 差0.0）。signal線は、ta内部のslow EMAが先頭26本をNaNにしてから
    # signalのewmを開始するため、自前実装（先頭からewmで再帰）と収束するまでに
    # 100本強かかる（実測、乱数系列3種で確認）。ここも十分後ろで比較する。
    np.testing.assert_allclose(
        got["macd"].to_numpy()[150:], reference.macd().to_numpy()[150:], rtol=1e-6
    )
    np.testing.assert_allclose(
        got["signal"].to_numpy()[150:], reference.macd_signal().to_numpy()[150:], rtol=1e-6
    )


def test_atr_matches_reference(sample_bars):
    from ta.volatility import AverageTrueRange

    # G: ta.volatility.AverageTrueRange は先頭 period 本を単純SMAで種付けしてから
    # Wilder再帰に切り替える。自前実装は先頭からewmで再帰する（種付けをしない）。
    # どちらも「Wilderの平滑化」の名の通った変種で、優劣はない。種付けの違いは
    # (1-1/period) の比で指数減衰するが、period=14 では30本後ろでもまだ
    # 相対誤差1.5%残っている（実測）。200本目以降なら安定して rtol=1e-6。
    reference = AverageTrueRange(
        high=mid(sample_bars, "high"),
        low=mid(sample_bars, "low"),
        close=mid(sample_bars, "close"),
        window=14,
    ).average_true_range()
    np.testing.assert_allclose(
        atr(sample_bars).to_numpy()[200:], reference.to_numpy()[200:], rtol=1e-6
    )


def test_bbands_matches_reference(sample_bars):
    from ta.volatility import BollingerBands

    reference = BollingerBands(mid(sample_bars, "close"), window=20, window_dev=2)
    got = bbands(sample_bars)
    np.testing.assert_allclose(
        got["upper"].to_numpy()[19:], reference.bollinger_hband().to_numpy()[19:], rtol=1e-9
    )
    np.testing.assert_allclose(
        got["lower"].to_numpy()[19:], reference.bollinger_lband().to_numpy()[19:], rtol=1e-9
    )


def test_donchian_matches_reference(sample_bars):
    from ta.volatility import DonchianChannel

    reference = DonchianChannel(
        high=mid(sample_bars, "high"),
        low=mid(sample_bars, "low"),
        close=mid(sample_bars, "close"),
        window=20,
    )
    got = donchian(sample_bars)
    np.testing.assert_allclose(
        got["upper"].to_numpy()[19:],
        reference.donchian_channel_hband().to_numpy()[19:],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        got["lower"].to_numpy()[19:],
        reference.donchian_channel_lband().to_numpy()[19:],
        rtol=1e-9,
    )


def test_rsi_bounds(sample_bars):
    values = rsi(sample_bars).dropna()
    assert values.between(0, 100).all()


def test_bbands_ordering(sample_bars):
    got = bbands(sample_bars).dropna()
    assert (got["lower"] <= got["middle"]).all()
    assert (got["middle"] <= got["upper"]).all()


def test_donchian_ordering(sample_bars):
    got = donchian(sample_bars).dropna()
    assert (got["lower"] <= got["upper"]).all()
    assert (got["width"] >= 0).all()


# --- E: RSI の分母がゼロのとき ---
#
# 自前実装は当初 `avg_loss.replace(0.0, np.nan)` で avg_loss==0 を NaN に
# 逃がしていたため、下落が1本も無い区間では RSI が NaN になっていた。
# ta.momentum.RSIIndicator のソース(_run)を読むと
#   np.where(emadn == 0, 100, 100 - 100/(1+rs))
# となっており、「下落ゼロなら100」が標準（Wilderの定義どおり、
# RS=avg_gain/avg_loss が発散するので上限の100に張り付く）。
# core.py の rsi() はこの規約に修正済み。ここではその挙動を固定する。


def _trending_bars(direction: str, n: int = 30) -> pd.DataFrame:
    """先頭から一貫して上昇 or 下降するバー（bid/askの列を持つ）。"""
    step = 0.01 if direction == "up" else -0.01
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    close = 150.0 + np.arange(n) * step
    body: dict[str, object] = {"close_time": open_time + pd.Timedelta(minutes=1)}
    body["bid_close"] = close
    body["bid_open"] = np.concatenate([[close[0]], close[:-1]])
    body["bid_high"] = np.maximum(body["bid_open"], body["bid_close"]) + 0.03
    body["bid_low"] = np.minimum(body["bid_open"], body["bid_close"]) - 0.03
    for f in ("open", "high", "low", "close"):
        body[f"ask_{f}"] = body[f"bid_{f}"] + 0.02
    body["volume"] = [100.0] * n
    return pd.DataFrame(body, index=open_time).rename_axis("open_time")


def test_rsi_is_100_when_no_losses_in_window():
    """一本も下落が無い区間（avg_loss==0）では RSI は上限の100。NaNではない。"""
    bars = _trending_bars("up", n=30)
    values = rsi(bars, period=14).dropna()
    assert len(values) > 0
    assert (values == 100.0).all()
    assert not values.isna().any()


def test_rsi_is_0_when_no_gains_in_window():
    """対称のケース: 一本も上昇が無い区間では avg_gain==0 なので RS=0、RSI=0。
    この側は元々の式（ゼロ除算にならない）でも正しく動く。"""
    bars = _trending_bars("down", n=30)
    values = rsi(bars, period=14).dropna()
    assert len(values) > 0
    assert (values == 0.0).all()


def test_rsi_matches_reference_with_no_losses():
    """ta も同じ規約（avg_loss==0 → 100）であることを、単調増加区間で確認する。

    スライスは[14:]（[13:]ではない）。ta は close.diff() 先頭のNaNを
    「変化なし(0)」として数えるため、min_periods=14 を自前実装より1本早く
    満たす（position13で早くも値が出る）。自前実装は先頭の未定義diffを
    正しくNaN扱いするので、position13はまだウォームアップ中でNaN。
    先頭のズレなので値そのものには影響しない（G参照）。
    """
    from ta.momentum import RSIIndicator

    bars = _trending_bars("up", n=30)
    close = mid(bars, "close")
    expected = RSIIndicator(close, window=14).rsi()
    np.testing.assert_allclose(
        rsi(bars, period=14).to_numpy()[14:], expected.to_numpy()[14:], rtol=1e-9
    )


# --- D: vwap の日次リセット ---
#
# sample_bars(300分, NY取引日1日分)だけでは groupby が1グループにしかならず、
# 日をまたいだリセットが一度も検証されない。multi_day_bars で確認する。


def test_vwap_resets_at_ny_trading_day_boundary(multi_day_bars):
    boundary = pd.Timestamp("2026-01-05 22:00:00", tz="UTC")
    pos = multi_day_bars.index.get_loc(boundary)
    assert pos > 0, "境界がフィクスチャの先頭に来ている(前提が崩れている)"

    typical = (
        mid(multi_day_bars, "high") + mid(multi_day_bars, "low") + mid(multi_day_bars, "close")
    ) / 3.0
    v = vwap(multi_day_bars)

    # 日2の最初の1本は、その1本だけのtypical price(累積がまだ無い)と一致する
    assert v.iloc[pos] == pytest.approx(typical.iloc[pos])
    # 日1最後の累積値をそのまま引きずっていない(単純な前日引き継ぎではない)
    assert v.iloc[pos] != pytest.approx(v.iloc[pos - 1])


def test_vwap_stays_within_days_price_range(multi_day_bars):
    """累積VWAPは、その日にこれまで出た価格レンジの中に収まるはず。"""
    from aitrading.timeutil import trading_day_start

    typical = (
        mid(multi_day_bars, "high") + mid(multi_day_bars, "low") + mid(multi_day_bars, "close")
    ) / 3.0
    day = pd.Series(
        trading_day_start(pd.DatetimeIndex(multi_day_bars.index), "ny"),
        index=multi_day_bars.index,
    )
    v = vwap(multi_day_bars)
    running_min = typical.groupby(day).cummin()
    running_max = typical.groupby(day).cummax()
    assert (v >= running_min - 1e-9).all()
    assert (v <= running_max + 1e-9).all()


# --- hist_vol: ta に直接の同等物が無いので、素朴なnumpy実装と照合する ---


def test_hist_vol_matches_manual_calculation(sample_bars):
    close = mid(sample_bars, "close")
    manual = np.log(close).diff().rolling(60, min_periods=60).std(ddof=0)
    np.testing.assert_allclose(
        hist_vol(sample_bars).to_numpy()[59:], manual.to_numpy()[59:], rtol=1e-12
    )


def test_hist_vol_is_non_negative(sample_bars):
    values = hist_vol(sample_bars).dropna()
    assert len(values) > 0
    assert (values >= 0).all()


def test_hist_vol_is_zero_for_constant_price():
    n = 80
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    body: dict[str, object] = {"close_time": open_time + pd.Timedelta(minutes=1)}
    for f in ("open", "high", "low", "close"):
        body[f"bid_{f}"] = [150.0] * n
        body[f"ask_{f}"] = [150.02] * n
    body["volume"] = [100.0] * n
    bars = pd.DataFrame(body, index=open_time).rename_axis("open_time")

    values = hist_vol(bars).dropna()
    assert len(values) > 0
    np.testing.assert_allclose(values.to_numpy(), 0.0, atol=1e-12)
