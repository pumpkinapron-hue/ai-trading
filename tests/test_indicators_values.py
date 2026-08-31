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
    # 数値が合わないことを実測で確認した。200本目以降なら十分に減衰する。
    # rtol は 1e-4。1e-6 では余裕が2.3倍しかなく、乱数系列を変えると実際に落ちる
    # （seed=3 で 1.188e-06）。ta との差は「ウォームアップの種付け流儀の違いが
    # 指数的に減衰しきる速さ」で決まる綱渡りなので、ここは同一指標であることの
    # 確認に徹し、式そのものは
    # test_rsi_matches_a_plain_loop_wilder_implementation（rtol=1e-12）で固定する。
    # 式の取り違え（単純平均にする / gain と loss を逆にする）は 1e-4 でも余裕で落ちる。
    np.testing.assert_allclose(
        rsi(sample_bars).to_numpy()[200:], expected.to_numpy()[200:], rtol=1e-4
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


# ============================================================================
# レビュー(task-9-review.md)で見つかった検証穴を埋める
# ============================================================================


def _bars_from(closes, highs=None, lows=None, volumes=None, spread=0.02):
    """明示的な値からバーを作る。合成ランダムウォークでは作れない状況を試すため。"""
    n = len(closes)
    index = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    highs = [c + 0.05 for c in closes] if highs is None else highs
    lows = [c - 0.05 for c in closes] if lows is None else lows
    body = {
        "close_time": index + pd.Timedelta(minutes=1),
        "bid_open": list(closes),
        "bid_high": list(highs),
        "bid_low": list(lows),
        "bid_close": list(closes),
        "volume": [100.0] * n if volumes is None else list(volumes),
    }
    for field in ("open", "high", "low", "close"):
        body[f"ask_{field}"] = [v + spread for v in body[f"bid_{field}"]]
    return pd.DataFrame(body, index=index).rename_axis("open_time")


def test_mid_is_the_average_of_bid_and_ask():
    """`mid()` の規約（ビッドとアスクの中間値）が一度も検証されていなかった。

    参照値照合は `ta` にも `mid()` の出力を渡しているので、`mid` が
    ビッドだけを返すようになっても両辺が同じだけずれて打ち消し合う。
    """
    bars = _bars_from([150.0, 151.0], spread=0.04)
    assert mid(bars, "close").tolist() == pytest.approx([150.02, 151.02])
    assert mid(bars, "high").tolist() == pytest.approx([150.07, 151.07])


def test_atr_true_range_uses_the_previous_close():
    """ATR の肝は「前バーの終値を含む窓」。窓に前終値が入っていないと、
    True Range が単なる Range（high-low）に退化する。

    conftest のフィクスチャは `open == 前バーの close` の連続系列なので、
    価格が飛ばず、この違いが一度も現れない（`prev_close` の項を丸ごと
    外しても全テストが通ってしまう）。ここでは窓を開けたバーを直接作る。
    """
    # 2本目で大きく窓を開ける。high-low は常に0.10だが、前終値との差は1.00
    closes = [150.0, 151.0] + [151.0] * 30
    bars = _bars_from(closes)

    got = atr(bars, period=2)
    # 前終値を含めなければ True Range は 0.10 のままで、ATRが1.0近くまで
    # 跳ね上がることはない
    assert got.iloc[1] > 0.4, "窓開けが True Range に反映されていない"
    # 窓開けの影響は指数的に減衰し、最終的に high-low の水準へ戻る
    assert got.iloc[-1] == pytest.approx(0.10, abs=0.01)


def test_vwap_is_weighted_by_volume():
    """VWAP の「Volume-Weighted」の部分が無検証だった。

    出来高で重み付けしていなければ、typical price の単純平均になる。
    """
    bars = _bars_from([150.0, 160.0], highs=[150.0, 160.0], lows=[150.0, 160.0],
                      volumes=[1.0, 9.0], spread=0.0)
    got = vwap(bars)
    assert got.iloc[0] == pytest.approx(150.0)
    # 出来高加重なら (150*1 + 160*9)/10 = 159.0。単純平均なら 155.0
    assert got.iloc[1] == pytest.approx(159.0)


def test_macd_histogram_is_line_minus_signal(sample_bars):
    """`histogram` 列が値として無検証だった（符号を逆にしても通っていた）。"""
    got = macd(sample_bars)
    pd.testing.assert_series_equal(
        got["histogram"], (got["macd"] - got["signal"]).rename("histogram")
    )


def test_bbands_num_std_scales_the_band_width(sample_bars):
    """`num_std` が実際に幅へ効いていることを固定する。"""
    one = bbands(sample_bars, num_std=1.0).dropna()
    two = bbands(sample_bars, num_std=2.0).dropna()
    pd.testing.assert_series_equal(one["middle"], two["middle"])
    np.testing.assert_allclose(
        (two["upper"] - two["middle"]).to_numpy(),
        (one["upper"] - one["middle"]).to_numpy() * 2.0,
        rtol=1e-11,
    )


@pytest.mark.parametrize(
    ("fn", "period", "warmup"),
    [(sma, 20, 19), (bbands, 20, 19), (rsi, 14, 13), (atr, 14, 13), (hist_vol, 60, 60)],
)
def test_warmup_is_not_shortened(fn, period, warmup):
    """`min_periods` を1に落とすと、窓が埋まる前から値が出てしまう。

    「まだ計算できない」を正直に NaN で返すことがウォームアップの意味で、
    ここが緩むと、実質もっと短い窓の指標を「20本移動平均」と呼ぶことになる。
    """
    got = fn(_bars_from([150.0 + i * 0.01 for i in range(200)]), period=period)
    column = got if isinstance(got, pd.Series) else got["middle" if fn is bbands else got.columns[0]]
    assert column.iloc[: warmup - 1].isna().all(), "ウォームアップ前に値が出ている"
    assert column.iloc[warmup:].notna().any(), "ウォームアップ後も値が出ていない"


def test_rsi_matches_a_plain_loop_wilder_implementation():
    """`ta` との照合とは別に、素のループで書いた Wilder 実装と厳密に突き合わせる。

    `ta` との比較は「種付けの違いが減衰しきるのを待つ」形なので、閾値・減衰率・
    系列長の綱渡りになっており、乱数系列によっては閾値を割る。こちらは
    綱渡りが無く、`ta` の版が上がっても壊れない。
    """
    rng = np.random.default_rng(7)
    closes = (150.0 + np.cumsum(rng.normal(0, 0.05, 300))).tolist()
    bars = _bars_from(closes)
    period = 14

    prices = mid(bars, "close").to_numpy()
    gains = np.diff(prices, prepend=np.nan).clip(min=0.0)
    losses = (-np.diff(prices, prepend=np.nan)).clip(min=0.0)
    avg_gain = np.full(len(prices), np.nan)
    avg_loss = np.full(len(prices), np.nan)
    alpha = 1.0 / period
    # 先頭は diff で NaN。pandas の ewm(adjust=False) は先頭のNaNを飛ばし、
    # 最初の有効値をそのまま種にする（0から積み始めるのではない）。
    avg_gain[1], avg_loss[1] = gains[1], losses[1]
    for i in range(2, len(prices)):
        avg_gain[i] = avg_gain[i - 1] + alpha * (gains[i] - avg_gain[i - 1])
        avg_loss[i] = avg_loss[i - 1] + alpha * (losses[i] - avg_loss[i - 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.where(
            avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        )

    np.testing.assert_allclose(
        rsi(bars, period=period).to_numpy()[period:], expected[period:], rtol=1e-12
    )
