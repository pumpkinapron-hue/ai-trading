"""チャート・データ品質・期待値スキャンの確認画面。

判断・描画ロジックを純関数として切り出し、Streamlit のUI呼び出し（`main()` と
その下請けの `_render_*` 関数）とは分けてある。前者は pytest から直接呼べるが、
後者は事実上できない（`st.*` の一部はStreamlitのスクリプト実行コンテキストが
無いと正しく動かない）。テストは前者だけを直接叩く。

このダッシュボードは、人間がAIトレード研究の各段階を目で確認するための窓であって
それ自体は判断をしない。特に「期待値スキャン」タブは、選んだ集計期間が
`config/settings.toml` でロックされていれば、理由の明示と `meta.db` への記録
なしには何も集計しない――一度OOSの結果を人間が見てしまったら、そのOOSはもう
OOSではない。この拒否は `aitrading.edge_scan.scan_period` が実装しているので、
ここでは `scan_period` だけを経由し、`scan()` を直接呼ばないことを徹底する
（ロック判定をここで独自に再実装すると、2箇所の判定が食い違う壊れ方をしうる）。

`aitrading` パッケージの import に `sys.path` 操作は不要（`uv sync` の
editable install で解決できる。`scripts/fetch_data.py` と同じ理由・同じ規約）。
"""

from __future__ import annotations

from dataclasses import asdict, fields

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aitrading import quality
from aitrading.config import Settings, load_settings
from aitrading.edge_scan import DEFAULT_HORIZONS, EdgeResult, scan_period
from aitrading.indicators import INDICATORS
from aitrading.indicators.registry import mid
from aitrading.quality import QualityReport
from aitrading.storage.lake import Lake
from aitrading.storage.meta import Meta
from aitrading.timeutil import Session, Timeframe, session_labels

#: チャートに渡す最大本数。Plotlyのローソク足に数十万〜数百万本を渡すとブラウザの
#: 描画がメインスレッドをブロックして固まる。1500本あれば1分足でも1日分強、上位足
#: なら数か月〜数年分に相当し、目視でのチャート確認には十分な密度。
MAX_CHART_BARS = 1500

#: セッション帯の背景色（低アルファ）。OFF（どの主要市場も開いていない帯）は塗らない。
_SESSION_COLORS: dict[Session, str] = {
    Session.TOKYO: "rgba(99, 102, 241, 0.10)",
    Session.LONDON: "rgba(16, 185, 129, 0.10)",
    Session.NEWYORK: "rgba(249, 115, 22, 0.10)",
    Session.LDN_NY_OVERLAP: "rgba(236, 72, 153, 0.12)",
}


# ============================================================================
# 純関数 —— pytest から直接呼べる。main() 内の判断はすべてここに出してある。
# ============================================================================


def limit_for_chart(bars: pd.DataFrame, max_bars: int = MAX_CHART_BARS) -> pd.DataFrame:
    """Plotly へ渡す本数を上限で切る。直近側（末尾）を残す。"""
    if len(bars) <= max_bars:
        return bars
    return bars.tail(max_bars)


def _add_session_bands(fig: go.Figure, bars: pd.DataFrame) -> None:
    """セッション帯を背景の縦帯として塗る。

    1本ごとに矩形を足すと足の本数だけ形状が増えて重くなるので、連続して同じ
    セッションが続く区間をまとめて1本の帯にする。帯の終端には（足の長さを仮定
    せず）実際の `close_time` を使う――日足はDSTで23/25時間になり得るため、
    固定長を仮定すると帯の境界がずれる。
    """
    if bars.empty:
        return
    labels = session_labels(pd.DatetimeIndex(bars.index))
    run_id = labels.ne(labels.shift()).cumsum()
    for _, run in labels.groupby(run_id):
        color = _SESSION_COLORS.get(run.iloc[0])
        if color is None:
            continue
        end = bars.loc[run.index[-1], "close_time"]
        fig.add_vrect(x0=run.index[0], x1=end, fillcolor=color, line_width=0, layer="below")


def candlestick_figure(
    bars: pd.DataFrame,
    overlays: dict[str, pd.Series] | None = None,
    trades: pd.DataFrame | None = None,
) -> go.Figure:
    """ローソク足＋セッション帯＋指標オーバーレイ＋トレードマーカー。

    `trades` は仕様書§17（チャート監査）の口。Phase 0 では常に `None` だが、
    後のフェーズでトレードログをそのまま渡せる形にしてある。

    `bars`/`overlays` の値は読むだけで書き換えない。呼び出し側（`main()`）は
    `st.cache_data` でキャッシュした結果をそのまま渡すことがあるため、ここで
    列を追加するような形の書き換えをしてしまうと、同じキャッシュ済みフレームを
    使う他の箇所（他のタブ・次の再実行）まで巻き込む。
    """
    fig = go.Figure()
    if bars.empty:
        return fig

    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=mid(bars, "open"),
            high=mid(bars, "high"),
            low=mid(bars, "low"),
            close=mid(bars, "close"),
            name="USDJPY",
        )
    )
    _add_session_bands(fig, bars)

    for name, series in (overlays or {}).items():
        fig.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name))

    if trades is not None and not trades.empty:
        fig.add_trace(
            go.Scatter(
                x=trades["entry_time"], y=trades["entry_price"],
                mode="markers", name="entry",
                marker=dict(symbol="triangle-up", size=12),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=trades["exit_time"], y=trades["exit_price"],
                mode="markers", name="exit",
                marker=dict(symbol="x", size=10),
            )
        )
        for _, row in trades.iterrows():
            for level, dash in (("sl", "dot"), ("tp", "dash")):
                if pd.notna(row.get(level)):
                    fig.add_shape(
                        type="line", x0=row["entry_time"], x1=row["exit_time"],
                        y0=row[level], y1=row[level], line=dict(dash=dash, width=1),
                    )

    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    return fig


#: 増えても既存の meta.db を読めなくしない（欠けていても "summary" 扱いにする）フィールド。
_OPTIONAL_REPORT_FIELDS: frozenset[str] = frozenset()


def quality_view(report: dict | None) -> dict:
    """`Meta.latest_quality()` が返す1レコードを表示用に分類する。

    正規の品質サマリ（`QualityReport.to_dict()`）と隔離レコード
    （`{"status": "quarantined", "chunk_start", "chunk_end", "bar_count", "error"}`、
    `scripts/fetch_data.py::_quarantine` が書く）はキーの形がまったく違う。
    全チャンクが隔離されると最終サマリが一度も書かれないため、`latest_quality()`
    は隔離レコードを返すことがある――`report["actual_bars"]` を無条件に読むと
    ここで `KeyError` になる（実測済みの不具合）。`status` を目印に種類を判定し、
    呼び出し側が分岐できる形にする。

    戻り値の `kind` は `"none"` / `"quarantined"` / `"summary"` / `"raw"` のいずれか。

    `QualityReport(**report)` と素で流し込んではいけない。`meta.db` はコードより
    長生きする永続ストアで、`QualityReport` は実際に一度フィールドが増えている
    （`conflicting_duplicate_count` / `wide_spread_threshold`）。増減した瞬間に
    `TypeError` でダッシュボードが落ちる――「Task 11 がレコードの形を変えて
    Task 12 が KeyError で落ちた」のと同じことが、キー名ではなく引数の形で起きる。
    知らないキーは捨て、足りないキーがあれば素の dict として見せて落とさない。
    """
    if report is None:
        return {"kind": "none"}
    if report.get("status") == "quarantined":
        return {"kind": "quarantined", "record": report}
    known = {f.name for f in fields(QualityReport)}
    unknown = sorted(set(report) - known)
    missing = sorted(known - set(report) - _OPTIONAL_REPORT_FIELDS)
    if missing:
        # 古い meta.db を新しいコードで読んだ場合。落とさずに素の dict として見せる。
        return {"kind": "raw", "record": report, "missing": missing}
    return {
        "kind": "summary",
        "record": QualityReport(**{k: v for k, v in report.items() if k in known}),
        "unknown": unknown,
    }


def quarantine_records(history: list[dict]) -> list[dict]:
    """品質レポートの履歴（`Meta.quality_history()`）から隔離レコードだけを残す。

    `latest_quality()`（＝`quality_view` が分類する対象）は最新1件しか見ないため、
    一部のチャンクだけ隔離された取得では、その後に記録される正規の最終サマリに
    隠れて「隔離が起きたこと」自体が見えなくなる。履歴全体をこちらで別に見せる。
    """
    return [r for r in history if r.get("status") == "quarantined"]


def indicator_result_columns(result: pd.Series | pd.DataFrame) -> list[str]:
    """指標の出力が持つ列名の一覧。`Series` なら1つだけ（指標名そのもの）。"""
    if isinstance(result, pd.DataFrame):
        return list(result.columns)
    return [result.name or "value"]


def signal_from_indicator(
    result: pd.Series | pd.DataFrame, column: str | None, op: str, threshold: float
) -> pd.Series:
    """指標の出力としきい値を比較して bool の signal を作る。

    `DataFrame` を返す指標（`macd`/`bbands`/`donchian` など）は `column` の
    指定が必要。`Series` を返す指標（`rsi`/`sma` など）は `column` を無視する。
    """
    if isinstance(result, pd.DataFrame):
        if column is None:
            raise ValueError("複数列を返す指標では column の指定が必要")
        series = result[column]
    else:
        series = result
    if op == "<":
        return series < threshold
    if op == ">":
        return series > threshold
    raise ValueError(f"未知の演算子: {op!r}（'<' か '>'）")


def scan_indicator_condition(
    bars: pd.DataFrame,
    settings: Settings,
    meta: Meta,
    *,
    indicator_name: str,
    column: str | None,
    op: str,
    threshold: float,
    period: str,
    unlock_reason: str | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    direction: str = "long",
    deduct_spread: bool = True,
    group_by: str | None = None,
) -> EdgeResult:
    """指標の条件からシグナルを作り、指定した期間で期待値スキャンする。

    **`scan_period` だけを経由する（`scan` を直接呼ばない）。** ロックされた
    期間は `scan_period` 自身の拒否にそのまま乗せる――`unlock_reason` が無ければ
    `PermissionError`、`unlock_reason` はあるが `meta` が設定と違う場所を指して
    いれば `ValueError`。ロック判定をここで独自に再実装しないのは、判定が2箇所に
    分かれると「片方だけ直す／食い違う」という壊れ方をしうるため
    （`edge_scan.scan_period` のdocstring参照）。
    """
    if indicator_name not in INDICATORS:
        raise ValueError(f"未知の指標: {indicator_name!r}（{sorted(INDICATORS)}）")
    result = INDICATORS[indicator_name](bars)
    signal = signal_from_indicator(result, column, op, threshold)
    return scan_period(
        bars,
        signal,
        settings,
        period,
        meta=meta,
        unlock_reason=unlock_reason,
        horizons=horizons,
        direction=direction,
        deduct_spread=deduct_spread,
        group_by=group_by,
    )


def edge_result_table(result: EdgeResult) -> pd.DataFrame:
    """`EdgeResult.horizons` を表示用のフラットな表にする（horizonごとに1行）。"""
    return pd.DataFrame([asdict(h) for h in result.horizons])


def edge_result_group_table(result: EdgeResult) -> pd.DataFrame:
    """`EdgeResult.by_group`（セッション別集計）を1つの表にする。

    `group_by="session"` を指定しなかった集計では `by_group` が空なので、
    その場合は空の（0行の）DataFrameを返す。
    """
    rows = [
        {"session": label, **asdict(stats)}
        for label, stats_list in result.by_group.items()
        for stats in stats_list
    ]
    return pd.DataFrame(rows)


def _load_bars(lake: Lake, symbol: str, timeframe: Timeframe, as_of: pd.Timestamp) -> pd.DataFrame:
    """`Lake.load` の薄いラッパー。

    `main()` はこれを直接使わず、下の `_load_bars_cached`（`st.cache_data` で
    包んだもの）を使う。ここを独立させてあるのは、テストが「キャッシュされて
    いない生の関数」を直接呼べるようにするため――`st.cache_data` を付けた
    関数はStreamlitのスクリプト実行コンテキストが無い場所（pytest）から呼んでも
    例外にはならないが、"missing ScriptRunContext" 系の警告ログを出す。判断
    ロジックそのものはこの関数には無い（`Lake.load` への素通し）ので、キャッシュ
    有り無しでテスト結果が変わることもない。
    """
    return lake.load(symbol, timeframe, as_of=as_of)


# ============================================================================
# Streamlit UI —— main() 自体はテストしない。判断はすべて上の純関数に出してある。
# ============================================================================


@st.cache_data(ttl="10m", max_entries=8, show_spinner="バーを読み込み中…")
def _load_bars_cached(
    _lake: Lake, symbol: str, timeframe_value: str, as_of: pd.Timestamp
) -> pd.DataFrame:
    """`_load_bars` のキャッシュ版。`main()` はこちらを使う。

    370万本超ある1分足を `lake.load()` で読み直すコストは軽くない。Streamlitは
    ウィジェットを触るたびにスクリプト全体を再実行するため、キャッシュ無しだと
    「指標のチェックボックスを1つ切り替えるだけ」でも毎回ディスクから読み直す
    ことになる。

    引数名の先頭アンダースコア（`_lake`）は「この引数はキャッシュキーに使わない
    （ハッシュ化しない）」というStreamlitの規約。`Lake` は `root: Path` しか
    持たない薄いラッパーで、同じセッション内で値が変わることも無いので、キャッシュ
    キーに含める必要が無い（含めようとすると `Lake` をハッシュ可能にする一手間が
    要るだけで、キャッシュの正しさには寄与しない）。`symbol`/`timeframe_value`
    （`Timeframe` ではなく `.value` の文字列で渡す——`Timeframe` 自体も
    ハッシュ可能だが、キャッシュキーは単純な組み込み型に保つ）/`as_of` の組が
    実際のキャッシュキーになる。

    `ttl="10m"` と `max_entries=8` は両方効かせる。データそのものは
    `scripts/fetch_data.py`/`scripts/build_bars.py` を実行しない限り変わらない
    ので鮮度はさほど気にしなくてよいが、ユーザーが `as_of` をあれこれ変えるたびに
    キャッシュエントリが積み上がるので、上限（`max_entries`）で頭打ちにする。
    `ttl` は、外部でデータが更新されたのに古いキャッシュを延々と見続ける事態を
    避けるための保険。

    ここで返す `DataFrame` を呼び出し側が書き換えないこと。`st.cache_data` は
    呼び出しのたびにコピーを返すので、このプロセス内で書き換えても他の呼び出しの
    結果までは壊れない（実測確認済み）が、それはStreamlitの実装詳細であって
    このモジュールの契約ではない――`candlestick_figure` 側にも書き換えない
    という契約を明記してあるのはそのため。
    """
    return _load_bars(_lake, symbol, Timeframe(timeframe_value), as_of)


def main() -> None:
    st.set_page_config(page_title="ai-trading", layout="wide")

    settings = load_settings()
    lake = Lake(settings.data_root)
    meta = Meta(settings.meta_db)

    st.sidebar.header("表示設定")
    timeframe = Timeframe(
        st.sidebar.selectbox("時間軸", [t.value for t in Timeframe], index=0)
    )
    today = pd.Timestamp.now(tz="UTC").date()
    as_of_date = st.sidebar.date_input(
        "as_of（この日の終わりより後は表示しない）",
        value=today,
        min_value=settings.data_start.date(),
        max_value=today,
    )
    # 選んだ暦日の終わりまでを含める。先読み防止の要は Lake.load 側の
    # close_time <= as_of だが、ここで「その日を含む」よう1日進めておかないと
    # 選んだ当日のバーが1本も表示されない。
    as_of = pd.Timestamp(as_of_date, tz="UTC") + pd.Timedelta(days=1)
    chosen_indicators = st.sidebar.multiselect("指標（チャートに重ねる）", sorted(INDICATORS))

    bars_full = _load_bars_cached(lake, settings.symbol, timeframe.value, as_of)
    bars = limit_for_chart(bars_full)

    chart_tab, quality_tab, edge_tab = st.tabs(["チャート", "データ品質", "期待値スキャン"])

    with chart_tab:
        _render_chart_tab(bars, bars_full, chosen_indicators)

    with quality_tab:
        _render_quality_tab(meta, settings.symbol, timeframe)

    with edge_tab:
        _render_edge_tab(bars_full, settings, meta)


def _render_chart_tab(
    bars: pd.DataFrame, bars_full: pd.DataFrame, chosen_indicators: list[str]
) -> None:
    if bars_full.empty:
        st.warning("データが無い。scripts/fetch_data.py を実行すること")
        return
    if len(bars) < len(bars_full):
        st.caption(f"表示は直近 {len(bars):,} 本に制限（全 {len(bars_full):,} 本中）")

    # **指標は必ず全系列 `bars_full` にかけてから、表示窓へ切り出す。**
    # 表示窓（末尾1500本）に直接かけると、先頭を切り落としたぶん値が変わる。
    # これは Task 9 のレビューが確立した事実そのもの――「入力の先頭を切り落とす
    # 検査は入れてはいけない。正しい9指標が全部落ちる（偽陽性100%）」。
    # ewm 系は原理的に全履歴に依存し、rolling 系もウォームアップの位置がずれる。
    # 実測（実データ7169本→表示1500本）では RSI で最大 8.26 ポイントの差が出た。
    # 放置すると「チャートで見た RSI」と「スキャンが使った RSI」が別物になる。
    overlays: dict[str, pd.Series] = {}
    for name in chosen_indicators:
        result = INDICATORS[name](bars_full)
        if isinstance(result, pd.DataFrame):
            overlays.update({f"{name}.{c}": result.loc[bars.index, c] for c in result.columns})
        else:
            overlays[name] = result.loc[bars.index]
    st.plotly_chart(candlestick_figure(bars, overlays), width="stretch")


def _render_quality_tab(meta: Meta, symbol: str, timeframe: Timeframe) -> None:
    report = meta.latest_quality(symbol, timeframe)
    view = quality_view(report)
    if view["kind"] == "none":
        st.info("品質レポートがまだ無い。scripts/fetch_data.py を実行すること")
    elif view["kind"] == "quarantined":
        q = view["record"]
        st.warning(
            "直近の記録は隔離のみ（正規の品質サマリはまだ無い）。"
            f" {q['chunk_start']} 〜 {q['chunk_end']}: {q['error']}"
        )
        if q.get("bar_count") is not None:
            st.caption(f"隔離時点で手元にあったバー数: {q['bar_count']}")
    else:
        qr: QualityReport = view["record"]
        cols = st.columns(3)
        cols[0].metric("取得本数 / 期待本数", f"{qr.actual_bars} / {qr.expected_bars}")
        cols[1].metric(
            "重複（うち値の食い違い）",
            f"{qr.duplicate_count}（{qr.conflicting_duplicate_count}）",
        )
        cols[2].metric("価格ジャンプ", qr.price_jump_count)
        st.caption(quality.format_summary(qr))
        if qr.gaps:
            st.dataframe(pd.DataFrame(qr.gaps), width="stretch", hide_index=True)

    history = meta.quality_history(symbol, timeframe)
    quarantined = quarantine_records(history)
    if quarantined:
        st.warning(f"これまでに隔離されたチャンクが {len(quarantined)} 件ある")
        st.dataframe(pd.DataFrame(quarantined), width="stretch", hide_index=True)


def _render_edge_tab(bars_full: pd.DataFrame, settings: Settings, meta: Meta) -> None:
    st.caption(
        "条件を選んで実行すると、その条件で期待値スキャン（scan_period）を実行する。"
        " 集計期間の一部はロックされている――ロックされた期間は解除理由を明示しないと"
        " 集計できず、解除は meta.db に記録される。一度見たOOSはもうOOSではない。"
    )
    if bars_full.empty:
        st.warning("データが無い。scripts/fetch_data.py を実行すること")
        return

    indicator_name = st.selectbox("指標", sorted(INDICATORS), key="edge_indicator")
    result = INDICATORS[indicator_name](bars_full)
    column: str | None = None
    if isinstance(result, pd.DataFrame):
        column = st.selectbox("列", indicator_result_columns(result), key="edge_column")

    left, right = st.columns(2)
    op = left.segmented_control("条件", ["<", ">"], default="<", key="edge_op")
    threshold = right.number_input("しきい値", value=30.0, key="edge_threshold")

    direction = st.segmented_control(
        "方向", ["long", "short"], default="long", key="edge_direction"
    )
    horizon_options = [str(h) for h in DEFAULT_HORIZONS]
    chosen_horizons = st.pills(
        "horizon（シグナル後 何本目のリターンを見るか）",
        horizon_options,
        selection_mode="multi",
        default=horizon_options,
        key="edge_horizons",
    )
    deduct_spread = st.checkbox("スプレッド控除後で評価する", value=True, key="edge_deduct_spread")
    group_by_session = st.checkbox(
        "セッション別に層別する", value=False, key="edge_group_by_session"
    )

    period_names = list(settings.periods)
    period = st.selectbox(
        "集計期間",
        period_names,
        format_func=lambda name: name + ("（ロック中）" if settings.periods[name].locked else ""),
        key="edge_period",
    )
    unlock_reason: str | None = None
    if settings.period_for(period).locked:
        st.warning(
            f"「{period}」はロックされている。集計するには解除理由が必要"
            "（meta.db に記録される）。"
        )
        unlock_reason = st.text_input("解除理由（必須）", key="edge_unlock_reason") or None

    if not st.button("期待値スキャンを実行", key="edge_run"):
        return
    if not op or not chosen_horizons:
        st.error("条件と horizon を選ぶこと")
        return

    try:
        edge = scan_indicator_condition(
            bars_full,
            settings,
            meta,
            indicator_name=indicator_name,
            column=column,
            op=op,
            threshold=float(threshold),
            period=period,
            unlock_reason=unlock_reason,
            horizons=tuple(int(h) for h in chosen_horizons),
            direction=direction,
            deduct_spread=deduct_spread,
            group_by="session" if group_by_session else None,
        )
    except (PermissionError, ValueError) as exc:
        st.error(str(exc))
        return

    st.metric("シグナル本数", edge.n_signals)
    st.caption(f"集計期間: {edge.covered_start} 〜 {edge.covered_end}")
    st.dataframe(edge_result_table(edge), width="stretch", hide_index=True)
    if group_by_session:
        st.dataframe(edge_result_group_table(edge), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
