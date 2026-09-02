"""期待値スキャン。バックテストエンジンではない。

TP/SL もポジション管理も資金管理も持たない。「そもそもこの条件に優位性の種が
あるか」を見るための純粋な統計。バックテストエンジンを作る前に使う。

**このツールの唯一の目的は「その優位性は本物か、それともノイズか」を判定すること。**
判定を誤らせる要素は、値が出ていても致命的として扱う（下記 `_effective_n` 参照）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from pathlib import Path

import numpy as np
import pandas as pd

from aitrading.config import Settings
from aitrading.storage.meta import Meta
from aitrading.timeutil import session_labels

DEFAULT_HORIZONS = (1, 5, 15, 30, 60)

#: USD/JPY のクォート規約（1pip = 0.01円）。Phase 0 は USD/JPY のみ対応
#: （`config/settings.toml` の `data.symbol` も単一シンボル）。EUR/USD 等
#: 「/JPY」で終わらないペアは 1pip = 0.0001 なので、このデフォルトのまま
#: 呼ぶと桁が100倍ずれた期待値が出る。他ペアを scan する側が `pip=` を
#: 明示すること。シンボル文字列からの自動判定はここでは行わない
#: ——Phase 0 に他シンボルの取得経路自体が無く、対応表を今ここに書いても
#: 検証しようが無い当て推量になるため（quality.py が祝日カレンダーを
#: 「外部依存の判断なのでPhase 0では入れていない」としているのと同じ理由）。
DEFAULT_PIP_USDJPY = 0.01


@dataclass
class HorizonStats:
    """1つの horizon（シグナル後 何本目か）についての集計。

    `n` と `n_eff` は違うものを数えている。

    - `n`: 実際に集計できたサンプル数（シグナルが出て、かつ horizon 本先の
      バーが存在した回数）。`EdgeResult.n_signals` とも一致しないことがある
      （後述）。
    - `n_eff`: そのうち「独立に近い」とみなせるサンプル数の概算。
      `ci95_pips` はこちらを使って計算する（下記 `_effective_n` 参照）。

    シグナルが毎足連続して出るような条件では、horizon本のリターン窓が
    隣接シグナル同士で大きく重なる（horizon=60 で1本おきにシグナルが
    出れば59本ぶん重複する）。重複したサンプルは互いにほぼ同じ情報しか
    持たないので、`n`（重複込みの生の個数）をそのまま `sqrt(n)` に使うと
    信頼区間の分母を過大評価し、区間が実際よりはるかに狭く出る。
    ランダムウォーク（優位性ゼロと分かっている系列）に毎足シグナルを立てて
    実測したところ、`n` をそのまま使う信頼区間は「本来5%程度であるべき
    誤検出率」が seed 300本中 70〜82% に達した（ただのノイズの大半が
    「有意な優位性」に見えてしまう）。詳細は `_effective_n` のdocstringと
    `task-10-report.md` の検証Aを参照。

    `n_signals`（`EdgeResult`側）と `n` も食い違う。`n_signals` はシグナルが
    立った本数そのもの、`n` はそのうち horizon 本先のバーが実在して実際に
    集計できた本数。長い horizon では終端付近のシグナルが軒並み落ちるため、
    両者の差が horizon が長いほど大きくなる。
    """

    horizon: int
    n: int
    n_eff: float
    mean_pips: float
    median_pips: float
    win_rate: float
    std_pips: float
    ci95_pips: tuple[float, float]


@dataclass
class EdgeResult:
    n_signals: int
    deduct_spread: bool
    direction: str
    covered_start: pd.Timestamp | None = None
    covered_end: pd.Timestamp | None = None
    horizons: list[HorizonStats] = field(default_factory=list)
    by_group: dict[str, list[HorizonStats]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON化のための dict。NaN/Infinity は `None` に変換する。

        `n=0`（集計できたサンプルが無い）ときの `HorizonStats` は
        `mean_pips` 等が NaN になる。Python の `json.dumps` は既定で
        NaN を素通しして非標準の `NaN` トークンを出力する（`allow_nan=True`
        が既定）。Python 同士（`json.loads`も既定でNaNトークンを許容）の
        往復だけなら壊れないが、JSON仕様（RFC 8259）としては不正で、
        strictなパーサ（多くのJS実装の`JSON.parse`を含む）は読めない。
        ダッシュボード（Streamlit）がこの出力をブラウザ側のJSで読む経路が
        将来できてもそのまま壊れないよう、ここで先に `None` へ変換しておく。
        """
        return _json_safe(asdict(self))


def _json_safe(value):
    if isinstance(value, pd.Timestamp):
        # 集計期間は ISO8601 文字列にする。NaN を None にするのと同じ理由で、
        # 標準の json.dumps がそのまま扱える形にしておく（default=str に
        # 頼ると、呼び出し側がそれを忘れた瞬間に落ちる）。
        return value.isoformat()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


#: 両側95%のt分位点（自由度1〜30）。自由度が小さいと正規分位点1.96では全く足りない。
#: scipy を依存に足さないための表。31以上は下の近似で足りる（誤差 0.5% 未満）。
_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}
_Z95 = 1.959964


def _t95(df: float) -> float:
    """両側95%のt分位点。

    正規分位点(1.96)を使ってはいけない。`_effective_n` の性質上、シグナルが
    密に出るケースでは実効サンプル数が必ず一桁になるので、**必ず**t補正が
    要る領域に入る。実測では df=1 で 1.96 の実際の被覆率は約68%、df=2 で約80%、
    df=4 で約87% しかない（名目95%のつもりで読むと、その差がそのまま
    「ノイズを優位性と誤認する率」になる）。
    """
    whole = int(np.floor(df))
    if whole <= 0:
        # ここに来るのは呼び出し側のバグ。黙って NaN を返すと、実効サンプルが
        # 2未満のときのガード（`_stats`）を外しても NaN 区間が出続けて、
        # ガードが効いているのか _t95 が補っているのか区別できなくなる
        # （実際、変異検査で2つの防御が互いを隠していた）。責務を1つに保つ。
        raise ValueError(f"自由度が正でない: {df}（実効サンプル数が2未満）")
    if whole in _T95:
        return _T95[whole]
    # 自由度が大きいところは正規分位点への収束が速い（Cornish-Fisher の1次項）
    return _Z95 * (1.0 + 1.0 / (4.0 * whole))


def _effective_n(mask: np.ndarray, horizon: int) -> float:
    """等重み平均の分散に対応する実効サンプル数。`ci95_pips` の分母に使う。

    `ci95_pips` が表しているのは「集計できた n サンプルの**等重み平均**」の
    区間なので、実効サンプル数もその等重み平均の分散から逆算しなければ
    辻褄が合わない。

    ランダムウォークの horizon 本リターンは、2つの窓が d 本ずれているとき
    相関が `max(0, 1 - d/horizon)`（重なっている割合そのもの）になる。
    したがって等重み平均の分散は

        Var(mean) = (σ² / n²) · ΣᵢΣⱼ max(0, 1 - |tᵢ-tⱼ|/horizon)

    で、これを `σ²/n_eff` と置くと

        n_eff = n² / ΣᵢΣⱼ max(0, 1 - |tᵢ-tⱼ|/horizon)

    となる。三角（Bartlett）カーネルによる実効サンプル数で、Newey-West の
    標準誤差と同じ考え方。

    **クラスタ数を足し合わせる方式（前の実装）はここで壊れる。**
    あちらは「独立な窓がいくつ入るか」をクラスタごとに数えて足していたが、
    等重み平均の重みを見ていない。密ブロック（毎足発火が続く区間）は
    サンプル数が多いので平均への寄与（重み）は大きいのに、持っている情報は
    数個ぶんしかない。そこに疎な単発シグナルを混ぜると、n_eff は両者を
    単純に足して情報量を過大評価する。実測（優位性ゼロのランダムウォーク、
    horizon=60、密ブロック1つ＋疎な単発）で誤検出率が 21〜27% に達した
    （本来5%）。この式なら重み付けが構成上正しいので、その混在でも崩れない。

    両極端では前の実装と一致する（一致すべきところでは一致する）:
    - 毎足連続 `n` 本: 二重和 ≈ n·horizon なので `n_eff ≈ n/horizon`
    - `horizon` 以上離れた疎な発火: 二重和 = n なので `n_eff == n`
    - `horizon` 未満の幅に固まった小バーストが B 個: `n_eff ≈ B`
      （前の実装の「クラスタあたり1個」と同じ。疎なバーストで検出力を
      失わないという前の実装の長所はそのまま保たれる）
    """
    positions = np.flatnonzero(mask).astype(np.int64)
    n = len(positions)
    if n == 0:
        return 0.0

    # ΣᵢΣⱼ max(0, 1 - |tᵢ-tⱼ|/horizon) を prefix sum で O(n log n) に落とす。
    # カーネルの台が有限（|d| < horizon）なので、各 i について寄与する j は
    # 連続した区間 [lo, hi) に限られる。
    lo = np.searchsorted(positions, positions - horizon, side="right")
    hi = np.searchsorted(positions, positions + horizon, side="left")
    prefix = np.concatenate(([0], np.cumsum(positions)))
    here = np.arange(n)

    left_count = here - lo
    right_count = hi - here - 1
    left_sum = prefix[here] - prefix[lo]
    right_sum = prefix[hi] - prefix[here + 1]
    abs_distance = (positions * left_count - left_sum) + (right_sum - positions * right_count)

    double_sum = float(((hi - lo) - abs_distance / horizon).sum())
    return float(n) * n / double_sum


def _stats(returns: pd.Series, horizon: int) -> HorizonStats:
    mask = returns.notna().to_numpy()
    values = returns.to_numpy()[mask]
    n = len(values)
    if n == 0:
        return HorizonStats(
            horizon, 0, 0.0, np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan)
        )

    mean = float(values.mean())
    n_eff = _effective_n(mask, horizon)

    # 実効サンプル数が2未満だと、平均のばらつきを推定する材料が無い。
    # ここで std=0 と置いて幅ゼロの区間を返してはいけない。幅ゼロの区間は
    # 必ず0を除外するので、**単発のシグナル1本が常に「有意」になる**
    # （実測: n=1 の誤検出率 100%）。素直に「算出不能」を返す。
    if n_eff < 2 or n < 2:
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        return HorizonStats(
            horizon=horizon,
            n=n,
            n_eff=n_eff,
            mean_pips=mean,
            median_pips=float(np.median(values)),
            win_rate=float((values > 0).mean()),
            std_pips=std,
            ci95_pips=(np.nan, np.nan),
        )

    std = float(values.std(ddof=1))
    half = _t95(n_eff - 1) * std / np.sqrt(n_eff)
    return HorizonStats(
        horizon=horizon,
        n=n,
        n_eff=n_eff,
        mean_pips=mean,
        median_pips=float(np.median(values)),
        win_rate=float((values > 0).mean()),
        std_pips=std,
        ci95_pips=(mean - half, mean + half),
    )


def _returns(
    bars: pd.DataFrame,
    signal: pd.Series,
    horizon: int,
    direction: str,
    pip: float,
    deduct_spread: bool,
) -> pd.Series:
    """シグナル発生足の確定後にエントリーし、horizon本後に決済した場合のpips。

    deduct_spread=True なら買いはAsk・決済はBidで往復コストが乗る。
    False ならミッド同士で比較する（コスト無視の理論値）。

    **エントリー価格にシグナル発生足自身の close を使う。** 次の足を待たない。
    これは先読みではなく、意図した規約——設計文書 §5 第3層
    （「close_timeが終わった瞬間にデータは利用可能」）をそのまま執行にも
    適用したもの。シグナルは bars[..t] だけから計算されている前提（それを
    保証するのは呼び出し側・指標側の責務で、`test_indicators_lookahead.py`
    の先読み検査がそこを担う）なので、close_time(t) の瞬間には bar t の
    close 価格は既に確定・既知であり、「知りえない情報を使っている」わけ
    ではない。

    ただし「その価格で実際に約定できる」という執行側の仮定（レイテンシ0、
    スリッページ無し）は理想化であり、これは exit 側（`shift(-horizon)`後の
    close）にも同様に適用している——エントリーだけ厳しくして exit は
    甘くする、といった非対称な扱いはしていない。約定モデルそのものは
    設計文書のギャップ分析表7項目6「Phase 1で確定」であり、その約定モデルが
    まだ決まっていない段階で「1本待つ」といった特定の遅延を先取りして
    仮定するのは、根拠のない当て推量を埋め込むことになる。edge_scan は
    そもそもTP/SLもポジション管理も持たない「バックテストエンジンではない」
    道具（モジュールdocstring参照）なので、より厳密な約定シミュレーションは
    後続のバックテストエンジン側の仕事として残す。ここで出るリターンは
    「コスト込みで見て、なお優位性の種があるか」の一次スクリーニングであり、
    実運用でそのまま得られる期待値の確定値ではない。
    """
    if deduct_spread:
        if direction == "long":
            entry, exit_price = bars["ask_close"], bars["bid_close"].shift(-horizon)
        else:
            entry, exit_price = bars["bid_close"], bars["ask_close"].shift(-horizon)
    else:
        mid_price = (bars["bid_close"] + bars["ask_close"]) / 2.0
        entry, exit_price = mid_price, mid_price.shift(-horizon)

    gross = exit_price - entry if direction == "long" else entry - exit_price
    return (gross / pip).where(signal)


def scan(
    bars: pd.DataFrame,
    signal: pd.Series,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    direction: str = "long",
    pip: float = DEFAULT_PIP_USDJPY,
    deduct_spread: bool = True,
    group_by: str | None = None,
) -> EdgeResult:
    """シグナル後のリターン分布を集計する。

    deduct_spread=True が既定。生のリターンで見るとほとんどの条件が有望に見え、
    取引コストを引くと消える。最初からコスト込みで見る。

    `pip` の既定値 `DEFAULT_PIP_USDJPY`（0.01）は USD/JPY 用。他の通貨ペアに
    使う場合は必ず明示すること（`DEFAULT_PIP_USDJPY` 参照）。

    `bars` は時刻昇順（`open_time` 昇順）でソート済みであること。`shift()`は
    行位置ベースで、`_effective_n` のクラスタ分割も行位置の並びが時刻順である
    前提で書いている。`Lake.load()` は常にソート済みで返す（`sort_values`）ので、
    Lake経由のデータをそのまま渡す通常の使い方では問題にならない。
    """
    if direction not in ("long", "short"):
        raise ValueError(f"未知の方向: {direction!r}（'long' か 'short'）")
    if pip <= 0:
        raise ValueError(f"pip は正の値であること: {pip!r}")
    if group_by not in (None, "session"):
        raise ValueError(f"未対応の層別: {group_by!r}（現在は 'session' のみ）")
    if any(h <= 0 for h in horizons):
        raise ValueError(f"horizon は正の整数であること: {horizons!r}")

    # dtype について: `signal` の index が `bars.index` の一部しか持たない
    # （あるいはズレている）と、reindex で新規に生まれる位置は当初 NaN になり、
    # 元が bool dtype でも Series 全体が object dtype に化ける
    # （pandas 3 で実測確認済み。`Series[bool].reindex(...)` は数値NaNの
    # bool版を持たないため）。`.fillna(False)` だけでは dtype は object の
    # ままなので、最後に明示的に `.astype(bool)` で戻す。
    #
    # 変異検査の実測: この `.astype(bool)` を外しても、現行の pandas 3.0.5
    # では `.sum()` / `&`（session側のブールSeriesとの積）/ `.where()` の
    # いずれも、object dtype に生の Python bool が入っている限り bool dtype
    # と同じ結果を返し、どのテストも落ちなかった（-W error でも警告は出ない）。
    # つまり「今の pandas・今のこのモジュールの使い方」では実害は無い。
    # それでも明示キャストを残すのは、object dtype が bool 配列と同じに
    # 振る舞うことは pandas の公開APIとして保証された契約ではなく、
    # 将来この関数に別のブール演算を足したときや pandas のバージョンが
    # 上がったときに暗黙の挙動へ依存し続けたくないため。
    signal = signal.reindex(bars.index).fillna(False).astype(bool)

    # 集計対象の実期間を必ず残す。`scan` は `Settings` を知らないので
    # ロック期間を拒めないが、レポートにこれが出ていれば「気づかないうちに
    # OOS を含めて集計していた」ことが後から分かる（`Lake.load` は全履歴を
    # 返すので、素直に scan へ渡すと OOS がそのまま入る）。
    index = pd.DatetimeIndex(bars.index)
    result = EdgeResult(
        n_signals=int(signal.sum()),
        deduct_spread=deduct_spread,
        direction=direction,
        covered_start=index.min() if len(index) else None,
        covered_end=index.max() if len(index) else None,
    )
    for horizon in horizons:
        returns = _returns(bars, signal, horizon, direction, pip, deduct_spread)
        result.horizons.append(_stats(returns, horizon))

    if group_by == "session":
        # session_labels は Session（str混在のEnum）を値に持つ object dtype の
        # Series を返す。.astype(str) は enum の __str__（"Session.TOKYO"の
        # ような形）ではなく str のペイロード（"TOKYO"）を素通しする
        # （実測確認済み。Session は `class Session(str, Enum)` なので
        # メンバー自体が str インスタンスであり、pandasのastype(str)は
        # __str__ を呼ばず値をそのまま使うため）。読みやすいラベルになる。
        groups = session_labels(pd.DatetimeIndex(bars.index)).astype(str)
        for label in sorted(groups.unique()):
            mask = groups == label
            stats = [
                _stats(
                    _returns(bars, signal & mask, h, direction, pip, deduct_spread), h
                )
                for h in horizons
            ]
            result.by_group[label] = stats

    return result


def authorize_locked_access(
    settings: Settings,
    what: str,
    *,
    meta: Meta | None = None,
    unlock_reason: str | None = None,
) -> None:
    """ロックされたものへのアクセスを許可し、その事実を監査記録に残す。

    **この判定を複製しないこと。** ロックの門はここ1箇所で、期待値スキャン
    （`scan_period`）もダッシュボードのチャートも同じ関数を通す。判定が2箇所に
    分かれると「片方だけ直す／食い違う」という壊れ方をする——実際、
    `Settings.slice_bars` が `PermissionError` を投げる門のような顔をしながら
    `allow_locked=True` で通れる状態になっていた。

    `what` はロック対象の名前（期間名や "chart" など）。記録にそのまま残る。

    人間が一度OOSの結果を見てしまったら、そのOOSはもうOOSではない。覗いた事実を
    meta.db に残しておかないと、後から検証の信頼性を主張できない。だから
    `unlock_reason` と `meta` の**両方**が要る。理由だけを書いて `meta` を省けば
    記録が残らずに覗けてしまい、「解除は記録に残る」という前提が抜け道になる。
    """
    if not unlock_reason:
        raise PermissionError(
            f"{what!r} はロックされている。アクセスするには unlock_reason を明示すること"
        )
    if meta is None:
        raise ValueError(
            "ロックを解除するには meta も渡すこと。"
            " unlock_reason だけでは監査記録が meta.db に残らない"
        )
    # 「*ある* Meta」ではなく「設定が指している meta.db」でなければならない。
    # 使い捨てのDBに書けてしまうなら、監査証跡は自己申告と変わらない。
    if Path(meta.db_path).resolve() != Path(settings.meta_db).resolve():
        raise ValueError(
            f"監査記録の宛先が設定と違う: {meta.db_path} "
            f"（settings.meta_db は {settings.meta_db}）。"
            " ロック解除の記録は、あとから第三者が辿れる1つの場所に残すこと"
        )
    meta.record_oos_unlock(what, unlock_reason)


def scan_period(
    bars: pd.DataFrame,
    signal: pd.Series,
    settings: Settings,
    period: str,
    *,
    meta: Meta | None = None,
    unlock_reason: str | None = None,
    **kwargs,
) -> EdgeResult:
    """期間を限定して集計する。ロック期間は理由の明示が要る。

    人間が一度OOSの結果を見てしまったら、そのOOSはもうOOSではない。覗いた
    事実を meta.db に残しておかないと、後から検証の信頼性を主張できない。

    そのため、ロック期間を解除するには `unlock_reason` と `meta` の**両方**が
    要る。`unlock_reason` だけを渡して `meta` を省略すると理由さえ書けば
    記録が残らずにロック期間を覗けてしまい、「解除は記録に残る」という設計
    （設計文書 §8 (3)）の前提が抜け穴になる。`meta` を省略できる Optional の
    まま単に「渡されたら記録する」設計にしていたのはその抜け穴そのものだった
    ため、ロック期間の解除時は必須に変えた（ロックされていない期間の集計では
    そもそも記録が要らないので `meta=None` のままでよい）。
    """
    target = settings.period_for(period)
    if target.locked:
        authorize_locked_access(
            settings, period, meta=meta, unlock_reason=unlock_reason
        )

    sliced = settings.slice_bars(bars, period, allow_locked=target.locked)
    # signal の bars.index への整列は scan() が一貫して行う（reindex →
    # fillna(False) → astype(bool)）。ここで先に reindex すると、pandas 3 の
    # 挙動（reindexで生じたNaNをfillna(False)しても bool dtype に戻らず
    # object dtype のまま残る）を経由した Series を scan() に渡すことになり、
    # 整列ロジックの入口を二重に持つことになる。scan() 側の一箇所に揃える。
    return scan(sliced, signal, **kwargs)
