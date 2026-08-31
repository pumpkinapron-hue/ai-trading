"""期待値スキャン。バックテストエンジンではない。

TP/SL もポジション管理も資金管理も持たない。「そもそもこの条件に優位性の種が
あるか」を見るための純粋な統計。バックテストエンジンを作る前に使う。

**このツールの唯一の目的は「その優位性は本物か、それともノイズか」を判定すること。**
判定を誤らせる要素は、値が出ていても致命的として扱う（下記 `_effective_n` 参照）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

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
    n_eff: int
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
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_safe(v) for v in value)
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _effective_n(mask: np.ndarray, horizon: int) -> int:
    """「独立に近い」サンプル数の概算。`ci95_pips` の分母に使う。

    考え方: シグナル位置（`mask` が True の位置、時系列順）を、隣同士の間隔が
    `horizon` 本以上離れているところで「クラスタ」に分割する。間隔が
    `horizon` 未満なら、その2つのシグナルの horizon本先までのリターン窓は
    重なっており、独立ではない。各クラスタの中には
    `(クラスタの幅 // horizon) + 1` 個ぶんの「重ならない horizon 幅の窓」が
    収まるとみなし、それをクラスタごとに足し合わせる。

    この式は両極端で直感と一致する。
    - シグナルが毎足連続で `n` 本出るとき: 全体が1クラスタになり、
      `n_eff ≈ n / horizon`（重複しない横幅ぶんの本数）。
    - シグナルが `horizon` 本以上離れて疎に出るとき: 各シグナルが自分だけの
      クラスタになり、`n_eff == n`（重なりが無いのでそのまま独立）。

    **単純に `n_eff = n / horizon` で固定する対策（草案で候補に挙がっていた
    もの）は採らなかった。** 実測すると、シグナルが疎（間隔が horizon 以上）
    なケースで著しく保守的になりすぎる。具体例（乱数シード300本、
    horizon=60、真の優位性=+8pip を仕込んだ検出力テスト）:

    | シグナルの出方 | `n/horizon` 固定での検出率 | このクラスタ方式での検出率 |
    |---|---|---|
    | 毎足連続 | 82.3% | 83.3%（ほぼ同等） |
    | 5本ずつのバースト×間隔180本 | **1.0%** | 90.7% |

    `n/horizon` 固定は「毎足連続で出る」場合にしか正しく効かず、実際のシグナル
    （RSIが閾値を割った、等）はもっと疎・不規則に出ることが普通なので、
    固定式のままだと本物の優位性まで大半見逃す道具になってしまう。

    誤検出率（真の優位性=0のランダムウォークで、CIが0をまたがない頻度。
    本来5%程度であるべき）も、疎・密・バースト状・不規則発火など複数の
    発火パターンで実測し、いずれも 5〜8% 程度に収まることを確認済み
    （`n`をそのまま使う場合の 70〜82% から大幅に改善。ただし正確に5.0%には
    一致しない——本関数は依然として正規分布の97.5%点(1.96)を使っており、
    実効サンプル数が小さいときの t分布との差は補正していない。だからこそ
    `n_eff` 自体を `HorizonStats` に出し、小さい値のときは読む側が慎重に
    扱えるようにしてある）。検証の詳細は `task-10-report.md` の検証A。
    """
    positions = np.flatnonzero(mask)
    if len(positions) == 0:
        return 0

    gaps = np.diff(positions)
    cluster_id = np.concatenate(([0], np.cumsum(gaps >= horizon)))

    total = 0
    for cluster in np.unique(cluster_id):
        cluster_positions = positions[cluster_id == cluster]
        span = int(cluster_positions[-1] - cluster_positions[0])
        total += span // horizon + 1
    return int(total)


def _stats(returns: pd.Series, horizon: int) -> HorizonStats:
    mask = returns.notna().to_numpy()
    values = returns.to_numpy()[mask]
    n = len(values)
    if n == 0:
        return HorizonStats(horizon, 0, 0, np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan))

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    n_eff = _effective_n(mask, horizon)
    half = 1.96 * std / np.sqrt(n_eff) if n_eff > 0 else 0.0
    return HorizonStats(
        horizon=horizon,
        n=n,
        n_eff=n_eff,
        mean_pips=mean,
        median_pips=float(np.median(values)),
        # ちょうど0（利益ゼロ）は「勝ち」に数えない。損益がプラスであることを
        # 「勝ち」の定義にする以上、建値どんとんは負け側に含めるのが妥当。
        # deduct_spread=True（既定）では往復コストが必ず乗るため、ちょうど0の
        # netリターンは連続値の分布上ほぼ測度ゼロで実害は小さい。
        # deduct_spread=False（mid同士）や、価格が丸められて連続する薄商いの
        # 1分足では実際に起こりうるので、境界の扱いをここに明記しておく。
        win_rate=float((values > 0).mean()),
        std_pips=std,
        # float() で明示的に素の Python float へ落とす。half は
        # np.sqrt(n_eff)（numpy.float64）を経由しているため、そのままだと
        # tuple の要素が numpy.float64 になる。このプラットフォーム
        # （numpy 2.5.2 / Windows）では float64 は素の float のサブクラスで
        # json.dumps はそのままでも動くが（実測確認済み）、他の全フィールドが
        # 明示キャストしている以上、ここだけ numpy の継承関係に暗黙に頼るのは
        # 一貫性がない。
        ci95_pips=(float(mean - half), float(mean + half)),
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

    result = EdgeResult(
        n_signals=int(signal.sum()), deduct_spread=deduct_spread, direction=direction
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
        if not unlock_reason:
            raise PermissionError(
                f"期間 {period!r} はロックされている。"
                " 集計するには unlock_reason を明示すること"
            )
        if meta is None:
            raise ValueError(
                "ロック期間を解除するには meta も渡すこと。"
                " unlock_reason だけでは監査記録が meta.db に残らない"
            )
        meta.record_oos_unlock(period, unlock_reason)

    sliced = settings.slice_bars(bars, period, allow_locked=target.locked)
    # signal の bars.index への整列は scan() が一貫して行う（reindex →
    # fillna(False) → astype(bool)）。ここで先に reindex すると、pandas 3 の
    # 挙動（reindexで生じたNaNをfillna(False)しても bool dtype に戻らず
    # object dtype のまま残る）を経由した Series を scan() に渡すことになり、
    # 整列ロジックの入口を二重に持つことになる。scan() 側の一箇所に揃える。
    return scan(sliced, signal, **kwargs)
