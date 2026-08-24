# Phase 0 研究基盤 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** USD/JPY の市場データを取得・保存し、先読みバイアスが構造的に起きない形で指標計算・期待値スキャン・チャート表示ができる研究基盤を作る。

**Architecture:** データソースは Protocol で抽象化し、Dukascopy 実装を1本用意する。1分足のみを取得して Parquet レイクに保存し、上位足・日足2系統（NY/JST）はすべて1分足から生成する。データアクセスは `as_of` 必須のカーソル API に一本化し、指定時刻より後のバーを物理的に返さない。指標は全て `bars: DataFrame -> Series|DataFrame` の純関数としてレジストリに登録し、レジストリ全体にトランケーション不変性テストを自動適用する。

**Tech Stack:** Python 3.12 / uv / pandas / numpy / pyarrow (Parquet) / sqlite3 (標準ライブラリ) / Streamlit / Plotly / pytest

**設計文書:** [2026-08-24-phase0-design.md](../specs/2026-08-24-phase0-design.md)
**上位仕様:** [spec-v0.1.md](../specs/spec-v0.1.md)

## Global Constraints

すべてのタスクの要件に、暗黙にこのセクションが含まれる。

- **Python 3.12 以上。** パッケージ管理は `uv`。`pip install` は使わない。
- **時刻はすべて UTC の tz-aware。** naive な `datetime` / `DatetimeIndex` を関数に渡したら `ValueError` を送出する。テストで縛る。
- **`Lake.load()` の `as_of` はキーワード必須引数。** デフォルト値を与えない。省略できると先読みが書けてしまう。
- **指標は `bars: pd.DataFrame` を第1引数に取り、`pd.Series` または `pd.DataFrame` を返す純関数。** グローバル状態を持たない。
- **バーの列名は固定:** `open_time, close_time, bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, volume`。価格は `float64`。
- **`open_time` / `close_time` は `datetime64[ns, UTC]`。** `close_time = open_time + timeframe の期間`。
- **モデルID・期間分割・パスは `config/settings.toml` に置く。** コードに直書きしない。
- **`data/` 配下は Git 管理外。** テストは `tmp_path` を使い、`data/` を汚さない。
- **各タスクの最後にコミットする。** コミットメッセージは日本語、本文に「何を・なぜ」を書く。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `pyproject.toml` | 依存関係・pytest設定 |
| `config/settings.toml` | データ範囲・期間分割・パス・モデルID |
| `src/aitrading/config.py` | settings.toml の読み込みと期間ロック判定 |
| `src/aitrading/timeutil.py` | Timeframe/Session の定義、UTC強制、日境界2系統、セッション判定、市場開閉 |
| `src/aitrading/datasource/base.py` | バースキーマ定義・検証、`BarSource` Protocol |
| `src/aitrading/datasource/dukascopy.py` | Dukascopy からの取得と正規化 |
| `src/aitrading/storage/lake.py` | Parquet 読み書きと `as_of` カーソル |
| `src/aitrading/storage/meta.py` | SQLite（取得済み区間・品質レポート・OOS解除ログ） |
| `src/aitrading/bars.py` | 1分足からのリサンプル（上位足・日足2系統） |
| `src/aitrading/quality.py` | データ品質チェック |
| `src/aitrading/indicators/registry.py` | 指標レジストリ（デコレータ） |
| `src/aitrading/indicators/core.py` | 指標本体 |
| `src/aitrading/edge_scan.py` | 期待値スキャン |
| `scripts/fetch_data.py` | CLI: データ取得 |
| `scripts/build_bars.py` | CLI: 上位足生成 |
| `dashboard/app.py` | Streamlit |

---

## Task 1: プロジェクト初期化と時刻基盤

時刻の規約は後から変えると全データを取り直すことになるので、最初に固めてテストで縛る。プロジェクトのセットアップとLLM依存禁止ガードもここに含める。

**Files:**
- Create: `pyproject.toml`
- Create: `src/aitrading/__init__.py`
- Create: `src/aitrading/timeutil.py`
- Create: `tests/__init__.py`
- Create: `tests/test_timeutil.py`
- Create: `tests/test_no_llm_in_execution.py`

**Interfaces:**
- Consumes: なし（最初のタスク）
- Produces:
  - `class Timeframe(str, Enum)` — 値は `"1m" "5m" "15m" "1h" "4h" "1D_ny" "1D_jst" "1W_ny" "1W_jst"`、メンバ名は `M1 M5 M15 H1 H4 D1_NY D1_JST W1_NY W1_JST`
  - `Timeframe.delta -> pd.Timedelta | None`（日足・週足は可変長なので `None`）
  - `class Session(str, Enum)` — `TOKYO LONDON NEWYORK LDN_NY_OVERLAP OFF`
  - `ensure_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex`（naive なら `ValueError`）
  - `session_labels(index: pd.DatetimeIndex) -> pd.Series`（値は `Session`）
  - `is_market_open(index: pd.DatetimeIndex) -> pd.Series`（bool）
  - `trading_day_start(index: pd.DatetimeIndex, convention: str) -> pd.DatetimeIndex`（`convention` は `"ny"` か `"jst"`、返り値は UTC）

- [ ] **Step 1: `pyproject.toml` を作る**

```toml
[project]
name = "aitrading"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "pyarrow>=16",
    "plotly>=5.20",
    "streamlit>=1.37",
    "dukascopy-python>=4.0",
]

[dependency-groups]
dev = ["pytest>=8", "ta>=0.11"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/aitrading"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["network: 実際のDukascopyサーバーに接続する"]
addopts = "-m 'not network'"
```

`uv sync` を実行して依存関係を入れる。`src/aitrading/__init__.py` と `tests/__init__.py` は空ファイルで作る。

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_timeutil.py`:

```python
import pandas as pd
import pytest

from aitrading.timeutil import (
    Session,
    Timeframe,
    ensure_utc,
    is_market_open,
    session_labels,
    trading_day_start,
)


def idx(*stamps: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(list(stamps), utc=True))


def test_timeframe_delta():
    assert Timeframe.M5.delta == pd.Timedelta(minutes=5)
    assert Timeframe.H4.delta == pd.Timedelta(hours=4)
    assert Timeframe.D1_NY.delta is None


def test_ensure_utc_rejects_naive():
    naive = pd.DatetimeIndex(pd.to_datetime(["2026-01-05 00:00"]))
    with pytest.raises(ValueError, match="tz-aware"):
        ensure_utc(naive)


def test_ensure_utc_converts_other_zone():
    tokyo = pd.DatetimeIndex(pd.to_datetime(["2026-01-05 09:00"])).tz_localize("Asia/Tokyo")
    assert ensure_utc(tokyo)[0] == pd.Timestamp("2026-01-05 00:00", tz="UTC")


def test_session_labels_tokyo():
    # 2026-01-05 は月曜。JST 10:00 = UTC 01:00 は東京単独。
    assert session_labels(idx("2026-01-05 01:00Z"))[0] == Session.TOKYO


def test_session_labels_overlap_follows_dst():
    # 冬（EST/GMT）: NY 09:00 = UTC 14:00、ロンドンは GMT 14:00 で 16:30 まで開いている
    assert session_labels(idx("2026-01-05 14:00Z"))[0] == Session.LDN_NY_OVERLAP
    # 夏（EDT/BST）: NY 09:00 = UTC 13:00。固定オフセットで書いていると外れる
    assert session_labels(idx("2026-07-06 13:00Z"))[0] == Session.LDN_NY_OVERLAP


def test_session_labels_off_hours():
    # JST 06:00 = UTC 21:00(前日)。東京前の薄商い帯
    assert session_labels(idx("2026-01-05 21:00Z"))[0] == Session.OFF


def test_market_closed_on_weekend():
    # 土曜はクローズ
    assert not is_market_open(idx("2026-01-10 12:00Z"))[0]
    # 金曜 NY 17:00 EST = 22:00 UTC 以降はクローズ
    assert not is_market_open(idx("2026-01-09 22:30Z"))[0]
    assert is_market_open(idx("2026-01-09 20:00Z"))[0]
    # 日曜 NY 17:00 EST = 22:00 UTC 以降はオープン
    assert is_market_open(idx("2026-01-11 23:00Z"))[0]


def test_trading_day_start_ny_winter():
    # 冬: NY 17:00 = 22:00 UTC。2026-01-06 01:00Z は 2026-01-05 22:00Z 始まりの日に属する
    got = trading_day_start(idx("2026-01-06 01:00Z"), "ny")[0]
    assert got == pd.Timestamp("2026-01-05 22:00", tz="UTC")


def test_trading_day_start_ny_summer():
    # 夏: NY 17:00 = 21:00 UTC
    got = trading_day_start(idx("2026-07-07 01:00Z"), "ny")[0]
    assert got == pd.Timestamp("2026-07-06 21:00", tz="UTC")


def test_trading_day_start_jst_has_no_dst():
    # JST 00:00 = 前日 15:00 UTC。夏でも冬でも同じ
    assert trading_day_start(idx("2026-01-06 01:00Z"), "jst")[0] == pd.Timestamp(
        "2026-01-05 15:00", tz="UTC"
    )
    assert trading_day_start(idx("2026-07-07 01:00Z"), "jst")[0] == pd.Timestamp(
        "2026-07-06 15:00", tz="UTC"
    )


def test_ny_and_jst_day_boundaries_differ():
    # 同じ瞬間が、NY基準とJST基準で別の日に属することがある
    ts = idx("2026-01-06 01:00Z")
    assert trading_day_start(ts, "ny")[0] != trading_day_start(ts, "jst")[0]
```

- [ ] **Step 3: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_timeutil.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.timeutil'`

- [ ] **Step 4: `src/aitrading/timeutil.py` を実装する**

```python
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
```

- [ ] **Step 5: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_timeutil.py -v`
Expected: PASS（12 passed）

- [ ] **Step 6: LLM依存禁止ガードのテストを書く**

`tests/test_no_llm_in_execution.py`:

```python
"""執行パスにLLMクライアントが混入していないことを検査する。

執行の判断は1秒未満で終わる必要がある一方、LLMは5〜15秒かかる。
方針を書くだけでは後から誰かが便利さに負けて入れてしまうので、テストで縛る。
Phase 0 では対象モジュールがまだ存在せず自明に通るが、誘惑が生まれる前に入れておく。
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "aitrading"

#: 実行時のホットパス。ここにLLMクライアントを入れてはいけない。
EXECUTION_PACKAGES = ("execution", "risk", "strategy")

LLM_MODULES = {"anthropic", "openai", "google", "cohere", "litellm", "langchain"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_execution_modules_do_not_import_llm_clients():
    offenders = []
    for package in EXECUTION_PACKAGES:
        for path in (SRC / package).rglob("*.py"):
            hits = _imported_roots(path) & LLM_MODULES
            if hits:
                offenders.append(f"{path.relative_to(SRC)}: {sorted(hits)}")
    assert not offenders, "執行パスにLLMクライアントが混入している:\n" + "\n".join(offenders)
```

`SRC / package` が存在しない場合 `rglob` は空を返すので、Phase 0 では自明に通る。

- [ ] **Step 7: テストを実行する**

Run: `uv run pytest tests/ -v`
Expected: PASS（13 passed）

- [ ] **Step 8: コミット**

```bash
git add pyproject.toml src/aitrading/__init__.py src/aitrading/timeutil.py tests/
git commit -m "時刻基盤を実装し、LLM依存禁止ガードを入れた

日境界2系統（NYクローズ17:00・JST0:00）とセッションラベルを実装した。
市場ローカルのタイムゾーン名を経由して変換するため、夏時間の切り替え週でも
1時間ずれない。テストで冬（1月）と夏（7月）の両方を検証している。

週末クローズを is_market_open で明示的に扱う。これがないと品質チェックが
毎週末に偽陽性を出す。

執行パスにLLMクライアントをimportしていないことの検査も入れた。対象モジュールは
まだ無く自明に通るが、誘惑が生まれる前に置いておく。"
```

---

## Task 2: 設定と期間分割（OOSロック）

期間分割はコードに直書きせず設定から読む。OOS期間はデフォルトでロックし、解除には明示的な操作が要る。

**Files:**
- Create: `config/settings.toml`
- Create: `src/aitrading/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: なし
- Produces:
  - `@dataclass(frozen=True) class Period: name: str; start: pd.Timestamp; end: pd.Timestamp; locked: bool`
  - `@dataclass(frozen=True) class Settings` — 属性 `symbol: str`, `data_start: pd.Timestamp`, `data_root: Path`, `meta_db: Path`, `periods: dict[str, Period]`, `models: dict[str, str]`
  - `load_settings(path: Path | None = None) -> Settings`
  - `Settings.period_for(name: str) -> Period`
  - `Settings.slice_bars(df: pd.DataFrame, period: str, *, allow_locked: bool = False) -> pd.DataFrame` — ロックされた期間を `allow_locked=False` で切ろうとしたら `PermissionError`

- [ ] **Step 1: `config/settings.toml` を作る**

```toml
[data]
symbol = "USDJPY"
start = "2015-01-01"
root = "data"
meta_db = "data/meta.db"

# 期間分割。locked = true の期間は明示的に解除しないと集計に使えない。
# 一度OOSの結果を見てしまったら、そのOOSはもうOOSではない。
[periods.training]
start = "2015-01-01"
end = "2021-12-31"
locked = false

[periods.validation]
start = "2022-01-01"
end = "2023-12-31"
locked = false

[periods.oos]
start = "2024-01-01"
end = "2026-08-24"
locked = true

[periods.forward]
start = "2026-08-25"
end = "2099-12-31"
locked = true

# 研究レイヤーのモデル割り当て。頻度と難易度で使い分ける。
# 執行パスでは使わない（tests/test_no_llm_in_execution.py 参照）。
[models]
strategy_generation = "claude-fable-5"
backtest_analysis = "claude-opus-5"
trade_review = "claude-sonnet-5"
knowledge_extraction = "claude-sonnet-5"
news_classification = "claude-haiku-4-5"
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/test_config.py`:

```python
import pandas as pd
import pytest

from aitrading.config import load_settings


@pytest.fixture
def settings():
    return load_settings()


def test_loads_symbol_and_periods(settings):
    assert settings.symbol == "USDJPY"
    assert set(settings.periods) == {"training", "validation", "oos", "forward"}


def test_oos_is_locked_by_default(settings):
    assert settings.period_for("oos").locked is True
    assert settings.period_for("training").locked is False


def test_periods_do_not_overlap(settings):
    ordered = sorted(settings.periods.values(), key=lambda p: p.start)
    for earlier, later in zip(ordered, ordered[1:]):
        assert earlier.end < later.start, f"{earlier.name} と {later.name} が重なっている"


def test_models_are_configured_not_hardcoded(settings):
    assert settings.models["strategy_generation"] == "claude-fable-5"
    assert settings.models["news_classification"] == "claude-haiku-4-5"


def _bars(index):
    return pd.DataFrame({"bid_close": range(len(index))}, index=index)


def test_slice_bars_returns_only_the_period(settings):
    index = pd.DatetimeIndex(
        pd.to_datetime(["2021-06-01", "2022-06-01", "2024-06-01"], utc=True)
    )
    got = settings.slice_bars(_bars(index), "training")
    assert len(got) == 1
    assert got.index[0] == pd.Timestamp("2021-06-01", tz="UTC")


def test_slice_bars_refuses_locked_period(settings):
    index = pd.DatetimeIndex(pd.to_datetime(["2024-06-01"], utc=True))
    with pytest.raises(PermissionError, match="oos"):
        settings.slice_bars(_bars(index), "oos")


def test_slice_bars_allows_locked_period_when_explicit(settings):
    index = pd.DatetimeIndex(pd.to_datetime(["2024-06-01"], utc=True))
    got = settings.slice_bars(_bars(index), "oos", allow_locked=True)
    assert len(got) == 1
```

- [ ] **Step 3: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.config'`

- [ ] **Step 4: `src/aitrading/config.py` を実装する**

```python
"""設定の読み込み。期間分割とモデルIDはコードに直書きしない。"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
        """指定期間だけを切り出す。ロックされた期間は明示的な解除が要る。

        過学習対策は、人間が「ちょっとだけ覗く」のを防げないと機能しない。
        """
        target = self.period_for(period)
        if target.locked and not allow_locked:
            raise PermissionError(
                f"期間 {target.name!r} はロックされている。"
                " 解除するには allow_locked=True を明示し、解除理由を meta.db に記録すること"
            )
        # end は日付指定なのでその日の終わりまで含める
        end = target.end + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        return df.loc[(df.index >= target.start) & (df.index <= end)]


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
```

- [ ] **Step 5: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS（7 passed）

- [ ] **Step 6: コミット**

```bash
git add config/settings.toml src/aitrading/config.py tests/test_config.py
git commit -m "設定の読み込みと期間ロックを実装

Training/Validation/OOS/Forward の期間を settings.toml から読む。
OOS と Forward は locked=true にしてあり、slice_bars で切ろうとすると
PermissionError になる。allow_locked=True を明示しないと通らない。

過学習対策は人間が「ちょっとだけ覗く」のを防げないと機能しないため、
方針ではなく仕組みで縛る。

モデルIDも settings.toml に置いた。新しいモデルが出たときに1箇所で切り替える。"
```

---

## Task 3: バースキーマと `BarSource` Protocol

データソースが何であれ、上位層が受け取る形を1つに固定する。ここがぶれると後続すべてに波及する。

**Files:**
- Create: `src/aitrading/datasource/__init__.py`
- Create: `src/aitrading/datasource/base.py`
- Create: `tests/test_datasource_base.py`

**Interfaces:**
- Consumes: `aitrading.timeutil.Timeframe`, `ensure_utc`
- Produces:
  - `BAR_COLUMNS: list[str]` — `["open_time","close_time","bid_open","bid_high","bid_low","bid_close","ask_open","ask_high","ask_low","ask_close","volume"]`
  - `PRICE_COLUMNS: list[str]` — `BAR_COLUMNS` のうち価格9列（`volume` を除く7列 + volume を除いた bid/ask 各4列 = 8列）
  - `validate_bars(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame` — 正規化して返す。違反は `ValueError`
  - `class BarSource(Protocol)` — `fetch(symbol, timeframe, start, end) -> pd.DataFrame`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_datasource_base.py`:

```python
import pandas as pd
import pytest

from aitrading.datasource.base import BAR_COLUMNS, validate_bars
from aitrading.timeutil import Timeframe


def make_bars(n: int = 3, tz: str | None = "UTC") -> pd.DataFrame:
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz=tz)
    close_time = open_time + pd.Timedelta(minutes=1)
    body = {"open_time": open_time, "close_time": close_time}
    for side, base in (("bid", 150.0), ("ask", 150.02)):
        for field, bump in (("open", 0.0), ("high", 0.03), ("low", -0.03), ("close", 0.01)):
            body[f"{side}_{field}"] = [base + bump] * n
    body["volume"] = [100.0] * n
    return pd.DataFrame(body)


def test_accepts_valid_bars():
    got = validate_bars(make_bars(), Timeframe.M1)
    assert list(got.columns) == BAR_COLUMNS
    assert got["bid_open"].dtype == "float64"


def test_rejects_naive_timestamps():
    with pytest.raises(ValueError, match="tz-aware"):
        validate_bars(make_bars(tz=None), Timeframe.M1)


def test_rejects_missing_column():
    bars = make_bars().drop(columns=["ask_high"])
    with pytest.raises(ValueError, match="ask_high"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_wrong_close_time():
    bars = make_bars()
    bars.loc[1, "close_time"] += pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match="close_time"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_duplicate_open_time():
    bars = make_bars()
    bars.loc[1, "open_time"] = bars.loc[0, "open_time"]
    bars.loc[1, "close_time"] = bars.loc[0, "close_time"]
    with pytest.raises(ValueError, match="重複"):
        validate_bars(bars, Timeframe.M1)


def test_rejects_ask_below_bid():
    bars = make_bars()
    bars.loc[1, "ask_close"] = bars.loc[1, "bid_close"] - 0.10
    with pytest.raises(ValueError, match="Ask"):
        validate_bars(bars, Timeframe.M1)


def test_sorts_by_open_time():
    bars = make_bars().iloc[::-1].reset_index(drop=True)
    got = validate_bars(bars, Timeframe.M1)
    assert got["open_time"].is_monotonic_increasing
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_datasource_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.datasource'`

- [ ] **Step 3: `src/aitrading/datasource/base.py` を実装する**

`src/aitrading/datasource/__init__.py` は空ファイル。

```python
"""バーの共通スキーマ。データソースが何であれ上位層が見る形はこれ1つ。"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import pandas as pd

from aitrading.timeutil import Timeframe, ensure_utc

TIME_COLUMNS = ["open_time", "close_time"]

PRICE_COLUMNS = [
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
]

BAR_COLUMNS = TIME_COLUMNS + PRICE_COLUMNS + ["volume"]


class BarSource(Protocol):
    """市場データの取得元。Dukascopy / OANDA / MT5 はすべてこれを実装する。"""

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """[start, end) のバーを BAR_COLUMNS のスキーマで返す。"""
        ...


def validate_bars(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """スキーマを検証して正規化する。違反は握りつぶさず ValueError にする。

    ここを緩めると、壊れたデータが静かにレイクに入って後段すべてを汚染する。
    """
    missing = [c for c in BAR_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"列が足りない: {missing}")

    out = df.loc[:, BAR_COLUMNS].copy()

    for column in TIME_COLUMNS:
        index = pd.DatetimeIndex(out[column])
        if index.tz is None:
            raise ValueError(f"{column} が tz-aware でない。UTCで渡すこと")
        out[column] = index.tz_convert("UTC")

    out = out.sort_values("open_time").reset_index(drop=True)

    if out["open_time"].duplicated().any():
        dupes = out.loc[out["open_time"].duplicated(), "open_time"].tolist()
        raise ValueError(f"open_time が重複している: {dupes[:5]}")

    delta = timeframe.delta
    if delta is not None:
        bad = out["close_time"] - out["open_time"] != delta
        if bad.any():
            raise ValueError(
                f"close_time が open_time + {delta} になっていない行が {int(bad.sum())} 件ある"
            )

    for column in PRICE_COLUMNS + ["volume"]:
        out[column] = out[column].astype("float64")

    crossed = out["ask_close"] < out["bid_close"]
    if crossed.any():
        raise ValueError(f"Ask が Bid を下回る行が {int(crossed.sum())} 件ある")

    return out
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_datasource_base.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: コミット**

```bash
git add src/aitrading/datasource/ tests/test_datasource_base.py
git commit -m "バーの共通スキーマと BarSource Protocol を定義

Bid/Ask を両方持つ11列のスキーマに固定した。mid に丸めると
「買いはAsk・売りはBid」をバックテストで再現できず、取引コスト込みの
期待値が検証できなくなるため。

validate_bars は違反を握りつぶさず ValueError にする。壊れたデータが
静かにレイクに入ると後段すべてを汚染する。"
```

---

## Task 4: Dukascopy アダプタ

外部サービスに触る唯一の場所。取得ロジックと正規化ロジックを分け、正規化だけをネットワークなしでテストする。

**Files:**
- Create: `src/aitrading/datasource/dukascopy.py`
- Create: `tests/test_dukascopy.py`

**Interfaces:**
- Consumes: `BAR_COLUMNS`, `validate_bars`, `Timeframe`
- Produces:
  - `class DukascopySource` — `BarSource` の実装
  - `DukascopySource.fetch(symbol, timeframe, start, end) -> pd.DataFrame`
  - `normalize(bid: pd.DataFrame, ask: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame` — ライブラリの生出力2枚（bid側・ask側）を `BAR_COLUMNS` に変換する純関数

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dukascopy.py`:

```python
import pandas as pd
import pytest

from aitrading.datasource.base import BAR_COLUMNS
from aitrading.datasource.dukascopy import DukascopySource, normalize
from aitrading.timeutil import Timeframe


def raw_side(base: float) -> pd.DataFrame:
    """dukascopy-python が返す形（timestamp index + OHLCV）を模した生データ。"""
    index = pd.date_range("2026-01-05 00:00", periods=3, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [base, base + 0.01, base + 0.02],
            "high": [base + 0.03] * 3,
            "low": [base - 0.03] * 3,
            "close": [base + 0.01] * 3,
            "volume": [100.0, 110.0, 120.0],
        },
        index=index,
    )


def test_normalize_produces_schema():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert list(got.columns) == BAR_COLUMNS
    assert len(got) == 3


def test_normalize_maps_bid_and_ask_separately():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert got.loc[0, "bid_open"] == pytest.approx(150.00)
    assert got.loc[0, "ask_open"] == pytest.approx(150.02)


def test_normalize_sets_close_time():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert got.loc[0, "close_time"] - got.loc[0, "open_time"] == pd.Timedelta(minutes=1)


def test_normalize_takes_volume_from_bid_side():
    got = normalize(raw_side(150.00), raw_side(150.02), Timeframe.M1)
    assert got["volume"].tolist() == [100.0, 110.0, 120.0]


def test_normalize_drops_rows_missing_on_one_side():
    bid = raw_side(150.00)
    ask = raw_side(150.02).iloc[1:]
    got = normalize(bid, ask, Timeframe.M1)
    assert len(got) == 2


def test_normalize_localizes_naive_index_as_utc():
    bid = raw_side(150.00)
    bid.index = bid.index.tz_localize(None)
    got = normalize(bid, raw_side(150.02), Timeframe.M1)
    assert got.loc[0, "open_time"] == pd.Timestamp("2026-01-05 00:00", tz="UTC")


@pytest.mark.network
def test_fetch_real_data():
    """実サーバーに触る唯一のテスト。既定では -m 'not network' で除外される。"""
    source = DukascopySource()
    got = source.fetch(
        "USDJPY",
        Timeframe.M1,
        pd.Timestamp("2026-01-05 00:00", tz="UTC"),
        pd.Timestamp("2026-01-05 01:00", tz="UTC"),
    )
    assert not got.empty
    assert list(got.columns) == BAR_COLUMNS
    assert (got["ask_close"] >= got["bid_close"]).all()
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_dukascopy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.datasource.dukascopy'`

- [ ] **Step 3: `src/aitrading/datasource/dukascopy.py` を実装する**

```python
"""Dukascopy からの取得。外部サービスに触るのはこのファイルだけ。

ライブラリの都合はすべてここに閉じ込める。dukascopy-python が不安定なら
.bi5 の直接ダウンロードに差し替えるが、normalize より上は影響を受けない。
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from aitrading.datasource.base import BAR_COLUMNS, validate_bars
from aitrading.timeutil import Timeframe

#: 内部の Timeframe → dukascopy-python の interval 定数名
_INTERVAL_NAMES = {
    Timeframe.M1: "INTERVAL_MIN_1",
    Timeframe.M5: "INTERVAL_MIN_5",
    Timeframe.M15: "INTERVAL_MIN_15",
    Timeframe.H1: "INTERVAL_HOUR_1",
    Timeframe.H4: "INTERVAL_HOUR_4",
}


def _as_utc_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        return index.tz_localize("UTC")
    return index.tz_convert("UTC")


def normalize(bid: pd.DataFrame, ask: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """ライブラリの生出力（bid側・ask側の2枚）を共通スキーマ1枚にする。

    片側にしか無い時刻は落とす。片側だけのバーはスプレッドが計算できず、
    約定モデルに使えないため。
    """
    delta = timeframe.delta
    if delta is None:
        raise ValueError(f"{timeframe} は取得対象ではない（1分足から生成する）")

    bid = bid.copy()
    ask = ask.copy()
    bid.index = _as_utc_index(bid)
    ask.index = _as_utc_index(ask)

    common = bid.index.intersection(ask.index).sort_values()
    bid = bid.loc[common]
    ask = ask.loc[common]

    body = {"open_time": common, "close_time": common + delta}
    for side, frame in (("bid", bid), ("ask", ask)):
        for field in ("open", "high", "low", "close"):
            body[f"{side}_{field}"] = frame[field].to_numpy()
    body["volume"] = bid["volume"].to_numpy()

    return validate_bars(pd.DataFrame(body, columns=BAR_COLUMNS), timeframe)


class DukascopySource:
    """BarSource の Dukascopy 実装。"""

    def fetch(
        self,
        symbol: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        import dukascopy_python
        from dukascopy_python.instruments import INSTRUMENT_FX_MAJORS_USD_JPY

        instruments = {"USDJPY": INSTRUMENT_FX_MAJORS_USD_JPY}
        if symbol not in instruments:
            raise ValueError(f"未対応のシンボル: {symbol!r}")

        interval = getattr(dukascopy_python, _INTERVAL_NAMES[timeframe])
        sides = {}
        for name, offer_side in (
            ("bid", dukascopy_python.OFFER_SIDE_BID),
            ("ask", dukascopy_python.OFFER_SIDE_ASK),
        ):
            sides[name] = dukascopy_python.fetch(
                instrument=instruments[symbol],
                interval=interval,
                offer_side=offer_side,
                start=pd.Timestamp(start).to_pydatetime(),
                end=pd.Timestamp(end).to_pydatetime(),
            )

        return normalize(sides["bid"], sides["ask"], timeframe)
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_dukascopy.py -v`
Expected: PASS（6 passed, 1 deselected）

- [ ] **Step 5: 実データで1時間分だけ取得して手で確かめる**

Run: `uv run pytest tests/test_dukascopy.py -v -m network`

Expected: PASS。失敗した場合は `dukascopy_python` の定数名（`_INTERVAL_NAMES` と `INSTRUMENT_FX_MAJORS_USD_JPY`）が実際のバージョンと違う可能性が高い。`uv run python -c "import dukascopy_python; print([n for n in dir(dukascopy_python) if n.startswith(('INTERVAL','OFFER')) ])"` で実際の名前を確認して `_INTERVAL_NAMES` を直す。**修正は `dukascopy.py` の中だけで完結させること** — 上位層のスキーマは変えない。

- [ ] **Step 6: コミット**

```bash
git add src/aitrading/datasource/dukascopy.py tests/test_dukascopy.py
git commit -m "Dukascopy アダプタを実装

bid側とask側を別々に取得して1枚のスキーマに合成する。片側にしか無い時刻は
落とす（スプレッドが計算できず約定モデルに使えないため）。

ネットワークに触るテストは network マーカーで既定から除外し、normalize の
変換ロジックだけを常時テストする。ライブラリを .bi5 直接ダウンロードに
差し替えても normalize より上は影響を受けない。"
```

---

## Task 5: Parquet レイクと `as_of` カーソル

先読み防止の第4層。`as_of` を必須にすることで、未来のデータをメモリに載せること自体をできなくする。

**Files:**
- Create: `src/aitrading/storage/__init__.py`
- Create: `src/aitrading/storage/lake.py`
- Create: `tests/test_lake.py`

**Interfaces:**
- Consumes: `BAR_COLUMNS`, `validate_bars`, `Timeframe`
- Produces:
  - `class Lake` — `__init__(self, root: Path)`
  - `Lake.save(symbol: str, timeframe: Timeframe, df: pd.DataFrame) -> None` — 年ごとに分割、既存と結合して重複除去
  - `Lake.load(symbol: str, timeframe: Timeframe, *, as_of: pd.Timestamp, start: pd.Timestamp | None = None) -> pd.DataFrame` — `open_time` を index にした DataFrame を返す
  - `Lake.available_years(symbol: str, timeframe: Timeframe) -> list[int]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_lake.py`:

```python
import pandas as pd
import pytest

from aitrading.storage.lake import Lake
from aitrading.timeutil import Timeframe

from tests.test_datasource_base import make_bars


def bars_over(start: str, periods: int) -> pd.DataFrame:
    open_time = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    df = make_bars(periods)
    df["open_time"] = open_time
    df["close_time"] = open_time + pd.Timedelta(minutes=1)
    return df


@pytest.fixture
def lake(tmp_path):
    return Lake(tmp_path)


def test_save_then_load_roundtrip(lake):
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 5))
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert len(got) == 5
    assert got.index.name == "open_time"
    assert got.index.tz is not None


def test_as_of_hides_future_bars(lake):
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 10))
    got = lake.load(
        "USDJPY", Timeframe.M1, as_of=pd.Timestamp("2026-01-05 00:05", tz="UTC")
    )
    # close_time <= as_of のバーだけ。00:00開始の足は00:01に確定するので5本。
    assert len(got) == 5
    assert got["close_time"].max() <= pd.Timestamp("2026-01-05 00:05", tz="UTC")


def test_as_of_is_keyword_only_and_required(lake):
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 3))
    with pytest.raises(TypeError):
        lake.load("USDJPY", Timeframe.M1)  # as_of なしは呼べない


def test_start_filters_lower_bound(lake):
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 10))
    got = lake.load(
        "USDJPY",
        Timeframe.M1,
        as_of=pd.Timestamp("2030-01-01", tz="UTC"),
        start=pd.Timestamp("2026-01-05 00:07", tz="UTC"),
    )
    assert got.index.min() >= pd.Timestamp("2026-01-05 00:07", tz="UTC")


def test_save_splits_by_year(lake):
    lake.save("USDJPY", Timeframe.M1, bars_over("2025-12-31 23:58", 5))
    assert lake.available_years("USDJPY", Timeframe.M1) == [2025, 2026]


def test_save_is_idempotent(lake):
    batch = bars_over("2026-01-05 00:00", 5)
    lake.save("USDJPY", Timeframe.M1, batch)
    lake.save("USDJPY", Timeframe.M1, batch)
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert len(got) == 5


def test_save_merges_new_bars_into_existing_year(lake):
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 3))
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:03", 3))
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert len(got) == 6
    assert got.index.is_monotonic_increasing


def test_load_missing_symbol_returns_empty_with_schema(lake):
    got = lake.load("EURUSD", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert got.empty
    assert "bid_close" in got.columns
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_lake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.storage'`

- [ ] **Step 3: `src/aitrading/storage/lake.py` を実装する**

`src/aitrading/storage/__init__.py` は空ファイル。

```python
"""Parquet レイク。データアクセスはここに一本化する。

load() の as_of がキーワード必須引数なのは意図的。省略できると、
時点Tのシミュレーション中に未来のバーを読むコードが書けてしまう。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aitrading.datasource.base import BAR_COLUMNS, validate_bars
from aitrading.timeutil import Timeframe


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
        """
        df = validate_bars(df, timeframe)
        if df.empty:
            return

        directory = self._dir(symbol, timeframe)
        directory.mkdir(parents=True, exist_ok=True)

        for year, chunk in df.groupby(df["open_time"].dt.year):
            path = self._path(symbol, timeframe, int(year))
            if path.exists():
                chunk = pd.concat([pd.read_parquet(path), chunk], ignore_index=True)
            chunk = (
                chunk.drop_duplicates(subset="open_time", keep="last")
                .sort_values("open_time")
                .reset_index(drop=True)
            )
            chunk.to_parquet(path, index=False)

    def load(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        as_of: pd.Timestamp,
        start: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """as_of 時点で確定しているバーだけを返す。

        close_time <= as_of が条件。形成中の足は返さない。
        """
        as_of = pd.Timestamp(as_of)
        if as_of.tz is None:
            raise ValueError("as_of は tz-aware で渡すこと")

        years = self.available_years(symbol, timeframe)
        if start is not None:
            start = pd.Timestamp(start)
            years = [y for y in years if y >= start.year]
        years = [y for y in years if y <= as_of.year]

        frames = [pd.read_parquet(self._path(symbol, timeframe, y)) for y in years]
        if frames:
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS})
            for column in ("open_time", "close_time"):
                df[column] = pd.Series(dtype="datetime64[ns, UTC]")

        df = df.loc[df["close_time"] <= as_of]
        if start is not None:
            df = df.loc[df["open_time"] >= start]

        return (
            df.sort_values("open_time")
            .set_index("open_time")
            .rename_axis("open_time")
        )
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_lake.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: コミット**

```bash
git add src/aitrading/storage/ tests/test_lake.py
git commit -m "Parquetレイクと as_of カーソルを実装（先読み防止 第4層）

load() の as_of をキーワード必須引数にした。省略できると、時点Tの
シミュレーション中に未来のバーを読むコードが書けてしまう。
返すのは close_time <= as_of のバーだけで、形成中の足は出てこない。

保存は年ごとに分割し、既存と結合して重複を落とす。再取得が冪等になるので
途中で失敗しても同じコマンドを再実行できる。"
```

---

## Task 6: SQLite メタデータ

取得済み区間・品質レポート・OOS解除ログを持つ。OOS解除を記録に残すのは、後から「いつ誰が覗いたか」を追うため。

**Files:**
- Create: `src/aitrading/storage/meta.py`
- Create: `tests/test_meta.py`

**Interfaces:**
- Consumes: `Timeframe`
- Produces:
  - `class Meta` — `__init__(self, db_path: Path)`（コンストラクタでスキーマを作る）
  - `Meta.record_fetch(symbol, timeframe, start, end) -> None`
  - `Meta.fetched_ranges(symbol, timeframe) -> list[tuple[pd.Timestamp, pd.Timestamp]]`（隣接・重複区間はマージ済み）
  - `Meta.record_quality(symbol, timeframe, report: dict) -> None`
  - `Meta.latest_quality(symbol, timeframe) -> dict | None`
  - `Meta.record_oos_unlock(period: str, reason: str) -> None`
  - `Meta.oos_unlocks() -> list[dict]`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_meta.py`:

```python
import pandas as pd
import pytest

from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe


@pytest.fixture
def meta(tmp_path):
    return Meta(tmp_path / "meta.db")


def ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def test_records_and_reads_fetch_range(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [
        (ts("2026-01-05"), ts("2026-01-06"))
    ]


def test_merges_adjacent_ranges(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-06"), ts("2026-01-07"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M1) == [
        (ts("2026-01-05"), ts("2026-01-07"))
    ]


def test_keeps_disjoint_ranges_separate(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-02-01"), ts("2026-02-02"))
    assert len(meta.fetched_ranges("USDJPY", Timeframe.M1)) == 2


def test_ranges_are_scoped_per_timeframe(meta):
    meta.record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert meta.fetched_ranges("USDJPY", Timeframe.M5) == []


def test_latest_quality_returns_most_recent(meta):
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 100})
    meta.record_quality("USDJPY", Timeframe.M1, {"actual_bars": 200})
    assert meta.latest_quality("USDJPY", Timeframe.M1)["actual_bars"] == 200


def test_latest_quality_none_when_absent(meta):
    assert meta.latest_quality("USDJPY", Timeframe.M1) is None


def test_oos_unlock_is_recorded(meta):
    meta.record_oos_unlock("oos", "戦略v1.2の最終確認")
    unlocks = meta.oos_unlocks()
    assert len(unlocks) == 1
    assert unlocks[0]["period"] == "oos"
    assert unlocks[0]["reason"] == "戦略v1.2の最終確認"
    assert unlocks[0]["at"] is not None


def test_reopening_same_db_keeps_data(tmp_path):
    path = tmp_path / "meta.db"
    Meta(path).record_fetch("USDJPY", Timeframe.M1, ts("2026-01-05"), ts("2026-01-06"))
    assert len(Meta(path).fetched_ranges("USDJPY", Timeframe.M1)) == 1
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_meta.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.storage.meta'`

- [ ] **Step 3: `src/aitrading/storage/meta.py` を実装する**

```python
"""SQLite のメタデータ。取得済み区間・品質レポート・OOS解除ログ。

将来ここにトレードログ・戦略バージョン・AI判断ログ（仕様書§15/§16/§19）が乗る。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from aitrading.timeutil import Timeframe

SCHEMA = """
CREATE TABLE IF NOT EXISTS fetch_ranges (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    start_at   TEXT NOT NULL,
    end_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS quality_reports (
    symbol     TEXT NOT NULL,
    timeframe  TEXT NOT NULL,
    created_at TEXT NOT NULL,
    payload    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS oos_unlocks (
    period     TEXT NOT NULL,
    reason     TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Meta:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def record_fetch(
        self, symbol: str, timeframe: Timeframe, start: pd.Timestamp, end: pd.Timestamp
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fetch_ranges (symbol, timeframe, start_at, end_at)"
                " VALUES (?, ?, ?, ?)",
                (symbol, timeframe.value, str(pd.Timestamp(start)), str(pd.Timestamp(end))),
            )

    def fetched_ranges(
        self, symbol: str, timeframe: Timeframe
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """取得済み区間。隣接・重複はマージして返す。

        マージしておくと「どこがまだ無いか」の差分計算が素直に書ける。
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT start_at, end_at FROM fetch_ranges"
                " WHERE symbol = ? AND timeframe = ? ORDER BY start_at",
                (symbol, timeframe.value),
            ).fetchall()

        merged: list[list[pd.Timestamp]] = []
        for row in rows:
            start, end = pd.Timestamp(row["start_at"]), pd.Timestamp(row["end_at"])
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def record_quality(self, symbol: str, timeframe: Timeframe, report: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO quality_reports (symbol, timeframe, created_at, payload)"
                " VALUES (?, ?, ?, ?)",
                (symbol, timeframe.value, _now(), json.dumps(report, default=str)),
            )

    def latest_quality(self, symbol: str, timeframe: Timeframe) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM quality_reports"
                " WHERE symbol = ? AND timeframe = ?"
                " ORDER BY rowid DESC LIMIT 1",
                (symbol, timeframe.value),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def record_oos_unlock(self, period: str, reason: str) -> None:
        """ロック期間を覗いたことを記録に残す。

        一度OOSの結果を見てしまったら、そのOOSはもうOOSではない。
        後から「いつ・なぜ覗いたか」を追えないと、検証の信頼性が主張できない。
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO oos_unlocks (period, reason, created_at) VALUES (?, ?, ?)",
                (period, reason, _now()),
            )

    def oos_unlocks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT period, reason, created_at FROM oos_unlocks ORDER BY rowid"
            ).fetchall()
        return [
            {"period": r["period"], "reason": r["reason"], "at": r["created_at"]}
            for r in rows
        ]
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_meta.py -v`
Expected: PASS（8 passed）

- [ ] **Step 5: コミット**

```bash
git add src/aitrading/storage/meta.py tests/test_meta.py
git commit -m "SQLiteメタデータを実装

取得済み区間は隣接・重複をマージして返す。「どこがまだ無いか」の差分計算が
素直に書けるようにするため。

OOS期間の解除を理由つきで記録に残す。一度OOSの結果を見てしまったらそのOOSは
もうOOSではないので、後から「いつ・なぜ覗いたか」を追えないと検証の信頼性を
主張できない。"
```

---

## Task 7: リサンプル（上位足・日足2系統）

先読み防止の第3層。生成した足の `close_time` を正しく置くことが要点。

**Files:**
- Create: `src/aitrading/bars.py`
- Create: `tests/test_bars.py`

**Interfaces:**
- Consumes: `Timeframe`, `trading_day_start`, `validate_bars`, `BAR_COLUMNS`
- Produces:
  - `resample(bars_1m: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame` — 入力は `open_time` を index にした1分足（`Lake.load` の返り値の形）。返り値も同じ形。

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_bars.py`:

```python
import pandas as pd
import pytest

from aitrading.bars import resample
from aitrading.timeutil import Timeframe


def minute_bars(start: str, periods: int) -> pd.DataFrame:
    open_time = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    n = len(open_time)
    body = {
        "close_time": open_time + pd.Timedelta(minutes=1),
        "bid_open": [150.0 + i * 0.01 for i in range(n)],
        "bid_high": [150.5 + i * 0.01 for i in range(n)],
        "bid_low": [149.5 + i * 0.01 for i in range(n)],
        "bid_close": [150.2 + i * 0.01 for i in range(n)],
        "volume": [10.0] * n,
    }
    for field in ("open", "high", "low", "close"):
        body[f"ask_{field}"] = [v + 0.02 for v in body[f"bid_{field}"]]
    return pd.DataFrame(body, index=open_time).rename_axis("open_time")


def test_m5_aggregates_ohlc_correctly():
    got = resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M5)
    assert len(got) == 2
    first = got.iloc[0]
    assert first["bid_open"] == pytest.approx(150.00)   # 1本目の open
    assert first["bid_close"] == pytest.approx(150.24)  # 5本目の close
    assert first["bid_high"] == pytest.approx(150.54)   # 5本の最大
    assert first["bid_low"] == pytest.approx(149.50)    # 5本の最小
    assert first["volume"] == pytest.approx(50.0)


def test_close_time_is_end_of_period():
    """5分足が使えるようになるのは5分が終わった瞬間。ここを誤ると先読みになる。"""
    got = resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M5)
    assert got.index[0] == pd.Timestamp("2026-01-05 00:00", tz="UTC")
    assert got.iloc[0]["close_time"] == pd.Timestamp("2026-01-05 00:05", tz="UTC")


def test_drops_incomplete_trailing_period():
    """埋まりきっていない最後の期間は返さない（第1層: 確定足しか出さない）。"""
    got = resample(minute_bars("2026-01-05 00:00", 7), Timeframe.M5)
    assert len(got) == 1


def test_h1_and_h4_aggregate():
    bars = minute_bars("2026-01-05 00:00", 60 * 8)
    assert len(resample(bars, Timeframe.H1)) == 8
    assert len(resample(bars, Timeframe.H4)) == 2


def test_daily_ny_and_jst_boundaries_differ():
    """同じ1分足から2通りの日足ができ、区切りが違う。"""
    bars = minute_bars("2026-01-05 00:00", 60 * 48)
    ny = resample(bars, Timeframe.D1_NY)
    jst = resample(bars, Timeframe.D1_JST)
    assert set(ny.index) != set(jst.index)


def test_daily_ny_boundary_is_22utc_in_winter():
    bars = minute_bars("2026-01-05 00:00", 60 * 48)
    ny = resample(bars, Timeframe.D1_NY)
    assert ny.index[0].hour == 22  # 冬時間の NY 17:00


def test_daily_jst_boundary_is_15utc():
    bars = minute_bars("2026-01-05 00:00", 60 * 48)
    jst = resample(bars, Timeframe.D1_JST)
    assert jst.index[0].hour == 15  # JST 00:00


def test_resample_is_deterministic():
    """同じ入力から必ず同じ出力。1分足から再生成すれば同じものができる。"""
    bars = minute_bars("2026-01-05 00:00", 100)
    pd.testing.assert_frame_equal(
        resample(bars, Timeframe.M15), resample(bars, Timeframe.M15)
    )


def test_rejects_m1_as_target():
    with pytest.raises(ValueError, match="1m"):
        resample(minute_bars("2026-01-05 00:00", 10), Timeframe.M1)
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_bars.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.bars'`

- [ ] **Step 3: `src/aitrading/bars.py` を実装する**

```python
"""1分足から上位足・日足2系統を生成する。

取得するのは1分足だけで、他はすべてここで作る生成物。
close_time を「その期間が終わった時刻」に置くのが要点で、
ここを誤るとマルチタイムフレーム解析が先読みになる。
"""

from __future__ import annotations

import pandas as pd

from aitrading.timeutil import Timeframe, trading_day_start

_AGGREGATION = {
    "bid_open": "first", "bid_high": "max", "bid_low": "min", "bid_close": "last",
    "ask_open": "first", "ask_high": "max", "ask_low": "min", "ask_close": "last",
    "volume": "sum",
}

#: 生成先の期間に何本の1分足が入るはずか（完全な期間の判定用）
_EXPECTED_MINUTES = {
    Timeframe.M5: 5,
    Timeframe.M15: 15,
    Timeframe.H1: 60,
    Timeframe.H4: 240,
}


def resample(bars_1m: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """1分足を上位足に集約する。index は open_time。"""
    if timeframe is Timeframe.M1:
        raise ValueError("1m はデータソースから取得するもので、生成対象ではない")
    if bars_1m.empty:
        return bars_1m.copy()

    index = pd.DatetimeIndex(bars_1m.index)
    if index.tz is None:
        raise ValueError("index が tz-aware でない。UTCで渡すこと")

    if timeframe.convention is not None:
        group = trading_day_start(index, timeframe.convention)
        if timeframe in (Timeframe.W1_NY, Timeframe.W1_JST):
            # 週足は「その日が属する週の最初の取引日開始」でまとめる
            group = pd.DatetimeIndex(group).to_period("W").start_time
            group = pd.DatetimeIndex(group).tz_localize("UTC")
        grouped = bars_1m.groupby(pd.Index(group, name="open_time"))
        out = grouped.agg(_AGGREGATION)
        # 日足・週足は長さが可変なので、次の期間の開始を close_time にする
        starts = pd.DatetimeIndex(out.index)
        out["close_time"] = list(starts[1:]) + [
            pd.DatetimeIndex(bars_1m["close_time"]).max()
        ]
        # 最後の期間は途中で切れている可能性が高いので落とす
        out = out.iloc[:-1]
    else:
        delta = timeframe.delta
        counts = bars_1m["volume"].resample(delta, label="left", closed="left").count()
        out = bars_1m.resample(delta, label="left", closed="left").agg(_AGGREGATION)
        out = out.loc[counts == _EXPECTED_MINUTES[timeframe]]
        out["close_time"] = pd.DatetimeIndex(out.index) + delta

    out = out.dropna(subset=["bid_open"])
    ordered = ["close_time"] + list(_AGGREGATION)
    return out.loc[:, ordered].rename_axis("open_time")
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_bars.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: コミット**

```bash
git add src/aitrading/bars.py tests/test_bars.py
git commit -m "1分足から上位足と日足2系統を生成するリサンプルを実装（先読み防止 第3層）

close_time を「その期間が終わった時刻」に置く。5分足が使えるようになるのは
5分が終わった瞬間であり、ここを誤るとマルチタイムフレーム解析が先読みになる。

埋まりきっていない最後の期間は返さない（確定足しか出さない）。

日足は NY基準（17:00 America/New_York）と JST基準（00:00 Asia/Tokyo）の
2系統を同じ1分足から生成する。区切りが違うのでテストで確認している。"
```

---

## Task 8: データ品質チェック

**Files:**
- Create: `src/aitrading/quality.py`
- Create: `tests/test_quality.py`

**Interfaces:**
- Consumes: `Timeframe`, `is_market_open`
- Produces:
  - `@dataclass class QualityReport` — 属性 `symbol: str`, `timeframe: str`, `expected_bars: int`, `actual_bars: int`, `gaps: list[dict]`, `longest_gap_minutes: float`, `duplicate_count: int`, `bad_spread_count: int`, `wide_spread_count: int`, `price_jump_count: int`
  - `QualityReport.to_dict() -> dict`
  - `check(bars: pd.DataFrame, symbol: str, timeframe: Timeframe, *, jump_atr_multiple: float = 10.0, wide_spread_quantile: float = 0.999) -> QualityReport`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_quality.py`:

```python
import pandas as pd
import pytest

from aitrading.quality import check
from aitrading.timeutil import Timeframe

from tests.test_bars import minute_bars


def test_clean_data_has_no_gaps():
    # 月曜 00:00 UTC から4時間。市場は開いている。
    report = check(minute_bars("2026-01-05 00:00", 240), "USDJPY", Timeframe.M1)
    assert report.gaps == []
    assert report.actual_bars == 240
    assert report.expected_bars == 240


def test_detects_missing_bars():
    bars = minute_bars("2026-01-05 00:00", 240)
    bars = bars.drop(bars.index[100:130])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert len(report.gaps) == 1
    assert report.longest_gap_minutes == pytest.approx(30.0)
    assert report.actual_bars == 210


def test_weekend_is_not_counted_as_a_gap():
    """週末クローズを欠損と数えると、品質チェックが毎週偽陽性を出す。"""
    # 金曜 20:00 UTC から月曜まで（間の週末は市場クローズ）
    friday = minute_bars("2026-01-09 20:00", 120)
    monday = minute_bars("2026-01-12 00:00", 120)
    report = check(pd.concat([friday, monday]), "USDJPY", Timeframe.M1)
    assert report.gaps == []


def test_detects_zero_and_negative_spread():
    bars = minute_bars("2026-01-05 00:00", 60)
    bars.loc[bars.index[5], "ask_close"] = bars.loc[bars.index[5], "bid_close"]
    bars.loc[bars.index[6], "ask_close"] = bars.loc[bars.index[6], "bid_close"] - 0.01
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.bad_spread_count == 2


def test_detects_price_jump():
    bars = minute_bars("2026-01-05 00:00", 240)
    bars.loc[bars.index[120], "bid_close"] += 50.0
    bars.loc[bars.index[120], "ask_close"] += 50.0
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.price_jump_count >= 1


def test_detects_duplicates():
    bars = minute_bars("2026-01-05 00:00", 60)
    bars = pd.concat([bars, bars.iloc[[10]]])
    report = check(bars, "USDJPY", Timeframe.M1)
    assert report.duplicate_count == 1


def test_to_dict_is_json_serializable():
    import json

    report = check(minute_bars("2026-01-05 00:00", 60), "USDJPY", Timeframe.M1)
    json.dumps(report.to_dict(), default=str)
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.quality'`

- [ ] **Step 3: `src/aitrading/quality.py` を実装する**

```python
"""データ品質チェック。取得直後に走らせて meta.db に記録する。

週末クローズを欠損と区別することが要点。区別しないと毎週末に偽陽性が出て、
アラートとして機能しなくなる。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from aitrading.timeutil import Timeframe, is_market_open


@dataclass
class QualityReport:
    symbol: str
    timeframe: str
    expected_bars: int
    actual_bars: int
    duplicate_count: int
    bad_spread_count: int
    wide_spread_count: int
    price_jump_count: int
    longest_gap_minutes: float
    gaps: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _mid_close(bars: pd.DataFrame) -> pd.Series:
    return (bars["bid_close"] + bars["ask_close"]) / 2.0


def check(
    bars: pd.DataFrame,
    symbol: str,
    timeframe: Timeframe,
    *,
    jump_atr_multiple: float = 10.0,
    wide_spread_quantile: float = 0.999,
) -> QualityReport:
    index = pd.DatetimeIndex(bars.index)
    duplicate_count = int(index.duplicated().sum())
    bars = bars.loc[~index.duplicated(keep="first")].sort_index()
    index = pd.DatetimeIndex(bars.index)

    # --- 欠損区間（市場が開いている時間だけを対象にする）---
    gaps: list[dict] = []
    step = timeframe.delta or pd.Timedelta(days=1)
    if len(index) > 1:
        deltas = index.to_series().diff()
        for at, delta in deltas.items():
            if pd.isna(delta) or delta <= step:
                continue
            span = pd.date_range(at - delta + step, at - step, freq=step)
            missing = int(is_market_open(span).sum()) if len(span) else 0
            if missing > 0:
                gaps.append(
                    {
                        "from": str(at - delta + step),
                        "to": str(at - step),
                        "missing_bars": missing,
                        "minutes": missing * step.total_seconds() / 60.0,
                    }
                )
    longest_gap = max((g["minutes"] for g in gaps), default=0.0)

    # --- 想定本数（市場が開いていた分だけ）---
    if len(index) > 1:
        full = pd.date_range(index.min(), index.max(), freq=step)
        expected = int(is_market_open(full).sum())
    else:
        expected = len(index)

    # --- スプレッド ---
    spread = bars["ask_close"] - bars["bid_close"]
    bad_spread = int((spread <= 0).sum())
    threshold = spread.loc[spread > 0].quantile(wide_spread_quantile)
    wide_spread = int((spread > threshold).sum()) if np.isfinite(threshold) else 0

    # --- 価格ジャンプ（ATR相当の何倍か）---
    mid = _mid_close(bars)
    true_range = (bars["bid_high"] - bars["bid_low"]).abs()
    atr = true_range.rolling(14, min_periods=14).mean()
    jump = mid.diff().abs()
    price_jumps = int((jump > atr * jump_atr_multiple).sum())

    return QualityReport(
        symbol=symbol,
        timeframe=timeframe.value,
        expected_bars=expected,
        actual_bars=len(bars),
        duplicate_count=duplicate_count,
        bad_spread_count=bad_spread,
        wide_spread_count=wide_spread,
        price_jump_count=price_jumps,
        longest_gap_minutes=float(longest_gap),
        gaps=gaps,
    )
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_quality.py -v`
Expected: PASS（7 passed）

- [ ] **Step 5: コミット**

```bash
git add src/aitrading/quality.py tests/test_quality.py
git commit -m "データ品質チェックを実装

欠損区間・重複・スプレッド異常・価格ジャンプを検出して QualityReport にまとめる。

週末クローズを欠損と区別するのが要点。is_market_open で市場が開いていた分だけを
想定本数に数える。区別しないと毎週末に偽陽性が出て、アラートとして機能しなくなる。"
```

---

## Task 9: 指標レジストリとトランケーション不変性テスト

先読み防止の第2層。**このタスクが Phase 0 で最も重要**。レジストリに登録するだけで先読み検査が自動適用される仕組みを作る。

**Files:**
- Create: `src/aitrading/indicators/__init__.py`
- Create: `src/aitrading/indicators/registry.py`
- Create: `src/aitrading/indicators/core.py`
- Create: `tests/conftest.py`
- Create: `tests/test_indicators_lookahead.py`
- Create: `tests/test_indicators_values.py`

**Interfaces:**
- Consumes: なし（バーの列名のみ）
- Produces:
  - `INDICATORS: dict[str, Callable[[pd.DataFrame], pd.Series | pd.DataFrame]]`
  - `indicator(name: str)` — レジストリ登録デコレータ
  - `mid(bars: pd.DataFrame, field: str) -> pd.Series`
  - 指標本体（すべて第1引数が `bars: pd.DataFrame`）: `sma`, `ema`, `rsi`, `macd`, `atr`, `bbands`, `vwap`, `donchian`, `hist_vol`
  - `macd` と `bbands` と `donchian` は `pd.DataFrame` を返す。他は `pd.Series`。

- [ ] **Step 1: レジストリと共通フィクスチャを作る**

`src/aitrading/indicators/__init__.py`:

```python
from aitrading.indicators.core import *  # noqa: F401,F403 — レジストリへの登録を発火させる
from aitrading.indicators.registry import INDICATORS, indicator  # noqa: F401
```

`src/aitrading/indicators/registry.py`:

```python
"""指標レジストリ。

登録された指標には、テスト側からトランケーション不変性検査が自動で適用される。
新しい指標を足したとき、テストを書き忘れても先読み検査だけは必ず走る。
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

Indicator = Callable[..., "pd.Series | pd.DataFrame"]

INDICATORS: dict[str, Indicator] = {}


def indicator(name: str) -> Callable[[Indicator], Indicator]:
    def register(fn: Indicator) -> Indicator:
        if name in INDICATORS:
            raise ValueError(f"指標名が重複している: {name!r}")
        INDICATORS[name] = fn
        return fn

    return register


def mid(bars: pd.DataFrame, field: str) -> pd.Series:
    """Bid と Ask の中間値。指標はミッドで計算し、コストは約定側で扱う。"""
    return (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0
```

`tests/conftest.py`:

```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_bars() -> pd.DataFrame:
    """再現可能な合成バー。指標テスト全体で共有する。"""
    rng = np.random.default_rng(20260824)
    n = 300
    open_time = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    walk = 150.0 + np.cumsum(rng.normal(0, 0.02, n))

    body = {"close_time": open_time + pd.Timedelta(minutes=1)}
    body["bid_close"] = walk
    body["bid_open"] = np.concatenate([[walk[0]], walk[:-1]])
    body["bid_high"] = np.maximum(body["bid_open"], body["bid_close"]) + 0.03
    body["bid_low"] = np.minimum(body["bid_open"], body["bid_close"]) - 0.03
    for f in ("open", "high", "low", "close"):
        body[f"ask_{f}"] = body[f"bid_{f}"] + 0.02
    body["volume"] = rng.uniform(50, 150, n)

    return pd.DataFrame(body, index=open_time).rename_axis("open_time")
```

- [ ] **Step 2: 先読み検査のテストを書く（まだ指標が無いので失敗する）**

`tests/test_indicators_lookahead.py`:

```python
"""全指標へのトランケーション不変性検査。

入力の末尾を切り落としても、残った部分の出力が1つも変わらないこと。
先読みしている指標はこれで必ず落ちる。center=True のローリング窓、
将来のバーを見た正規化、全期間統計を使った標準化は、すべて検出される。
"""

import pandas as pd
import pytest

from aitrading.indicators import INDICATORS

TRUNCATE = 10


def assert_same_prefix(full, truncated) -> None:
    head = full.iloc[: len(truncated)]
    if isinstance(full, pd.DataFrame):
        pd.testing.assert_frame_equal(head, truncated)
    else:
        pd.testing.assert_series_equal(head, truncated)


def test_registry_is_not_empty():
    assert INDICATORS, "指標が1つも登録されていない"


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_look_ahead(name, sample_bars):
    fn = INDICATORS[name]
    full = fn(sample_bars)
    truncated = fn(sample_bars.iloc[:-TRUNCATE])
    assert_same_prefix(full, truncated)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_preserves_index(name, sample_bars):
    result = INDICATORS[name](sample_bars)
    pd.testing.assert_index_equal(result.index, sample_bars.index)


@pytest.mark.parametrize("name", sorted(INDICATORS))
def test_indicator_does_not_mutate_input(name, sample_bars):
    before = sample_bars.copy(deep=True)
    INDICATORS[name](sample_bars)
    pd.testing.assert_frame_equal(sample_bars, before)


def test_detector_catches_a_deliberate_lookahead(sample_bars):
    """検査そのものが機能していることを確かめる。"""

    def cheating(bars: pd.DataFrame) -> pd.Series:
        # 中央寄せの窓は未来のバーを見る
        return bars["bid_close"].rolling(5, center=True, min_periods=1).mean()

    with pytest.raises(AssertionError):
        assert_same_prefix(cheating(sample_bars), cheating(sample_bars.iloc[:-TRUNCATE]))
```

- [ ] **Step 3: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_indicators_lookahead.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.indicators'`（レジストリだけ作った状態なら `test_registry_is_not_empty` が FAIL）

- [ ] **Step 4: `src/aitrading/indicators/core.py` を実装する**

```python
"""テクニカル指標。すべて「時点tまでの情報しか使わない」純関数。

TA-Lib を使わないのは移植性のためだけではない。全指標をレジストリに集め、
トランケーション不変性検査を機械的に適用できるようにするため。
外部ライブラリではこの保証を自分たちで持てない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aitrading.indicators.registry import indicator, mid

__all__ = ["sma", "ema", "rsi", "macd", "atr", "bbands", "vwap", "donchian", "hist_vol"]


@indicator("sma")
def sma(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    return mid(bars, "close").rolling(period, min_periods=period).mean().rename("sma")


@indicator("ema")
def ema(bars: pd.DataFrame, period: int = 20) -> pd.Series:
    return mid(bars, "close").ewm(span=period, adjust=False).mean().rename("ema")


@indicator("rsi")
def rsi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = mid(bars, "close").diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder の平滑化。ewm は過去のみを見る
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).rename("rsi")


@indicator("macd")
def macd(
    bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    close = mid(bars, "close")
    line = (
        close.ewm(span=fast, adjust=False).mean()
        - close.ewm(span=slow, adjust=False).mean()
    )
    signal_line = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": line, "signal": signal_line, "histogram": line - signal_line}
    )


@indicator("atr")
def atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low = mid(bars, "high"), mid(bars, "low")
    prev_close = mid(bars, "close").shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return (
        true_range.ewm(alpha=1 / period, adjust=False, min_periods=period)
        .mean()
        .rename("atr")
    )


@indicator("bbands")
def bbands(bars: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    close = mid(bars, "close")
    middle = close.rolling(period, min_periods=period).mean()
    spread = close.rolling(period, min_periods=period).std(ddof=0) * num_std
    return pd.DataFrame(
        {"lower": middle - spread, "middle": middle, "upper": middle + spread}
    )


@indicator("vwap")
def vwap(bars: pd.DataFrame) -> pd.Series:
    """当日の累積VWAP。日境界はNY基準。

    全期間の合計で割ると未来を見ることになるので、必ず累積和で計算する。
    """
    from aitrading.timeutil import trading_day_start

    typical = (mid(bars, "high") + mid(bars, "low") + mid(bars, "close")) / 3.0
    day = pd.Series(trading_day_start(pd.DatetimeIndex(bars.index), "ny"), index=bars.index)
    volume = bars["volume"]
    cum_pv = (typical * volume).groupby(day).cumsum()
    cum_v = volume.groupby(day).cumsum()
    return (cum_pv / cum_v).rename("vwap")


@indicator("donchian")
def donchian(bars: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    upper = mid(bars, "high").rolling(period, min_periods=period).max()
    lower = mid(bars, "low").rolling(period, min_periods=period).min()
    return pd.DataFrame({"lower": lower, "upper": upper, "width": upper - lower})


@indicator("hist_vol")
def hist_vol(bars: pd.DataFrame, period: int = 60) -> pd.Series:
    returns = np.log(mid(bars, "close")).diff()
    return returns.rolling(period, min_periods=period).std(ddof=0).rename("hist_vol")
```

- [ ] **Step 5: 先読み検査を実行して通ることを確認する**

Run: `uv run pytest tests/test_indicators_lookahead.py -v`
Expected: PASS（1 + 9×3 + 1 = 29 passed）

- [ ] **Step 6: 参照値照合テストを書く**

`tests/test_indicators_values.py`:

```python
"""指標の正しさを既存ライブラリ（ta）の出力と照合する。

ここが狂うと以降の分析が全部崩れる。照合は開発時の1回で、
以降は ta に依存しない（ta は dev 依存にのみ入っている）。
"""

import numpy as np
import pandas as pd
import pytest

from aitrading.indicators.core import atr, bbands, ema, macd, rsi, sma
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
        ema(sample_bars).to_numpy()[30:], expected.to_numpy()[30:], rtol=1e-6
    )


def test_rsi_matches_reference(sample_bars):
    from ta.momentum import RSIIndicator

    close = mid(sample_bars, "close")
    expected = RSIIndicator(close, window=14).rsi()
    np.testing.assert_allclose(
        rsi(sample_bars).to_numpy()[30:], expected.to_numpy()[30:], rtol=1e-6
    )


def test_macd_matches_reference(sample_bars):
    from ta.trend import MACD

    close = mid(sample_bars, "close")
    reference = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    got = macd(sample_bars)
    np.testing.assert_allclose(
        got["macd"].to_numpy()[40:], reference.macd().to_numpy()[40:], rtol=1e-6
    )
    np.testing.assert_allclose(
        got["signal"].to_numpy()[40:], reference.macd_signal().to_numpy()[40:], rtol=1e-6
    )


def test_atr_matches_reference(sample_bars):
    from ta.volatility import AverageTrueRange

    reference = AverageTrueRange(
        high=mid(sample_bars, "high"),
        low=mid(sample_bars, "low"),
        close=mid(sample_bars, "close"),
        window=14,
    ).average_true_range()
    np.testing.assert_allclose(
        atr(sample_bars).to_numpy()[30:], reference.to_numpy()[30:], rtol=1e-6
    )


def test_bbands_matches_reference(sample_bars):
    from ta.volatility import BollingerBands

    reference = BollingerBands(mid(sample_bars, "close"), window=20, window_dev=2)
    got = bbands(sample_bars)
    np.testing.assert_allclose(
        got["upper"].to_numpy()[25:], reference.bollinger_hband().to_numpy()[25:], rtol=1e-9
    )
    np.testing.assert_allclose(
        got["lower"].to_numpy()[25:], reference.bollinger_lband().to_numpy()[25:], rtol=1e-9
    )


def test_rsi_bounds(sample_bars):
    values = rsi(sample_bars).dropna()
    assert values.between(0, 100).all()


def test_bbands_ordering(sample_bars):
    got = bbands(sample_bars).dropna()
    assert (got["lower"] <= got["middle"]).all()
    assert (got["middle"] <= got["upper"]).all()
```

- [ ] **Step 7: 参照値照合を実行して通ることを確認する**

Run: `uv run pytest tests/test_indicators_values.py -v`
Expected: PASS（8 passed）

数値が合わない場合、`ta` 側の平滑化方式（SMA平均 vs Wilder）が違う可能性がある。**`ta` に合わせて自前実装を変えるのではなく、まずどちらが正しいかを確認すること。** RSI と ATR は Wilder の平滑化（`alpha=1/period`）が標準。

- [ ] **Step 8: コミット**

```bash
git add src/aitrading/indicators/ tests/conftest.py tests/test_indicators_lookahead.py tests/test_indicators_values.py
git commit -m "指標レジストリと9指標を実装、先読み検査を全指標に自動適用（第2層）

レジストリに登録された全指標へ、トランケーション不変性検査を
パラメトライズで自動適用する。入力の末尾を切り落としても残りの出力が
変わらないことを検査するので、center=True の窓・将来を見た正規化・
全期間統計での標準化はすべて落ちる。新しい指標を足したとき、テストを
書き忘れても先読み検査だけは必ず走る。

検査そのものが機能していることを、わざと先読みする関数を用意して確認している。

TA-Libを使わない理由は移植性ではなく、この保証を自分たちで持つため。
値の正しさは ta ライブラリと照合してテストで固定した。"
```

---

## Task 10: 期待値スキャン

バックテストエンジンではない。TP/SL もポジション管理も持たない純粋な統計。

**Files:**
- Create: `src/aitrading/edge_scan.py`
- Create: `tests/test_edge_scan.py`

**Interfaces:**
- Consumes: `Settings`（`slice_bars`）, `session_labels`, `Meta`
- Produces:
  - `@dataclass class HorizonStats` — `horizon: int`, `n: int`, `mean_pips: float`, `median_pips: float`, `win_rate: float`, `std_pips: float`, `ci95_pips: tuple[float, float]`
  - `@dataclass class EdgeResult` — `n_signals: int`, `horizons: list[HorizonStats]`, `by_group: dict[str, list[HorizonStats]]`, `to_dict() -> dict`
  - `scan(bars, signal, *, horizons=(1,5,15,30,60), direction="long", pip=0.01, deduct_spread=True, group_by=None) -> EdgeResult`
  - `scan_period(bars, signal, settings, period, *, meta=None, unlock_reason=None, **kwargs) -> EdgeResult` — ロック期間は `unlock_reason` 必須、指定時は `meta.record_oos_unlock` を呼ぶ

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_edge_scan.py`:

```python
import numpy as np
import pandas as pd
import pytest

from aitrading.config import load_settings
from aitrading.edge_scan import scan, scan_period
from aitrading.storage.meta import Meta


@pytest.fixture
def rising_bars():
    """1分ごとに mid が +1pip ずつ上がる、スプレッド2pip固定のバー。"""
    n = 200
    index = pd.date_range("2026-01-05 00:00", periods=n, freq="1min", tz="UTC")
    mid_price = 150.00 + np.arange(n) * 0.01
    body = {"close_time": index + pd.Timedelta(minutes=1), "volume": [100.0] * n}
    for f in ("open", "high", "low", "close"):
        body[f"bid_{f}"] = mid_price - 0.01
        body[f"ask_{f}"] = mid_price + 0.01
    return pd.DataFrame(body, index=index).rename_axis("open_time")


def test_computes_mean_return_per_horizon(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10, 20, 30]] = True
    result = scan(rising_bars, signal, horizons=(5,), deduct_spread=False)
    stats = result.horizons[0]
    assert stats.n == 3
    # 5分で mid が 5pip 上がる
    assert stats.mean_pips == pytest.approx(5.0, abs=1e-6)
    assert stats.win_rate == pytest.approx(1.0)


def test_spread_is_deducted(rising_bars):
    """生のリターンで見ると、ほとんどの指標が有望に見えてしまう。"""
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True
    gross = scan(rising_bars, signal, horizons=(5,), deduct_spread=False).horizons[0]
    net = scan(rising_bars, signal, horizons=(5,), deduct_spread=True).horizons[0]
    # 買いはAsk、決済はBid。往復で2pip分不利になる
    assert net.mean_pips == pytest.approx(gross.mean_pips - 2.0, abs=1e-6)


def test_short_direction_flips_sign(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True
    result = scan(rising_bars, signal, horizons=(5,), direction="short", deduct_spread=False)
    assert result.horizons[0].mean_pips == pytest.approx(-5.0, abs=1e-6)


def test_signals_without_enough_future_bars_are_dropped(rising_bars):
    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[len(rising_bars) - 2]] = True
    result = scan(rising_bars, signal, horizons=(60,), deduct_spread=False)
    assert result.horizons[0].n == 0


def test_group_by_session_splits_results(rising_bars):
    signal = pd.Series(True, index=rising_bars.index)
    result = scan(rising_bars, signal, horizons=(5,), group_by="session")
    assert result.by_group, "セッション別の集計が空"
    assert all(isinstance(k, str) for k in result.by_group)


def test_scan_period_refuses_locked_period_without_reason(rising_bars):
    settings = load_settings()
    signal = pd.Series(True, index=rising_bars.index)
    with pytest.raises(PermissionError, match="oos"):
        scan_period(rising_bars, signal, settings, "oos")


def test_scan_period_records_unlock(tmp_path, rising_bars):
    settings = load_settings()
    meta = Meta(tmp_path / "meta.db")
    signal = pd.Series(True, index=rising_bars.index)
    scan_period(
        rising_bars, signal, settings, "oos",
        meta=meta, unlock_reason="戦略v1.2の最終確認",
    )
    unlocks = meta.oos_unlocks()
    assert len(unlocks) == 1
    assert unlocks[0]["reason"] == "戦略v1.2の最終確認"


def test_scan_period_on_training_needs_no_reason(rising_bars):
    settings = load_settings()
    signal = pd.Series(True, index=rising_bars.index)
    result = scan_period(rising_bars, signal, settings, "training")
    # rising_bars は2026年なので training(〜2021) には入らない
    assert result.n_signals == 0


def test_to_dict_is_json_serializable(rising_bars):
    import json

    signal = pd.Series(False, index=rising_bars.index)
    signal.iloc[[10]] = True
    json.dumps(scan(rising_bars, signal, horizons=(5,)).to_dict(), default=str)
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_edge_scan.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aitrading.edge_scan'`

- [ ] **Step 3: `src/aitrading/edge_scan.py` を実装する**

```python
"""期待値スキャン。バックテストエンジンではない。

TP/SL もポジション管理も資金管理も持たない。「そもそもこの条件に優位性の種が
あるか」を見るための純粋な統計。バックテストエンジンを作る前に使う。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from aitrading.config import Settings
from aitrading.storage.meta import Meta
from aitrading.timeutil import session_labels

DEFAULT_HORIZONS = (1, 5, 15, 30, 60)


@dataclass
class HorizonStats:
    horizon: int
    n: int
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
        return asdict(self)


def _stats(returns: pd.Series, horizon: int) -> HorizonStats:
    values = returns.dropna().to_numpy()
    n = len(values)
    if n == 0:
        return HorizonStats(horizon, 0, np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan))

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    half = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return HorizonStats(
        horizon=horizon,
        n=n,
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
    """シグナル発生足の確定後にエントリーし、horizon本後に決済した場合のpips。"""
    if direction == "long":
        # 買いはAsk、決済はBid
        entry = bars["ask_close"] if deduct_spread else bars["bid_close"]
        exit_price = bars["bid_close"].shift(-horizon)
        gross = exit_price - entry
    elif direction == "short":
        entry = bars["bid_close"] if deduct_spread else bars["ask_close"]
        exit_price = bars["ask_close"].shift(-horizon)
        gross = entry - exit_price
    else:
        raise ValueError(f"未知の方向: {direction!r}（'long' か 'short'）")

    if not deduct_spread:
        # ミッド同士で比較する（コスト無視）
        mid_now = (bars["bid_close"] + bars["ask_close"]) / 2.0
        mid_future = mid_now.shift(-horizon)
        gross = mid_future - mid_now if direction == "long" else mid_now - mid_future

    return (gross / pip).where(signal.astype(bool))


def scan(
    bars: pd.DataFrame,
    signal: pd.Series,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    direction: str = "long",
    pip: float = 0.01,
    deduct_spread: bool = True,
    group_by: str | None = None,
) -> EdgeResult:
    """シグナル後のリターン分布を集計する。

    deduct_spread=True が既定。生のリターンで見るとほとんどの条件が有望に見え、
    取引コストを引くと消える。最初からコスト込みで見る。
    """
    signal = signal.reindex(bars.index).fillna(False).astype(bool)

    result = EdgeResult(
        n_signals=int(signal.sum()), deduct_spread=deduct_spread, direction=direction
    )
    for horizon in horizons:
        returns = _returns(bars, signal, horizon, direction, pip, deduct_spread)
        result.horizons.append(_stats(returns, horizon))

    if group_by == "session":
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
    elif group_by is not None:
        raise ValueError(f"未対応の層別: {group_by!r}（現在は 'session' のみ）")

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

    人間が一度OOSの結果を見てしまったら、そのOOSはもうOOSではない。
    覗いた事実を meta.db に残しておかないと、後から検証の信頼性を主張できない。
    """
    target = settings.period_for(period)
    if target.locked:
        if not unlock_reason:
            raise PermissionError(
                f"期間 {period!r} はロックされている。"
                " 集計するには unlock_reason を明示すること"
            )
        if meta is not None:
            meta.record_oos_unlock(period, unlock_reason)

    sliced = settings.slice_bars(bars, period, allow_locked=target.locked)
    return scan(sliced, signal.reindex(sliced.index).fillna(False), **kwargs)
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_edge_scan.py -v`
Expected: PASS（9 passed）

- [ ] **Step 5: コミット**

```bash
git add src/aitrading/edge_scan.py tests/test_edge_scan.py
git commit -m "期待値スキャンを実装（バックテストエンジンではない）

シグナル後 +1/+5/+15/+30/+60 本のリターン分布を集計する。TP/SLもポジション管理も
持たない純粋な統計で、バックテストエンジンを作る前に「そもそも優位性の種があるか」
を見るために使う。

スプレッド控除を既定にした。生のリターンで見るとほとんどの条件が有望に見え、
取引コストを引くと消えるため。

セッション別の層別集計に対応。全期間を平均すると時間帯ごとの差が消える。

ロック期間の集計には unlock_reason が必須で、覗いた事実を meta.db に記録する。"
```

---

## Task 11: 取得CLI

**Files:**
- Create: `scripts/fetch_data.py`
- Create: `scripts/build_bars.py`
- Create: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `load_settings`, `DukascopySource`, `Lake`, `Meta`, `quality.check`, `bars.resample`
- Produces:
  - `scripts/fetch_data.py` — `fetch(settings, source, lake, meta, *, start=None, end=None, chunk_days=30) -> None`、`main(argv=None) -> int`
  - `scripts/build_bars.py` — `build(settings, lake, *, timeframes=None) -> None`、`main(argv=None) -> int`

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_scripts.py`:

```python
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_bars  # noqa: E402
import fetch_data  # noqa: E402

from aitrading.config import Period, Settings  # noqa: E402
from aitrading.storage.lake import Lake  # noqa: E402
from aitrading.storage.meta import Meta  # noqa: E402
from aitrading.timeutil import Timeframe  # noqa: E402

from tests.test_bars import minute_bars


class FakeSource:
    """ネットワークに触らないダミーのデータソース。"""

    def __init__(self):
        self.calls: list[tuple] = []

    def fetch(self, symbol, timeframe, start, end):
        self.calls.append((symbol, timeframe, start, end))
        minutes = int((pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() // 60)
        df = minute_bars(str(pd.Timestamp(start)), max(minutes, 1))
        return df.reset_index()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        symbol="USDJPY",
        data_start=pd.Timestamp("2026-01-05", tz="UTC"),
        data_root=tmp_path,
        meta_db=tmp_path / "meta.db",
        periods={
            "training": Period(
                "training",
                pd.Timestamp("2026-01-01", tz="UTC"),
                pd.Timestamp("2026-12-31", tz="UTC"),
                False,
            )
        },
        models={},
    )


def test_fetch_saves_bars_and_records_range(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    fetch_data.fetch(
        settings, source, lake, meta,
        start=pd.Timestamp("2026-01-05", tz="UTC"),
        end=pd.Timestamp("2026-01-05 02:00", tz="UTC"),
    )
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert not got.empty
    assert meta.fetched_ranges("USDJPY", Timeframe.M1)


def test_fetch_records_quality_report(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    fetch_data.fetch(
        settings, source, lake, meta,
        start=pd.Timestamp("2026-01-05", tz="UTC"),
        end=pd.Timestamp("2026-01-05 02:00", tz="UTC"),
    )
    assert meta.latest_quality("USDJPY", Timeframe.M1) is not None


def test_fetch_is_idempotent(settings):
    lake, meta, source = Lake(settings.data_root), Meta(settings.meta_db), FakeSource()
    window = dict(
        start=pd.Timestamp("2026-01-05", tz="UTC"),
        end=pd.Timestamp("2026-01-05 02:00", tz="UTC"),
    )
    fetch_data.fetch(settings, source, lake, meta, **window)
    first = len(lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC")))
    fetch_data.fetch(settings, source, lake, meta, **window)
    second = len(lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC")))
    assert first == second


def test_build_bars_generates_all_timeframes(settings):
    lake = Lake(settings.data_root)
    lake.save("USDJPY", Timeframe.M1, minute_bars("2026-01-05 00:00", 60 * 48).reset_index())
    build_bars.build(settings, lake)
    as_of = pd.Timestamp("2030-01-01", tz="UTC")
    for tf in (Timeframe.M5, Timeframe.H1, Timeframe.D1_NY, Timeframe.D1_JST):
        assert not lake.load("USDJPY", tf, as_of=as_of).empty, f"{tf} が生成されていない"


def test_generated_5m_matches_direct_resample(settings):
    """1分足から再生成すれば必ず同じものができることを担保する。"""
    from aitrading.bars import resample

    lake = Lake(settings.data_root)
    source_bars = minute_bars("2026-01-05 00:00", 600)
    lake.save("USDJPY", Timeframe.M1, source_bars.reset_index())
    build_bars.build(settings, lake, timeframes=[Timeframe.M5])

    as_of = pd.Timestamp("2030-01-01", tz="UTC")
    stored = lake.load("USDJPY", Timeframe.M5, as_of=as_of)
    expected = resample(source_bars, Timeframe.M5)
    assert len(stored) == len(expected)
    pd.testing.assert_series_equal(
        stored["bid_close"].reset_index(drop=True),
        expected["bid_close"].reset_index(drop=True),
        check_names=False,
    )
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fetch_data'`

- [ ] **Step 3: `scripts/fetch_data.py` を実装する**

```python
"""データ取得CLI。

data/ はGit管理外なので、別PCではこれを実行してレイクを再構築する。
何度実行しても同じ結果になる（レイク側で重複を落とす）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitrading import quality
from aitrading.config import Settings, load_settings
from aitrading.datasource.base import BarSource
from aitrading.datasource.dukascopy import DukascopySource
from aitrading.storage.lake import Lake
from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe


def fetch(
    settings: Settings,
    source: BarSource,
    lake: Lake,
    meta: Meta,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    chunk_days: int = 30,
) -> None:
    """1分足を分割して取得し、レイクへ保存して品質レポートを記録する。

    一度に全期間を要求すると失敗時のやり直しが大きいので、区切って進める。
    """
    start = pd.Timestamp(start or settings.data_start)
    end = pd.Timestamp(end or pd.Timestamp.now(tz="UTC")).floor("min")

    cursor = start
    step = pd.Timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + step, end)
        bars = source.fetch(settings.symbol, Timeframe.M1, cursor, chunk_end)
        if not bars.empty:
            lake.save(settings.symbol, Timeframe.M1, bars)
            meta.record_fetch(settings.symbol, Timeframe.M1, cursor, chunk_end)
            print(f"{cursor:%Y-%m-%d} 〜 {chunk_end:%Y-%m-%d}: {len(bars)} 本")
        cursor = chunk_end

    stored = lake.load(settings.symbol, Timeframe.M1, as_of=end)
    if not stored.empty:
        report = quality.check(stored, settings.symbol, Timeframe.M1)
        meta.record_quality(settings.symbol, Timeframe.M1, report.to_dict())
        print(
            f"品質: {report.actual_bars}/{report.expected_bars} 本、"
            f"欠損区間 {len(report.gaps)} 箇所、最長 {report.longest_gap_minutes:.0f} 分"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="USD/JPY の1分足を取得してレイクへ保存する")
    parser.add_argument("--start", help="開始日 (YYYY-MM-DD)。既定は settings.toml の値")
    parser.add_argument("--end", help="終了日 (YYYY-MM-DD)。既定は現在時刻")
    parser.add_argument("--chunk-days", type=int, default=30, help="1回の取得日数")
    args = parser.parse_args(argv)

    settings = load_settings()
    fetch(
        settings,
        DukascopySource(),
        Lake(settings.data_root),
        Meta(settings.meta_db),
        start=pd.Timestamp(args.start, tz="UTC") if args.start else None,
        end=pd.Timestamp(args.end, tz="UTC") if args.end else None,
        chunk_days=args.chunk_days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: `scripts/build_bars.py` を実装する**

```python
"""上位足・日足2系統の生成CLI。

1分足だけが取得物で、他はすべてここで作る生成物。
いつ再生成しても同じ結果になる。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitrading.bars import resample
from aitrading.config import Settings, load_settings
from aitrading.storage.lake import Lake
from aitrading.timeutil import Timeframe

DERIVED = [
    Timeframe.M5,
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.H4,
    Timeframe.D1_NY,
    Timeframe.D1_JST,
    Timeframe.W1_NY,
    Timeframe.W1_JST,
]


def build(
    settings: Settings, lake: Lake, *, timeframes: list[Timeframe] | None = None
) -> None:
    as_of = pd.Timestamp.now(tz="UTC")
    source = lake.load(settings.symbol, Timeframe.M1, as_of=as_of)
    if source.empty:
        raise SystemExit("1分足が無い。先に scripts/fetch_data.py を実行すること")

    for timeframe in timeframes or DERIVED:
        derived = resample(source, timeframe)
        if derived.empty:
            print(f"{timeframe.value}: 生成できるバーが無い")
            continue
        lake.save(settings.symbol, timeframe, derived.reset_index())
        print(f"{timeframe.value}: {len(derived)} 本")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="1分足から上位足・日足2系統を生成する")
    parser.add_argument("--timeframe", action="append", help="生成する時間軸（複数可）")
    args = parser.parse_args(argv)

    settings = load_settings()
    selected = [Timeframe(t) for t in args.timeframe] if args.timeframe else None
    build(settings, Lake(settings.data_root), timeframes=selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: PASS（5 passed）

- [ ] **Step 6: 実データで1週間分だけ通す**

```bash
uv run python scripts/fetch_data.py --start 2026-01-05 --end 2026-01-12
```

Expected: 取得本数と品質サマリが表示される。`data/bars/USDJPY/1m/2026.parquet` ができている。

```bash
uv run python scripts/build_bars.py
```

Expected: 8つの時間軸それぞれの本数が表示される。

- [ ] **Step 7: コミット**

```bash
git add scripts/ tests/test_scripts.py
git commit -m "取得・生成のCLIを実装

fetch_data.py は1分足を30日ずつ区切って取得する。一度に全期間を要求すると
失敗時のやり直しが大きいため。何度実行しても同じ結果になる。取得後に品質
チェックを走らせて meta.db に記録する。

build_bars.py は1分足から8つの時間軸を生成する。生成した5分足が resample の
出力と一致することをテストで確認しており、いつ再生成しても同じ結果になる。"
```

---

## Task 12: Streamlit ダッシュボード

**Files:**
- Create: `dashboard/app.py`
- Create: `tests/test_dashboard.py`

**Interfaces:**
- Consumes: `load_settings`, `Lake`, `Meta`, `INDICATORS`, `scan`, `session_labels`
- Produces:
  - `dashboard/app.py` — `candlestick_figure(bars, overlays=None, trades=None) -> plotly.graph_objects.Figure`、`main() -> None`
  - `trades` は `entry_time / exit_time / side / entry_price / exit_price / sl / tp` を持つ DataFrame（Phase 0 では常に `None`。仕様書§17の口だけ用意する）

- [ ] **Step 1: 失敗するテストを書く**

`tests/test_dashboard.py`:

```python
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))

import app  # noqa: E402

from tests.test_bars import minute_bars


def test_candlestick_figure_has_ohlc_trace():
    fig = app.candlestick_figure(minute_bars("2026-01-05 00:00", 60))
    assert any(trace.type == "candlestick" for trace in fig.data)


def test_overlays_are_added_as_lines():
    bars = minute_bars("2026-01-05 00:00", 60)
    overlay = bars["bid_close"].rolling(5).mean().rename("sma5")
    fig = app.candlestick_figure(bars, overlays={"sma5": overlay})
    names = [trace.name for trace in fig.data]
    assert "sma5" in names


def test_trades_are_drawn_when_supplied():
    """仕様書§17の口。Phase 0ではデータを流さないが、描ける形にはしておく。"""
    bars = minute_bars("2026-01-05 00:00", 60)
    trades = pd.DataFrame(
        {
            "entry_time": [bars.index[5]],
            "exit_time": [bars.index[20]],
            "side": ["BUY"],
            "entry_price": [150.1],
            "exit_price": [150.4],
            "sl": [149.9],
            "tp": [150.5],
        }
    )
    fig = app.candlestick_figure(bars, trades=trades)
    names = [trace.name for trace in fig.data if trace.name]
    assert any("entry" in n.lower() for n in names)


def test_empty_bars_produce_empty_figure():
    empty = minute_bars("2026-01-05 00:00", 0)
    fig = app.candlestick_figure(empty)
    assert fig.data == () or len(fig.data) == 0
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 3: `dashboard/app.py` を実装する**

```python
"""チャート・データ品質・期待値スキャンの確認画面。

描画ロジック（candlestick_figure）と Streamlit のUIを分けてある。
前者はテストできるが、後者はできないため。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aitrading.config import load_settings
from aitrading.indicators import INDICATORS
from aitrading.storage.lake import Lake
from aitrading.storage.meta import Meta
from aitrading.timeutil import Timeframe


def candlestick_figure(
    bars: pd.DataFrame,
    overlays: dict[str, pd.Series] | None = None,
    trades: pd.DataFrame | None = None,
) -> go.Figure:
    """ローソク足＋指標＋トレードマーカー。

    trades は仕様書§17（チャート監査）の口。Phase 0 では常に None だが、
    後のフェーズでトレードログをそのまま渡せる形にしてある。
    """
    fig = go.Figure()
    if bars.empty:
        return fig

    mid_of = lambda field: (bars[f"bid_{field}"] + bars[f"ask_{field}"]) / 2.0  # noqa: E731
    fig.add_trace(
        go.Candlestick(
            x=bars.index,
            open=mid_of("open"),
            high=mid_of("high"),
            low=mid_of("low"),
            close=mid_of("close"),
            name="USDJPY",
        )
    )

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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="ai-trading", layout="wide")
    settings = load_settings()
    lake, meta = Lake(settings.data_root), Meta(settings.meta_db)

    st.sidebar.header("表示設定")
    timeframe = Timeframe(
        st.sidebar.selectbox(
            "時間軸", [t.value for t in Timeframe], index=0
        )
    )
    as_of = pd.Timestamp(
        st.sidebar.date_input("as_of（この時刻より後は表示しない）", pd.Timestamp.now().date()),
        tz="UTC",
    ) + pd.Timedelta(days=1)
    chosen = st.sidebar.multiselect("指標", sorted(INDICATORS))

    bars = lake.load(settings.symbol, timeframe, as_of=as_of).tail(1500)

    chart_tab, quality_tab, edge_tab = st.tabs(["チャート", "データ品質", "期待値スキャン"])

    with chart_tab:
        if bars.empty:
            st.warning("データが無い。scripts/fetch_data.py を実行すること")
        else:
            overlays: dict[str, pd.Series] = {}
            for name in chosen:
                result = INDICATORS[name](bars)
                if isinstance(result, pd.DataFrame):
                    overlays.update({f"{name}.{c}": result[c] for c in result.columns})
                else:
                    overlays[name] = result
            st.plotly_chart(candlestick_figure(bars, overlays), use_container_width=True)

    with quality_tab:
        report = meta.latest_quality(settings.symbol, timeframe)
        if report is None:
            st.info("品質レポートがまだ無い")
        else:
            st.metric("取得本数", f"{report['actual_bars']} / {report['expected_bars']}")
            st.metric("最長欠損", f"{report['longest_gap_minutes']:.0f} 分")
            st.dataframe(pd.DataFrame(report["gaps"]))

    with edge_tab:
        st.caption(
            "集計は Training 期間のみ。OOS期間はロックされており、"
            "解除には理由の明示と記録が要る。"
        )
        unlocks = meta.oos_unlocks()
        if unlocks:
            st.warning(f"OOS期間は過去 {len(unlocks)} 回解除されている")
            st.dataframe(pd.DataFrame(unlocks))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `uv run pytest tests/test_dashboard.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 全テストを通す**

Run: `uv run pytest -v`
Expected: 全て PASS（network マーカーは deselected）

- [ ] **Step 6: 実際に画面を開いて目で確かめる**

```bash
uv run streamlit run dashboard/app.py
```

確認すること:
- 時間軸を `1m` → `5m` → `1D_ny` → `1D_jst` と切り替えてローソク足が描かれる
- `1D_ny` と `1D_jst` で足の区切りが違う
- 指標を選ぶと線が重なる
- データ品質タブに取得本数と欠損区間が出る
- `as_of` を過去日にすると、それ以降のローソクが消える

- [ ] **Step 7: コミット**

```bash
git add dashboard/ tests/test_dashboard.py
git commit -m "Streamlitダッシュボードを実装

チャート／データ品質／期待値スキャンの3タブ。描画ロジックを
candlestick_figure に切り出してテスト可能にし、Streamlit のUIと分けた。

トレードのIN/OUT・SL/TPを描く口（仕様書§17）を用意した。Phase 0では
データを流さないが、後のフェーズでトレードログをそのまま渡せる形にしてある。

サイドバーの as_of で「その時刻より後を表示しない」を切り替えられる。
先読み防止の仕組みが実際に効いていることを目で確認できる。"
```

---

## Task 13: READMEの更新と Phase 0 完了確認

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 全テストとカバレッジを確認する**

Run: `uv run pytest -v`
Expected: 全て PASS

Run: `uv run pytest -m network -v`
Expected: PASS（実サーバーに接続できること）

- [ ] **Step 2: まっさらな状態から復元できるか確かめる**

```bash
mv data data_backup
uv run python scripts/fetch_data.py --start 2026-01-05 --end 2026-01-12
uv run python scripts/build_bars.py
```

Expected: エラーなく `data/` が再構築される。確認できたら `rm -rf data && mv data_backup data` で戻す。

これが通らなければ「別PCへの移行」が成立していない。

- [ ] **Step 3: README のステータス表を更新する**

`README.md` の「現在のステータス」表で Phase 0 の状態を `**設計完了 / 実装前**` から `**完了**` に変える。「開発ルール」の最終行を「Phase 0（1〜4＋期待値スキャン）完了。次は Phase 1（バックテストエンジン）。」に変える。

- [ ] **Step 4: コミットしてプッシュ**

```bash
git add README.md
git commit -m "Phase 0 完了。READMEのステータスを更新

データ取得→保存→上位足生成→品質チェック→指標→期待値スキャン→チャート表示
まで一通り通り、全テストが緑。data/ を消してもスクリプトで再構築できることを
確認済み（別PCへの移行が成立する条件）。

次は Phase 1（バックテストエンジン）。"
git push origin main
```

---

## Self-Review

**1. Spec coverage（設計文書の各節に対応するタスクがあるか）**

| 設計 | タスク |
|---|---|
| §2 AIの役割分担 | — 方針のみ。コード化は Phase 1 |
| §2.1 執行パスのレイテンシ／LLM禁止 | Task 1 Step 6 |
| §2.2 モデル割り当て | Task 2（settings.toml） |
| §2.3 構造化出力 | — Phase 1（LLM呼び出しはPhase 0スコープ外） |
| §3.1 データソース抽象化 | Task 3 |
| §3.2 Dukascopy | Task 4 |
| §3.3 保存形式 | Task 5, Task 6 |
| §3.4 時間軸の識別子 | Task 1（`Timeframe`） |
| §3.5 初期データスコープと期間分割 | Task 2 |
| §3.6 `as_of` カーソル | Task 5 |
| §4.1 UTC保存 | Task 1 |
| §4.2 日境界2系統 | Task 1, Task 7 |
| §4.3 セッションラベル | Task 1 |
| §4.4 市場の開閉 | Task 1, Task 8 |
| §5 第1層 確定足 | Task 7（不完全期間を落とす）, Task 5（`close_time <= as_of`） |
| §5 第2層 トランケーション不変性 | Task 9 |
| §5 第3層 リサンプル規約 | Task 7 |
| §5 第4層 `as_of` | Task 5 |
| §6 品質チェック | Task 8 |
| §7 指標 | Task 9 |
| §8 期待値スキャン | Task 10 |
| §9 ダッシュボード | Task 12 |
| §10 プロジェクト構造 | 各タスクで作成 |
| §11 テスト方針 | 各タスク |
| §13 移行手順 | Task 13 Step 2 |

**2. Placeholder scan** — TBD / TODO / 「後で実装」の類は無し。全ステップに実際のコードが入っている。初稿に紛れていた誤ったimport例（Task 11）は削除済み。

**3. Type consistency** — 確認済み。`Timeframe` のメンバ名（`M1` `D1_NY` 等）と値（`"1m"` `"1D_ny"` 等）、`Lake.load` のキーワード必須 `as_of`、`validate_bars(df, timeframe)` の引数順、`check(bars, symbol, timeframe)` の引数順、`scan(bars, signal, *, ...)` のキーワード専用引数は、定義タスクと使用タスクで一致している。`minute_bars` / `make_bars` はテスト間で再利用するため、定義元（`tests/test_bars.py` / `tests/test_datasource_base.py`）を明示した。
