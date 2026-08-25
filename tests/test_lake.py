import pandas as pd
import pytest

from aitrading.storage.lake import Lake
from aitrading.timeutil import Timeframe

from tests.helpers import make_bars


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


# --- 以下はブリーフの8テストではカバーされない挙動を確認する追加テスト ---


def test_load_empty_result_matches_schema_of_non_empty_result(lake):
    """存在しないシンボルの0件フレームが、実データを読んだフレームと同じ

    列・index名・dtypeを持つこと。`.empty` だけの確認だと、pandasのバージョン
    依存で分解能がズレる（例: pandas 3.0.5 は date_range(tz="UTC") で
    datetime64[us, UTC] を返すが、ここを "datetime64[ns, UTC]" と決め打ちすると
    0件のときだけ型が違うフレームになる）ケースを見逃す。
    """
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 3))
    non_empty = lake.load(
        "USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC")
    )
    empty = lake.load(
        "EURUSD", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC")
    )
    assert list(empty.columns) == list(non_empty.columns)
    assert empty.index.name == non_empty.index.name == "open_time"
    assert empty.dtypes.equals(non_empty.dtypes)


def test_start_must_be_tz_aware(lake):
    """as_of と同じ規約が start にも適用される（naive は ValueError）。

    素通りさせると、tz-aware な open_time 列との比較で TypeError になり
    「naive入力はValueError」という全体の規約を破る。
    """
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 3))
    with pytest.raises(ValueError, match="tz-aware"):
        lake.load(
            "USDJPY",
            Timeframe.M1,
            as_of=pd.Timestamp("2030-01-01", tz="UTC"),
            start=pd.Timestamp("2026-01-05 00:00"),  # naive
        )


def test_save_rejects_overlap_created_by_merging_with_existing_year(lake):
    """個別には妥当な2バッチでも、結合すると重なるなら保存を拒否する。

    新規バッチ単体の validate_bars だけでは、既存データとの結合後にしか
    現れない重なりを検出できない。結合・重複除去したあとの年単位チャンクを
    再度 validate_bars に通すことでこれを塞ぐ。
    """
    first = bars_over("2026-01-05 00:00", 1)  # open 00:00:00, close 00:01:00
    lake.save("USDJPY", Timeframe.M1, first)

    second = first.copy()
    second["open_time"] = pd.Timestamp("2026-01-05 00:00:30", tz="UTC")
    second["close_time"] = pd.Timestamp("2026-01-05 00:01:30", tz="UTC")

    with pytest.raises(ValueError, match="重な"):
        lake.save("USDJPY", Timeframe.M1, second)

    # 失敗した結合はディスクに書かれず、既存の1本だけが残る
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert len(got) == 1


def test_save_propagates_validate_bars_errors_without_writing(lake):
    """壊れたバーは書き込まれる前に validate_bars が弾く。"""
    bad = bars_over("2026-01-05 00:00", 3)
    bad.loc[1, "ask_close"] = bad.loc[1, "bid_close"] - 0.10  # Ask が Bid を下回る
    with pytest.raises(ValueError, match="Ask"):
        lake.save("USDJPY", Timeframe.M1, bad)
    assert lake.available_years("USDJPY", Timeframe.M1) == []


def test_save_rejects_conflicting_resave_of_same_open_time(lake):
    """同じ open_time で値が違う再保存はエラーになる（黙って上書きしない）。

    同じ範囲を同じ値で再取得するのは常に成功する必要がある（idempotent）が、
    同じ open_time に違う値が来るのはデータソース側で何かが変わったという
    ことであり、keep="last" で黙って上書きしてはいけない。
    """
    first = bars_over("2026-01-05 00:00", 3)
    lake.save("USDJPY", Timeframe.M1, first)

    conflicting = first.copy()
    conflicting.loc[1, ["bid_close", "ask_close"]] += 0.05  # 同じ open_time、値だけ違う（スプレッドは維持）

    with pytest.raises(ValueError, match="衝突"):
        lake.save("USDJPY", Timeframe.M1, conflicting)

    # 失敗した再保存で1回目の値が上書きされていない
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert got["bid_close"].iloc[1] == first.loc[1, "bid_close"]


def test_save_writes_nothing_when_any_year_fails_validation(lake):
    """複数年にまたがるバッチの一部の年だけ検証に失敗したら、1バイトも書かない。

    2025年ぶんは単体では正常でも、2026年ぶんが既存データと値衝突を起こすなら、
    2025年のファイルも一切書き換わってはいけない。さもないと呼び出し側は
    例外を見てもどの年が書けたか分からず、レイクが部分的に矛盾した状態になる。
    """
    original = bars_over("2026-01-01 00:00", 1)
    lake.save("USDJPY", Timeframe.M1, original)  # 2026年の既存データ

    straddling = bars_over("2025-12-31 23:59", 2)  # 2025年1本 + 2026年1本
    straddling.loc[1, ["bid_close", "ask_close"]] += 0.05  # 2026年側を既存と値衝突させる（スプレッドは維持）

    with pytest.raises(ValueError, match="衝突"):
        lake.save("USDJPY", Timeframe.M1, straddling)

    # 2025年のファイルは作られていない。2026年の既存データも書き換わっていない。
    assert lake.available_years("USDJPY", Timeframe.M1) == [2026]
    got = lake.load("USDJPY", Timeframe.M1, as_of=pd.Timestamp("2030-01-01", tz="UTC"))
    assert len(got) == 1
    assert got["bid_close"].iloc[0] == original["bid_close"].iloc[0]


def test_as_of_cannot_be_passed_positionally(lake):
    """as_of を第3位置引数として渡すと TypeError になる（* の効果そのものを確認）。

    test_as_of_is_keyword_only_and_required は「省略できない」ことしか見ておらず、
    デフォルト値が無ければキーワード専用でなくても省略時は TypeError になるため、
    将来 `*,` が誤って落ちても検出できない。これは第3位置引数として直接渡す形で
    `*` の存在そのものを確認する。
    """
    lake.save("USDJPY", Timeframe.M1, bars_over("2026-01-05 00:00", 3))
    with pytest.raises(TypeError):
        lake.load("USDJPY", Timeframe.M1, pd.Timestamp("2030-01-01", tz="UTC"))
