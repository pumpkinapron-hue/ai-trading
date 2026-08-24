# ai-trading

AIトレード研究・自動売買システム。FX（USD/JPY）を最初の対象として、
再現性のある売買優位性を発見・検証するための研究基盤を段階的に構築する。

**実資金は投入しない。** 発注機能は Phase 3 まで実装しない。

## 現在のステータス

| フェーズ | 内容 | 状態 |
|---|---|---|
| Phase 0 | 研究基盤（データ取得・保存・チャート・指標・期待値スキャン） | **設計完了 / 実装前** |
| Phase 1 | バックテストエンジン、戦略、リスク管理 | 未着手 |
| Phase 2 | フォワードテスト（仮想資金・数か月） | 未着手 |
| Phase 3 | 極小ロットでの実運用 | 未着手 |
| Phase 4 | 日本株・米国株への横展開 | 未着手 |

## 設計の要点

### AIの役割分担 — 研究はLLM、執行はルール

戦略の発案・レビュー・改善提案は LLM が担当し、実際の売買判断は
**LLMが生成した決定論的なルール**が下す。LLMは同じ入力に同じ出力を返さないため、
執行をLLMに直接やらせるとバックテストと実運用が原理的に一致しなくなる。

執行パスは1秒未満で完結する（指標計算〜条件判定が数ms、発注のネットワークが数十〜数百ms）。
LLMを挟むと5〜15秒かかるため、執行系モジュールがLLMクライアントを import していないことを
テストで検査する。

研究レイヤーのモデルは頻度と難易度で割り当てる。戦略発案は低頻度なので最上位モデル
（`claude-fable-5`）を使い、毎トレードの定型レビューは `claude-sonnet-5`、
ニュース分類は `claude-haiku-4-5`。モデルIDは `config/settings.toml` に置き、コードに直書きしない。

### 先読みバイアスを構造的に排除する

「気をつける」ではなく、書こうとしても書けない構造にしている。

1. 確定足しか分析APIに出てこない
2. 全指標へのトランケーション不変性テスト（末尾を削っても出力が変わらないこと）
3. リサンプルは `close_time` 基準（4時間足の確定前にその内容は参照できない）
4. `as_of` カーソル — 指定時刻より後のデータは物理的に返さない

### 日本基準とNY基準を並列に持つ

日足・週足の区切りを `NY_CLOSE`（17:00 America/New_York, DST追従）と
`JST`（00:00 Asia/Tokyo）の2系統で生成し、別々に保存する。
加えて各バーに `TOKYO` / `LONDON` / `NEWYORK` / `LDN_NY_OVERLAP` / `OFF` の
セッションラベルを付け、時間帯ごとの優位性の差を分析できるようにする。

### Out-of-Sample はデフォルトでロックする

期待値スキャンの集計対象は Training 期間のみ。OOS期間を見るには明示的なフラグが必要で、
その解除は記録に残る。人間が一度OOSの結果を見てしまったら、そのOOSはもうOOSではないため。

詳細は [docs/specs/2026-08-24-phase0-design.md](docs/specs/2026-08-24-phase0-design.md)。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/specs/spec-v0.1.md](docs/specs/spec-v0.1.md) | 最上位仕様書（原本） |
| [docs/specs/2026-08-24-phase0-design.md](docs/specs/2026-08-24-phase0-design.md) | Phase 0 設計文書 |
| [docs/decisions/](docs/decisions/) | ADR（設計判断の記録） |

## 技術スタック

| 用途 | 選定 |
|---|---|
| 言語 / パッケージ管理 | Python 3.12 / uv |
| データ処理 | pandas, numpy, pyarrow |
| 市場データ | Dukascopy（無料ヒストリカル、Bid/Ask両方、2005年〜） |
| バーデータ保存 | Parquet |
| メタデータ保存 | SQLite |
| 指標 | 自前実装（TA-Libは使わない） |
| ダッシュボード | Streamlit + Plotly |
| テスト | pytest |

## ディレクトリ構成

```
ai-trading/
├── config/          設定（データ範囲・期間分割・タイムゾーン）
├── src/aitrading/
│   ├── datasource/  データソースアダプタ（Dukascopy / 将来 OANDA・MT5）
│   ├── storage/     Parquetレイク + SQLiteメタ
│   ├── indicators/  テクニカル指標
│   ├── timeutil.py  UTC / 日境界2系統 / セッションラベル
│   ├── bars.py      ティック→バー集約・リサンプル
│   ├── quality.py   データ品質チェック
│   └── edge_scan.py 期待値スキャン
├── scripts/         CLI（データ取得・上位足生成）
├── dashboard/       Streamlit アプリ
├── tests/
├── docs/
└── data/            ← Git管理外。scripts/fetch_data.py で再取得する
```

将来のサブシステムの置き場所（Phase 1以降に追加）:

| サブシステム | 置き場所 |
|---|---|
| バックテストエンジン | `src/aitrading/backtest/` |
| 戦略定義 | `strategies/{active,experimental,archived}/` |
| リスク管理 | `src/aitrading/risk/` |
| トレードログ | `src/aitrading/trade_log/` + `data/meta.db` |
| ニュース解析 | `src/aitrading/news/` |
| トレーダー知識DB | `trader_knowledge/` |
| 発注 | `src/aitrading/execution/` |

## セットアップ

```bash
uv sync
```

## 別PCへの移行

市場データはリポジトリに含めない（再取得できるため）。

```bash
git clone <repo-url>
cd ai-trading
uv sync
python scripts/fetch_data.py
streamlit run dashboard/app.py
```

## 開発ルール

仕様書 §27 に従い、一度に全体を作らない。以下の順で実装し、
各段階でテストが通るまで次へ進まない。

```
1. データ取得   2. データ保存   3. チャート表示   4. テクニカル指標
5. バックテストエンジン   6. BUY/SELL/WAIT戦略   7. リスク管理
8. AI分析   9. トレードログ   10. ダッシュボード   11. ニュース解析
12. トレーダー知識DB   13. フォワードテスト   14. API発注   15. 実運用
```

現在は 1〜4（＋期待値スキャン）を対象とする Phase 0。
