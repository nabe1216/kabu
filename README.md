# DIVIDEND HEIST

東証プライム × 高配当銘柄の自動シグナル生成ツール。

J-Quants API から日次でデータを取得し、スクリーニング・バケット分類・シグナル判定・ボックス相場判定を行い、ペルソナ5風のダッシュボード（`index.html`）で表示する。

GitHub Pages + GitHub Actions で完全自動運用。月3,300円（J-Quants Standard）のみで稼働。

---

## 構成

```
.
├── index.html              # 公開ダッシュボード（GitHub Pages から配信）
├── data/
│   └── results.json        # GitHub Actions が毎日生成（自動コミット）
├── scripts/
│   ├── generate.py         # メイン処理（J-Quants → スクリーニング → JSON）
│   └── requirements.txt
├── .github/
│   └── workflows/
│       └── daily.yml       # 毎日 19:30 JST に generate.py を実行
├── .gitignore
└── README.md
```

---

## 仕様サマリ

### ユニバース
- 東証プライム × 時価総額300億円以上（約500銘柄）

### スクリーニング（8条件すべて通過 = PASS）
1. 減配ゼロ（過去10期）※累進・DOE銘柄は免除
2. 売上の安定性（過去5期で-10%超下落が2期連続なし）
3. 営業利益の安定性（同上）
4. 純利益の安定性（同上）
5. 配当性向 ≤ 50%
6. 自己資本比率 ≥ 50%（金融4業種+不動産業は免除）
7. PER × PBR ≤ 22.5
8. 最低利回り ≥ 3.0%（累進・DOE銘柄は免除）

### バケット分類
| バケット | 条件 | 取引方針 |
|---|---|---|
| **コア・累進** | スクリーニング通過 ∩ 累進配当銘柄 | 長期保有、SELL は要検討 |
| **コア・DOE** | スクリーニング通過 ∩ DOE採用銘柄 | 長期保有、SELL は要検討 |
| **スイング** | スクリーニング通過 ∩ 上記以外 | BUY/SELL シグナルに機械的に従う |
| **対象外** | スクリーニング非通過 | 投資対象外（参考表示のみ） |

### シグナル判定（過去5年の月次配当利回り分布から）
- 🟢 **BUY**: 現在利回り ≥ Q75
- 🔴 **SELL**: 現在利回り ≤ Q25
- ⚪ **NEUTRAL**: 上記以外

### ボックス判定（うねり取り候補）
- ADX(14) < 22
- 値幅 8〜20%
- 上下限タッチ各2回以上
- ATR / 終値 ≥ 1.5%

### 撤退判定（緊急シグナル）
- 累進配当撤回
- DPS 10%超減配
- 営業利益 YoY -20% 以下

詳細は `設計対応表_v1.4.md` を参照。

---

## セットアップ

詳細は同梱の **セットアップ手順書 (PDF)** を参照。要点のみ：

1. [J-Quants Standard](https://jpx-jquants.com/) を契約（3,300円/月）し、リフレッシュトークンを発行
2. このリポジトリを Public で作成
3. ファイルを配置してコミット&プッシュ
4. **Settings → Secrets and variables → Actions** で `J_QUANTS_API_KEY` にリフレッシュトークンを登録
5. **Settings → Pages** で `main` ブランチをソースに設定
6. **Actions** タブから `Generate Dividend Signals` を手動実行（初回）
7. `https://<username>.github.io/dividend-heist/` にアクセスして表示確認

---

## ローカル開発

```bash
# 依存インストール
pip install -r scripts/requirements.txt

# .env を作成（リポジトリにはコミットしない）
echo 'J_QUANTS_API_KEY=your_refresh_token_here' > .env

# 実行
export $(cat .env | xargs)
python scripts/generate.py

# 出力確認
ls -lh data/results.json
```

---

## 設定変更

`scripts/generate.py` の冒頭にある定数を編集すれば、即時反映可能：

```python
MIN_YIELD_THRESHOLD = 3.0       # 最低利回り
GRAHAM_THRESHOLD = 22.5         # PER × PBR の上限
PAYOUT_RATIO_MAX = 50.0         # 配当性向上限
EQUITY_RATIO_MIN = 0.50         # 自己資本比率下限
BOX_ADX_THRESHOLD = 22.0        # ボックス判定の ADX 上限
```

累進配当・DOE 銘柄リストは `PROGRESSIVE_DECLARED` / `DOE_LIST` セットを編集。

---

## ライセンス

個人利用。J-Quants の利用規約を遵守すること。
