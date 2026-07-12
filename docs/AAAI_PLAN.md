# AAAI 2026 実験計画（統合・最新版）

> 締切: **2026-07-28**。本書は 2026-06-30 時点の確定計画。
> 旧 `docs/PROPOSAL_tokenizer_comparison.md` / `REPORT_tokenizer_experiment_status.md` は
> **2026-06-24 の設計大転換より前**の内容で、下記が現行のデザインに置き換わっている。

---

## 0. 一行サマリ

**「事前学習スキーム（ブロック drop + 並び替えによるデータ拡張） vs 標準自己回帰（Vanilla AR）」を、
モデルサイズ N を完全固定（86M）した 3 点で比較し、下流タスク（生成・補完・分析の SFT、
トークン非依存指標）で提案スキームの優位を示す。**

---

## 1. 設計の経緯（なぜ今のデザインか）

### 1.1 当初デザイン（撤回済み）
「提案トークン化 vs 既存トークン化」を Chinchilla scaling 則で compute-optimal に比較する案。
楽曲コーパス固定 → 各トークナイザで D を得る → r\*=D/N で N を決め各手法を最適学習 →
トークン非依存指標でパレート比較、という構成だった。

### 1.2 致命的欠陥の判明（2026-06-24, 査読者 judge）
2 条件が **同一語彙** であると判明。Chinchilla は語彙/符号化が異なる時のみ有効で、
同一語彙では「N を D に比例させて削る」のは誤用（情報を捨てて D が減っただけなのに
N を削るのは不当ペナルティ）。→ **scaling 則ベースの N 可変デザインは破棄。**

### 1.3 現行デザイン（採用）
「トークン化比較」を撤回し、**事前学習スキームの比較**に再フレーミング:
- 提案 = ブロック drop（k∈{0,1,2}）+ 並び替えによるデータ拡張（v5）
- ベースライン = ブロック移動なし（no-aug）= **標準 AR ベースライン**（NLP の FIM/MAE 的位置づけ。
  独立した REMI 実装は不要）
- 「効率」= 圧縮効率ではなく **学習計算効率（FLOPs 削減）**。MAE の 75% マスクに類似の主張。

---

## 2. 本実験（事前学習）= N 固定 3 点

全モデル同一アーキテクチャ **d768 / L12 / h12 / ff2048 = 86M**（config: `configs/models/mortm/tokencmp/B_remi_86M.json`）。
同一 130,500 曲（×12 転調）。LR 1.2e-3、batch8 / accum64（eff 512–1024）。

| 条件 | 内容 | データ量 | epoch | 計算量 | 状態 |
|---|---|---|---|---|---|
| **R (Vanilla AR)** | no-aug 全ブロック固定順 | 10.33B | 1ep | C_R | ✅ **既存 `B_remi_86M` 流用**（val 1.371） |
| **P-Efficient** | 提案 drop+reorder | 6.91B | 1ep | ≈0.67 C_R | ⬜ **新規学習要** |
| **P-IsoFLOP** | 提案 drop+reorder | ~10.3B | 1.5ep | ≈C_R | ⬜ **新規学習要** |

**主張の二段構え:**
- P-Efficient は R を **33% 少ない計算**で同等 → 効率（学習計算効率）で勝つ
- P-IsoFLOP は **同計算**で R を上回る → 絶対性能でも勝つ

※ 旧 A_prop_60M / C_prop_127M は N 不一致のため**破棄**（参考に重みは残存）。
※ N=86M は tokens/param 80–120 ≫ 20 で過剰適合領域ではない（査読者の小N懸念は不要）。

### 2.1 既存アセット
- 学習済みモデル: `out/models/mortm/tokencmp/MORTM.B_remi_86M_1.3707563877105713.pth`（= R）
- データ: `/home/takaaki-nagoshi/data/scaling/tokencmp/{proposed,remi}/{train,eval}.json`
  - proposed train = 6.91B（1,566,000 npz）, remi train = 10.33B（同 npz 数）, eval 各 2,217 曲（原曲のみ）
- 生成スクリプト: `scaling/make_tokencmp_datasets.py`
- 学習ランナー: `scaling/tokencmp_train.py`（A→B→C 順だったので **P-Efficient/P-IsoFLOP 用に要調整**）

### 2.2 残作業（本実験）
1. **P-Efficient 学習**: 86M, proposed データ 6.91B, 1 epoch（≈ R の 0.67 計算）
2. **P-IsoFLOP 学習**: 86M, proposed データ, **1.5 epoch**（≈ R と同計算）
3. R は B_remi_86M をそのまま使用（再学習不要）
- 所要見積り: 新規 2 本で約 40h（2 日弱、80M 級・no-aug 長系列 batch8）。

### 2.3 ⚠ 既知の落とし穴（再発防止）
- **loss=0 バグ**: `train.py` の `loss_mask` は `<MGEN>/<CGEN>/<META>...` 出現後しか損失計算しない（SFT 用）。
  foundation データはこれらを含まないため、マーカー無し系列は **全トークン損失（mask 全 1）** にする
  `has_marker` フォールバックが必須。repo (`mortm/train/train.py`) と installed (`.venv/.../mortm/train/train.py`)
  両方にパッチ済。**pip 再 install で上書きされたら再パッチ。**
- DDP 不均一バッチ問題（過去に 80M で NCCL デッドロック）→ 必要なら単一 GPU + accum 倍化。

---

## 3. 下流タスク（SFT）= 本番の勝負どころ

R / P-Efficient / P-IsoFLOP の 3 モデルを **同一データ・同一ステップ** で SFT し、
**トークン非依存指標**で比較する。token PPL は厳禁。**bits-per-note（音符あたり）正規化尤度が必須**
（test の総音符数はトークナイザ/スキーム不変）。+ `mortm/utils/eval.py` の音符レベル分布距離。

下流は **生成・補完・分析の 3 カテゴリ**（手厚い方向をユーザーが選択）。

| 下流タスク | 指標（案） | 状態 |
|---|---|---|
| 生成 (meta→melody) | 音符分布距離 / リスニング / bits-per-note | 🟡 デモ SFT パイプライン構築済（[[project-sft-tasks]]）。本実験 3 モデルへの適用は未 |
| 補完 (past+future→melody) | 音符 F1 等の参照一致 | ⬜ データ・評価器 未構築 |
| 分析 (melody→meta) | 分類精度 / F1 | 🟡 `AnalysisDataMaker` + `ConvertSFT_analysis.py` 実装済だが保留中 |

※ 注意: デモ SFT（`train_sft.py`, MIDICaps, LoRA）は **80M SOTA モデル**の controllable 生成デモであり、
本実験の R/P-Efficient/P-IsoFLOP 比較とは**別物**。下流評価では 3 モデルを同条件 SFT する必要がある。

---

## 4. 公平性・査読突破ロジック（記録）

- **同一語彙**だから「事前学習スキーム比較」として中立。R は独立 REMI 不要＝標準 AR ベースラインとして正当。
- トークン数差（提案/REMI = 0.669）は「トークン化の質」でなく主に **ブロック削除 (k∈{0,1,2})** 由来 ＝ fertility 交絡。
  論文では実学習量として正直に提示。
- loss 絶対値は提案（v5 eval）と no-aug（eval）で分布が違い**直接比較不可** → だから下流（PPL 非依存）で勝負。
- 詳細な公平性検証は [[project-tokenizer-comparison]] に記録。

---

## 5. 論文に使える既存成果（測定済み）

- スケーリング則 v5（25 点）/ no-aug（25 点）対照、深さアブレーション、α 同定不能の谷の図、D/N 最適 ≫ 20。
  → [[project-scaling-v5-grid]] [[project-scaling-noaug]] [[project-depth-ablation]]
- 観察: 「データ拡張がデータ scaling 効率（β）を下げる」（窓 24/ストライド 8 の 2/3 重複）→ limitation として記述可（再実験はしない）。

---

## 6. TODO チェックリスト（優先順）

- [ ] **P-Efficient 学習**（86M, proposed 6.91B, 1ep）— 約 20h
- [ ] **P-IsoFLOP 学習**（86M, proposed ~10.3B, 1.5ep）— 約 20h
- [ ] R = B_remi_86M を流用（作業なし）
- [ ] 下流評価器（bits-per-note）を `mortm/utils/eval.py` ベースで整備
- [ ] 3 モデル × 3 下流タスク（生成/補完/分析）の同条件 SFT
- [ ] パレート図（X=FLOPs, Y=下流性能）+ compute-equivalent 点の表
- [ ] 論文執筆（締切 7/28 厳守、新規大型実験は原則打ち切り）

---

## 7. 関連ファイル早見

| 種別 | パス |
|---|---|
| モデル config | `configs/models/mortm/tokencmp/{A_prop_60M,B_remi_86M,C_prop_127M}.json` |
| 学習済み（R 流用元） | `out/models/mortm/tokencmp/MORTM.B_remi_86M_1.3707563877105713.pth` |
| 本実験データ | `/home/takaaki-nagoshi/data/scaling/tokencmp/{proposed,remi}/{train,eval}.json` |
| データ生成 | `scaling/make_tokencmp_datasets.py` |
| 学習ランナー | `scaling/tokencmp_train.py`（R/P-Eff/P-IsoFLOP 用に要調整） |
| 旧 proposal/report | `docs/PROPOSAL_tokenizer_comparison.md`, `docs/REPORT_tokenizer_experiment_status.md`（旧設計、参考） |

関連 memory: project-tokenizer-comparison / project-aaai-deadline / project-scaling-v5-grid / project-scaling-noaug / project-sft-tasks
