# MORTM5 下流自己回帰モデル 8-Way Ablation 実験計画書・最終報告書

## 1. 背景と目的

前段のマルチトラックMIDI離散トークナイザー **`MidiVision`** の比較実験（[MIDIVISION_EXPERIMENT_PLAN.md](file:///Users/dayo/My_Projects/MORTM/MIDIVISION_EXPERIMENT_PLAN.md)）により、1チャンク（半拍＝1/8小節、形状 `[8, 128, 24]`）を **16トークン（176 bits、コードブック部分空間32次元、`K=2048`）** に量子化する以下の2モデルが最高精度（`Note F1 > 93.1%`, `Onset F1 > 96.4%`）を達成した：

- **MidiVision Model 3 (`m3_g8l2` / Fine-Group)**: グループ数 $G=8$, 残差段数 $L=2$, 潜在次元 $D=256$, コードブックサイズ $K=2048$
- **MidiVision Model 4 (`m4_g4l4` / Deep-RVQ)**: グループ数 $G=4$, 残差段数 $L=4$, 潜在次元 $D=128$, コードブックサイズ $K=2048$

本実験では、これら2種類の離散トークン列を自己回帰生成する下流基盤モデル **MORTM5 Transformer** において、**最も音楽的文脈把握能力・収束速度・生成品質に優れたアーキテクチャを決定するための $2 \times 2 \times 2 = 8$ 候補（3軸統制アブレーション）** を設計・検証し、最終結論を確定した。

---

## 2. 比較する3つの設計軸（$2 \times 2 \times 2 = 8$ 候補）

### 軸1：ベース・トークナイザー（`M3: G8-L2` vs. `M4: G4-L4`）
- **`M3 (G=8, L=2)`**: 1チャンクを8つの独立グループ × 2段の残差レイヤーで表現。Onset F1 (`96.54%`) とコードブック稼働率 (`100.0%`) に優れる。
- **`M4 (G=4, L=4)`**: 1チャンクを4つのグループ × 4段の深層残差レイヤー（Coarse-to-Fine）で表現。Note F1 (`93.48%`)・Velocity MAE (`0.1111`)・量子化誤差の低さ (`32.71%`) に優れる。

### 軸2：系列モデリング構造（`Flat` vs. `Grouped Soft-Token`）
- **`Flat` モード（[`AblationTransformer1`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py#L364-L616), `model_mode="flat"`）**:
  - 1チャンク内の $L \times G = 16$ 個の離散トークンをそのまま1次元に展開し、1トークン＝1ステップとして連鎖律 $\prod_{g=0}^{G-1} P(z_{l,g} \mid z_{<l}, z_{l,<g})$ に従い厳密に自己回帰予測する。
  - 設定コンテキスト長 `position_length = 4096` で扱える楽曲長は **`255 チャンク`（約 31.9 小節）**。
- **`Grouped` モード（[`AblationTransformer2`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py#L619-L1071), `model_mode="grouped"`）**:
  - 各RVQ残差レイヤー $l \in \{0, \dots, L-1\}$ に属する **$G$ 個の並列グループトークンの Embedding を分散保持係数 $1/\sqrt{G}$ で加算集約し、1つの「Soft-Token」** として Transformer に入力する（[`_aggregate_group_embeddings`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py#L685-L704)）。
  - 出力層（[`DecodeCodebookHead`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/layers.py#L380-L440) / `Wout`）では1つの Soft-Token から $G$ 個のグループヘッドのロジット `[B, S_soft, G, group_vocab_size]` を同時独立予測 $\prod_{g=0}^{G-1} P(z_{l,g} \mid z_{<l})$ する。
  - 1チャンクが **$L$ ステップ**（M3では `2 Soft-Tokens/chunk`、M4では `4 Soft-Tokens/chunk`）に圧縮されるため、M4 (`pos=2048`) で **`510 チャンク`（約 63.8 小節＝8,160 生トークン相当）**、M3 (`pos=1024`) で **`509 チャンク`（約 63.6 小節＝8,144 生トークン相当）** の長距離文脈を見渡せる。

### 軸3：RVQ残差レイヤー間の語彙空間（`Shared` vs. `Separated`）
- **`Shared`（`separate_rvq_layers = False`）**:
  - 同じグループ $g$ であれば、RVQレイヤー $l=0 \dots L-1$ 間で同一のトークンID空間（$G \times K$）を共有する（語彙サイズ M4: `8,325` / M3: `16,517`）。
- **`Separated`（`separate_rvq_layers = True`）**:
  - すべての $(l, g)$ ペアに完全に独立したトークンID空間（$L \times G \times K = 32,768$、特殊トークン込みで `32,901`）を割り当てる。
  - データローダ（[`MORTM5MemmapDataset`](file:///Users/dayo/My_Projects/MORTM/mortm/train/datasets.py#L727-L731)）にてオンザフライで $+l \cdot G \cdot K$ のオフセット変換を行う。
  - 注目すべき点として、`Separated` の4候補（`#2`, `#4`, `#6`, `#8`）は `Flat` と `Grouped (DecodeCodebookHead)` の双方で **総パラメータ数が1パラメータの狂いもなく完全に同一（`135,522,693` = Muon `84.93M` + AdamW `50.59M`）** であり、純粋な構造比較が成立する。

---

## 3. 8-Way Ablation モデル仕様一覧（[`mortm/models/modules/config.py`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/config.py#L45-L100) 準拠）

全モデル共通で **4スロット楽器プレフィックス条件付け（`use_instrument_tokens = True`, `token_offset = 133`）** を有効化している。

| ID | 呼称 (`model_name`) | 設定JSON | Tokenizer | Mode (`Class`) | RVQ語彙 | Global `vocab_size` | `pos_len` | 1ch Step数 | カバー小節数 | 総パラメータ数 (Muon / AdamW) | 実施状態 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`#1`** | `MORTM5_M3_Flat_Shared` | [`m3_flat_shared.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m3_flat_shared.json) | M3 (`G8L2`) | `flat` (`Ablation1`) | Shared | `16,517` | `4096` | `16` | `31.9 bars` (`255 ch`) | `110,340,485` (`84.93M / 25.41M`) | 省略（軸優劣確定） |
| **`#2`** | `MORTM5_M3_Flat_Separated` | [`m3_flat_separated.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m3_flat_separated.json) | M3 (`G8L2`) | `flat` (`Ablation1`) | Separated | `32,901` | `4096` | `16` | `31.9 bars` (`255 ch`) | `135,522,693` (`84.93M / 50.59M`) | 省略（軸優劣確定） |
| **`#3`** | `MORTM5_M3_Grouped_Shared` | [`m3_grouped_shared.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m3_grouped_shared.json) | M3 (`G8L2`) | `grouped` (`Ablation2`) | Shared | `16,517` | `1024` | `2` | `63.6 bars` (`509 ch`) | `111,056,424` (`84.93M / 26.12M`) | 省略（軸優劣確定） |
| **`#4`** | **`MORTM5_M3_Grouped_Separated`** | [`m3_grouped_separated.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m3_grouped_separated.json) | M3 (`G8L2`) | `grouped` (`Ablation2`) | Separated | `32,901` | `1024` | `2` | `63.6 bars` (`509 ch`) | `135,522,693` (`84.93M / 50.59M`) | **100% 完走 (物理予測 SoTA 🏆)** |
| **`#5`** | `MORTM5_M4_Flat_Shared` | [`m4_flat_shared.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m4_flat_shared.json) | M4 (`G4L4`) | `flat` (`Ablation1`) | Shared | `8,325` | `4096` | `16` | `31.9 bars` (`255 ch`) | `97,749,381` (`84.93M / 12.81M`) | 省略（軸優劣確定） |
| **`#6`** | `MORTM5_M4_Flat_Separated` | [`m4_flat_separated.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m4_flat_separated.json) | M4 (`G4L4`) | `flat` (`Ablation1`) | Separated | `32,901` | `4096` | `16` | `31.9 bars` (`255 ch`) | `135,522,693` (`84.93M / 50.59M`) | **100% 完走 (Token PPL 最小)** |
| **`#7`** | `MORTM5_M4_Grouped_Shared` | [`m4_grouped_shared.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m4_grouped_shared.json) | M4 (`G4L4`) | `grouped` (`Ablation2`) | Shared | `8,325` | `2048` | `4` | `63.8 bars` (`510 ch`) | `98,056,212` (`84.93M / 13.12M`) | **100% 完走 ✅** |
| **`#8`** | **`MORTM5_M4_Grouped_Separated`** | [`m4_grouped_separated.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/ablation/m4_grouped_separated.json) | M4 (`G4L4`) | `grouped` (`Ablation2`) | Separated | `32,901` | `2048` | `4` | `63.8 bars` (`510 ch`) | `135,522,693` (`84.93M / 50.59M`) | **100% 完走 (物理予測 Co-SoTA 🏆)** |

---

## 4. 4スロット楽器プレフィックス条件付け（Symbolic Instrument Conditioning）

[`mortm/models/modules/config.py`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/config.py#L200-L240) にて定義された `133` 個の特殊・楽器トークン（ID `0..132`, `INST_RVQ_TOKEN_OFFSET = 133`）を使用：

| トークンID範囲 | 定数名 | 意味 | 対応 `slot_code` |
| :---: | :--- | :--- | :---: |
| `0` | `PAD_TOKEN_ID` | パディング `<PAD>`（損失計算時は `ignore_index=0` で除外） | — |
| `1` | `BOS_TOKEN_ID` | 系列開始 `<BOS>` | — |
| `2` | `EOS_TOKEN_ID` | 楽曲終端 `<EOS>` | — |
| `3 .. 130` | `INST_PROGRAM_OFFSET + prog` | MIDI Program `0..127`（`<INST_PROG_0>` 〜 `<INST_PROG_127>`） | `0 .. 127` |
| `131` | `INST_DRUMS_TOKEN_ID` | ドラムトラック `<INST_DRUMS>`（Slot 0 専用） | `128` |
| `132` | `INST_EMPTY_TOKEN_ID` | 未使用スロット `<INST_EMPTY>` | `-1` |
| `133 ..` | `token_offset + ...` | MidiVision RVQ 離散トークン | — |

---

## 5. 統制されたバックボーン・学習ハイパーパラメータ（[`ablation_muon_cos.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/ablation_muon_cos.json)）

- **Transformer バックボーン共通仕様**:
  - `d_model = 768`, `d_layer = 12`, `num_heads = 12` (`head_dim = 64`), `dim_feedforward = 2048`, `dropout = 0.1`
  - 正規化: `RMSNorm` (`norm_type = "rmsnorm"`, $\epsilon = 10^{-5}$) / 位置符号化: `RoPE` ($\theta = 10000.0$) / 活性化: `SwiGLU` (`use_silu = True`, `use_bias = False`)
- **オプティマイザ（[`HybridMuonOptimizer`](file:///Users/dayo/My_Projects/MORTM/mortm/train/muon.py)）**:
  - **2D 隠れ層重み行列（Attention / SwiGLU FFN, `84,934,656` params）**: `torch.optim.Muon` (`lr = 1.0e-3`, `momentum = 0.95`, `nesterov = True`, `ns_steps = 5`, `weight_decay = 0.1`, `adjust_lr_fn = "match_rms_adamw"`)
  - **Embedding / 出力層 `Wout` / RMSNorm**: `torch.optim.AdamW` (`lr = 1.0e-3`, `betas = (0.9, 0.95)`, `eps = 1e-8`, `weight_decay = 0.1`、1Dは `0.0`)
- **学習率スケジューラ（[`get_cosine_schedule_with_warmup`](file:///Users/dayo/My_Projects/MORTM/mortm/train/noam.py)）**:
  - `warmup_ratio = 0.05`（全ステップの5%で線形ウォームアップ）、`min_lr_ratio = 0.1`（終盤 `1.0e-4` まで Cosine 減衰、`auto_total_steps = True`）
- **データセット & バッチ統制条件**:
  - 事前トークン化済み GigaMIDI Dataset (`train`: `801,838` 曲 / `val`: `4,950` 曲、`num_epochs = 1`、`seed = 42`)
  - 有効バッチサイズ: **`64 windows / optimizer step`** 全モデル完全統一（`big_batch_size = 2048`、`32 × 2` または `#6` のみ VRAM 制約回避のため等価な `16 × 4`）
  - Validation 間隔: **`50,000,000` (50M) トークンごと** に全件検証 & W&B 同期
  - 実行 GPU: **NVIDIA RTX PRO 6000 Blackwell Server Edition (101.97 GB VRAM)**

---

## 6. 第1段階評価：トークン空間 Perplexity (PPL) とスケーリング推移

### 6.1 100% 完走・トークン空間損失比較表（Iso-Data: 全80万曲 1 Epoch 統一）

`#8` (`M4 + Grouped + Separated`) をピボットとして、3つの設計軸を1要素ずつ変化させる **One-Factor-At-A-Time (OFAT) 統制比較** を全モデル 100% 完走させた結果、トークン空間の交差エントロピー損失 (`val/loss`) および Perplexity (`val/ppl`) は以下の通りとなった：

| ID | モデル名 | 検証する設計軸（対 `#8` 比） | 総Steps | 学習Tokens | 総FLOPs (`6ND`) | `train/loss` | **`val/loss`** | **`val/ppl`** | `#8` との差 ($\Delta\text{Loss}$) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`#7`** | `MORTM5_M4_Grouped_Shared` | **軸3**: `Shared` vs `Separated` | `20,609` | `6.90B` | `1.35 × 10¹⁸` | `2.3108` | `2.1791` | `8.84` | `+0.3333 nats` |
| **`#4`** | `MORTM5_M3_Grouped_Separated` | **軸1**: `M3 (G8L2)` vs `M4 (G4L4)` | `20,627` | `6.93B` | `0.70 × 10¹⁸` | `2.1834` | `2.1282` | `8.40` | `+0.2824 nats` |
| **`#8`** | `MORTM5_M4_Grouped_Separated` | **ピボット** (`M4 + Grouped + Sep`) | `20,609` | `6.90B` | `1.40 × 10¹⁸` | `1.9460` | `1.8458` | `6.33` | `0.0000` (基準) |
| **`#6`** | `MORTM5_M4_Flat_Separated` | **軸2**: `Flat` vs `Grouped` | `32,989` | `6.88B` | `5.59 × 10¹⁸` | `1.5615` | **`1.5484`** | **`4.70`** | `-0.2974 nats` (PPL最小) |

> [!WARNING]
> **トークン空間 PPL 単独比較に潜む評価上の罠（第2段階評価の必要性）**:
> - 一見すると `#6 (Flat)` が `val_ppl = 4.70` で圧勝しているように見える（フル語彙 `32,901` クラス Softmax という不利な条件下ですら `#8` の `2,181` クラス [`DecodeCodebookHead`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/layers.py#L380-L413) を `-0.2974 nats` 下回っている）。
> - しかし、**ベース・トークナイザーが異なるモデル間（`M3: G8L2` vs `M4: G4L4`）や、1チャンクあたりの自己回帰ステップ数が異なるモデル間（`16 steps` vs `4 steps` vs `2 steps`）では、1ステップあたりに課される条件付き情報量が根本的に異なるため、平均トークン PPL の直接比較は公平ではない**。
> - そこで第7章にて、全モデルの予測トークンを [`MidiVision`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py) で物理ピアノロール空間 `[N_chunks, 8, 128, 24]` へデコードして統一評価した結果、**トークン PPL 順位が完全に逆転し、提案手法 `Grouped Soft-Token`（`#4`, `#8`）こそが真の次チャンク予測精度でも計算効率でも `#6 (Flat)` を上回るパレート最適解である** ことが実証された。

---

### 6.2 Iso-FLOPs（同一計算量）vs Iso-Data（同一トークン量）スケーリング比較

[`run_remote_mortm5.py`](file:///Users/dayo/My_Projects/MORTM/run_remote_mortm5.py#L488-L500) で記録した `axis/flops`（$= 6 N_{\text{params}} S_{\text{seq}}$）に基づく同一計算量断面でのトークン損失推移：

| 計算量予算 (`axis/flops`) | `#4` (`M3 Grouped`, $8\text{ tok/step}$) | `#8` (`M4 Grouped`, $4\text{ tok/step}$) | `#6` (`M4 Flat`, $1\text{ tok/step}$) | スケーリング上の知見 |
| :--- | :---: | :---: | :---: | :--- |
| **`0.70 × 10¹⁸ FLOPs`** (`#4` 100%完走点) | `2.1282` (`PPL 8.40`, 100%完走) | **`2.0709` (`PPL 7.93`, 50%途中)** | `2.2314` (`PPL 9.31`, 12.5%序盤) | 超低計算予算では多くの楽曲を見られる `M4 Grouped (#8)` が最速収束 |
| **`1.40 × 10¹⁸ FLOPs`** (`#8` 100%完走点) | — | **`1.8458` (`PPL 6.33`, 100%完走)** | `1.9162` (`PPL 6.80`, 25%途中・LR高止まり) | `#8` は `Flat` の 1/4 の計算量で全 80万曲を完走 |
| **`1.79 × 10¹⁸ FLOPs`** (`#6` 32%通過点) | — | *(上限到達 `1.8458`)* | **`1.8450` (`PPL 6.33`, 32%途中)** | トークン平均 NLL 上では `#6` が 32% (`2.20B`) 地点で `#8` を下回る |
| **`5.59 × 10¹⁸ FLOPs`** (`#6` 100%完走点) | — | — | **`1.5484` (`PPL 4.70`, 100%完走)** | チャンク内連鎖律 $\prod_{g} P(z_{l,g} \mid z_{<l}, z_{l,<g})$ により平均 `PPL 4.70` に到達 |

---

## 7. 第2段階評価：統一物理ピアノロール空間による最終検証と「PPL パラドックス」の数理的解明

語彙空間・トークナイザー・チャンク内ステップ数の違いによる評価の不平等を完全に排除するため、専用評価基盤 [`eval_mortm5_pianoroll.py`](file:///Users/dayo/My_Projects/MORTM/eval_mortm5_pianoroll.py)（結果データ: [`pianoroll_eval_results.json`](file:///Users/dayo/My_Projects/MORTM/out/mortm5_ablation/pianoroll_eval_results.json) / レポート: [`pianoroll_eval_report.md`](file:///Users/dayo/My_Projects/MORTM/out/mortm5_ablation/pianoroll_eval_report.md)）を構築し、[MIDIVISION_EXPERIMENT_PLAN.md](file:///Users/dayo/My_Projects/MORTM/MIDIVISION_EXPERIMENT_PLAN.md) §8.1 と完全同一の物理ピアノロール指標で全モデルを直接検証した。

### 7.1 評価プロトコルと「教師データの漏洩ゼロ」保証

- **統一評価データセット**: 未見 GigaMIDI 検証セット `data/gmd/eval.json` 先頭 `150` 曲（計 **`87,523` チャンク**：1〜2トラック疎領域 `33,088` チャンク、3〜4トラック密領域 `54,435` チャンク）。
- **なぜピアノロール変換時に教師データが一切混入しないか（3段階の独立性保証）**:
  1. **チャンク単位教師強制 (`Chunk-level Teacher Forcing`)**: 対象チャンク $c$ を予測する際、全モデルに対して **過去チャンク ($< c$) の正解トークン履歴のみ** を KV キャッシュに与え、**対象チャンク $c$ の正解トークンは1つたりとも見せない**（[`RolloutStrategy._run_branched_intra_chunk_decoder`](file:///Users/dayo/My_Projects/MORTM/eval_mortm5_pianoroll.py#L389-L450) の Shared-Prefix Branched Attention により、`model.generate(temperature=0.0, repetition_penalty=1.0)` と数学的に完全一致することを検証済み）。
  2. **チャンク内の完全自己回帰ロールアウト**: チャンク $c$ を構成する全16個の RVQ トークンは、各モデル自身の予測出力のみを入力に戻しながら自己回帰生成する（`#4`: 2 steps / `#8, #7`: 4 steps / `#6`: 16 steps、構造的文法マスク付き greedy `argmax`）。
  3. **チャンク独立 2D CNN デコード**: [`MidiVisionCodecService.decode_rvq_3d`](file:///Users/dayo/My_Projects/MORTM/eval_mortm5_pianoroll.py#L680-L686) が呼ぶ `MidiVision.decode_tokens` は `[N_chunks, 16] -> [N_chunks, 8, 128, 24]` のチャンク独立 2D CNN であり、前後の正解チャンクからの畳み込み受容野リークは物理的にゼロである。
  4. **長尺曲のスライディングウィンドウ統制**: 各モデルの最大コンテキスト長（`#6`: `255 ch = 31.9小節`、`#4`: `509 ch = 63.6小節`、`#8, #7`: `510 ch = 63.8小節`）を超える長尺曲では、[`SlidingWindowPlanner`](file:///Users/dayo/My_Projects/MORTM/eval_mortm5_pianoroll.py#L260-L297) により最低50%以上の過去文脈オーバーラップを維持しつつ、全 `87,523` チャンクを重複なく厳密に1回ずつ評価した。

---

### 7.2 統一物理ピアノロール復元ベンチマーク結果（全 `87,523` チャンク）

#### 表7.1：【主評価・完全公平比較】チャンク単位教師強制 ＋ チャンク内自己回帰生成 (`Chunk-level TF`)

| 順位 | モデル | 構成 (`Tokenizer / Mode / Vocab`) | 学習FLOPs / 1ch推論 | Token `val/ppl` | **Note F1** (`Rec / Prec`) ↑ | **Onset F1** (`Rec / Prec`) ↑ | **Velocity MAE** ↓ | `1〜2 Trk` Note / Onset F1 | `3〜4 Trk` Note / Onset F1 | RVQ `L0` / 全層一致率 |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *上限* | *Oracle (`M3`)* | `G8-L2` 直接再構成上限 | — | — | `93.14%` (`95.6% / 90.8%`) | `96.54%` (`97.9% / 95.3%`) | `0.1127` | `93.02% / 95.96%` | `93.18% / 96.70%` | `100.0% / 100.0%` |
| *上限* | *Oracle (`M4`)* | `G4-L4` 直接再構成上限 | — | — | `93.48%` (`95.2% / 91.8%`) | `96.44%` (`97.6% / 95.3%`) | `0.1111` | `93.55% / 95.65%` | `93.46% / 96.64%` | `100.0% / 100.0%` |
| **1位 🏆** | **`#4`** | **`M3 (G8L2)` / `Grouped` / `Separated`** | **`0.70e18` / `2 steps`** | `8.40` | **`78.58%`** (`81.8% / 75.7%`) | **`73.00%`** (`72.9% / 73.1%`) | **`0.2596`** | `75.63%` / **`62.69%`** | **`79.44% / 75.67%`** | `52.93%` / **`47.39%`** |
| **同率1位 🏆** | **`#8`** | **`M4 (G4L4)` / `Grouped` / `Separated`** | **`1.40e18` / `4 steps`** | `6.33` | **`78.52%`** (`80.7%` / **`76.4%`**) | **`71.85%`** (`71.4% / 72.4%`) | `0.2664` | **`75.74%`** / `60.94%` | `79.32% / 74.67%` | **`55.53%`** / `44.23%` |
| 3位 | **`#6`** | `M4 (G4L4)` / `Flat` / `Separated` | `5.59e18` / `16 steps` | **`4.70`** | `77.26%` (`78.9% / 75.7%`) | `70.13%` (`68.7% / 71.6%`) | `0.2820` | `75.05% / 60.28%` | `77.89% / 72.64%` | `51.73% / 41.36%` |
| 4位 | **`#7`** | `M4 (G4L4)` / `Grouped` / `Shared` | `1.35e18` / `4 steps` | `8.84` | `75.65%` (`78.3% / 73.2%`) | `68.74%` (`68.2% / 69.3%`) | `0.2860` | `72.74% / 57.73%` | `76.49% / 71.55%` | `53.01% / 42.64%` |

#### 表7.2：【参考評価】ステップ単位教師強制 (`Step-level 1-Step-Ahead TF`)

> 学習時の損失計算と同様に、チャンク内の途中ステップでも「直前ステップまでの正解トークン」を入力して1ステップ先のみを予測させた場合の結果。

| モデル | 構成 (`Tokenizer / Mode / Vocab`) | Note F1 (`Rec / Prec`) | Onset F1 (`Rec / Prec`) | Velocity MAE ↓ | `1〜2 Trk` Note / Onset F1 | `3〜4 Trk` Note / Onset F1 | RVQ `L0` / 全層一致率 | `Chunk-TF` との Note F1 差 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`#4`** | `M3 (G8L2)` / `Grouped` / `Separated` | **`77.40%`** (`81.3% / 73.9%`) | **`73.94%`** (`74.3% / 73.6%`) | **`0.2574`** | `75.24% / 65.87%` | **`78.03% / 76.00%`** | `52.93% / 56.85%` | `-1.18 pt` |
| **`#6`** | `M4 (G4L4)` / `Flat` / `Separated` | `76.12%` (`79.1% / 73.4%`) | `73.07%` (`73.7% / 72.4%`) | `0.2639` | **`77.56% / 68.81%`** | `75.71% / 74.15%` | **`60.22% / 63.20%`** | `-1.14 pt` *(Onset `+2.94 pt`)* |
| **`#8`** | `M4 (G4L4)` / `Grouped` / `Separated` | `73.71%` (`76.8% / 70.9%`) | `69.79%` (`69.9% / 69.7%`) | `0.2844` | `72.94% / 62.48%` | `73.94% / 71.63%` | `55.53% / 59.91%` | **`-4.81 pt` (残差キメラ劣化)** |
| **`#7`** | `M4 (G4L4)` / `Grouped` / `Shared` | `71.00%` (`74.9% / 67.5%`) | `66.42%` (`66.7% / 66.1%`) | `0.3021` | `70.06% / 58.85%` | `71.28% / 68.31%` | `53.01% / 53.77%` | **`-4.65 pt` (残差キメラ劣化)** |

> [!NOTE]
> **なぜ `Grouped` モデル（`#8`, `#7`）は教師を減らした `Chunk-level TF` の方が `+4.7〜+4.8 pt` も精度が高いのか？（残差キメラ現象の発見）**:
> - `Step-level TF` では、モデルが `L0` で正解と異なる和音 $\hat{z}_0 \neq z_0^{\text{gt}}$ を予測した場合でも、`L1〜L3` の入力には強制的に正解 $z_0^{\text{gt}}$ が注入される。結果として「予測 $\hat{z}_0$ の上に、正解 $z_0^{\text{gt}}$ 専用の微細残差 $\Delta z_{1..3}(z_0^{\text{gt}})$ を無理やり加算する」という潜在空間上の **残差キメラ（不整合ベクトル）** が発生し、デコード波形が崩壊する（`L=2` の `#4` では残差1段のみのため影響が `-1.18 pt` と軽微）。
> - 一方、実際の推論条件である `Chunk-level TF` では、モデル自身が選んだ $\hat{z}_0$ を条件として整合的な微細残差 $\Delta z_{1..3}(\hat{z}_0)$ を自己修復的に積層できるため、`#8` の Note F1 は **`73.71% → 78.52%`（`+4.81 pt`）** へと劇的に向上する。

---

### 7.3 「PPL パラドックス」の数理的・実証的解明（なぜ PPL 最小の `#6` が `#4`・`#8` に敗北したのか？）

「なぜ `#6 (Flat)` は `val_ppl = 4.70` と圧倒的に低いのに、実際の次チャンク予測精度では `#4 (PPL 8.40)` や `#8 (PPL 6.33)` に明確に負けたのか？」を解明するため、同一の `M4` トークナイザー・同一パラメータ数 (`135.52M`) を持つ **`#8 (Grouped)` vs `#6 (Flat)`** について、未見検証曲 15曲（`8,525` チャンク：短尺領域 `c < 255` 計 `2,717` チャンク ＋ 長尺領域 `c >= 255` 計 `5,808` チャンク）で **グループ位置 $g \in \{0, 1, 2, 3\}$ ごとの損失 (NLL) と `L0` 正解率を完全分解** した結果、すべての謎が数学的に氷解した：

#### 表7.3：グループ位置 $g \in \{0, 1, 2, 3\}$ 別の損失 (NLL) および短尺・長尺文脈別 `L0` トークン正解率分解

| 指標 / 対象領域 | モデル | **`g = 0`（チャンク先頭・漏洩ゼロ）** | `g = 1` | `g = 2` | `g = 3`（チャンク末尾） | 4グループ平均 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`L0` 損失 (NLL ↓)** | **`#8` (`Grouped`)** | **`2.1453 nats` (勝利 🏆)** | `2.1627` | `2.1102` | `2.2178` | `2.1590` (全層平均 `1.8555`) |
| *(全 `8,525` チャンク)* | **`#6` (`Flat`)** | `2.2548 nats` (敗北) | **`1.7977`** | **`1.5146`** | **`1.4267`** | **`1.7485`** (全層平均 **`1.5163`**) |
| **`L1` 損失 (NLL ↓)** | **`#8` (`Grouped`)** | `1.7724 nats` | `1.8048` | `1.7210` | `1.8487` | `1.7867` |
| *(全 `8,525` チャンク)* | **`#6` (`Flat`)** | `1.7588 nats` | **`1.5509`** | **`1.3353`** | **`1.3012`** | **`1.4866`** |
| **短尺領域 (`c < 255`) `L0` 正解率 ↑** | **`#8` (`Grouped`)** | **`51.56%` (勝利 🏆)** | `49.80%` | `51.60%` | `50.64%` | `50.90%` |
| *(32小節以内・文脈長ハンデなし)* | **`#6` (`Flat`)** | `50.83%` | `57.86%` *(教師あり)* | `63.27%` *(教師あり)* | `65.81%` *(教師あり)* | `59.44%` *(見かけの平均)* |
| **長尺領域 (`c >= 255`) `L0` 正解率 ↑** | **`#8` (`Grouped`)** | **`53.79%` (`+2.23%` 成長 🏆)** | `53.98%` (`+4.18%`) | `54.86%` (`+3.26%`) | `53.89%` (`+3.25%`) | **`54.13%`** |
| *(32小節超・64小節文脈の恩恵)* | **`#6` (`Flat`)** | `49.72%` (**`-1.11%` 劣化**) | `58.04%` *(教師あり)* | `62.47%` *(教師あり)* | `64.69%` *(教師あり)* | `58.73%` *(見かけの平均)* |

この分解データから、PPL 逆転現象を引き起こした **3つの決定的メカニズム** が完全に特定された：

1. **メカニズム①：「未来予測の不確実性コスト4重課税」vs「チャンク内カンニング穴埋め効果」（PPL の錯覚）**
   - `MidiVision` のエンコーダ射影層 `Linear(3072, 128)` は、4トラック全体の音響特徴を $G=4$ グループすべてに大域混合してから分割量子化するため、同一チャンク内の `g=0, 1, 2, 3` は互いに極めて強い相互情報量 $I(z_{0,g}; z_{0,<g} \mid z_{<c})$ を共有している。
   - `#8 (Grouped)` は未知の次チャンクの `g=0..3` を **一切のヒントなしで4個同時に並列予測** するため、「次の半拍でどのコード・ドラムが来るか分からない」という本質的な未来予測エントロピー（`~2.15 nats`）を **4回重複して支払わされる**。
   - 対して `#6 (Flat)` は、その本質的な未来予測コストを **`g=0` のたった1回（`2.2548 nats`）しか支払っていない**。残りの `g=1, 2, 3` では「すでに同じ時刻の正解 `g=0` トークンが教師強制で与えられた状態」でチャンク内の残りピースを埋めるパズルを解いているため、損失が **`2.2548 → 1.7977 → 1.5146 → 1.4267 nats`** と劇的に低下する。全16ステップで単純平均すると、この容易な「チャンク内穴埋めステップ（全体の75%）」が平均損失を `1.5163` (`PPL 4.70`) まで押し下げていたに過ぎない。

2. **メカニズム②：「真の次チャンク予測 (`g=0`)」と長尺構造把握における `Grouped` の本質的勝利**
   - チャンク内の正解リークがゼロである **先頭トークン `g=0` 同士で純粋勝負させると、なんと損失 (NLL) でも正解率でも `#8 (2.1453 nats)` が `#6 (2.2548 nats)` に明確に勝っている**。
   - コンテキスト長上限に達しない前半32小節以内 (`c < 255`) に限定しても `#8` (`51.56%`) > `#6` (`50.83%`) であり、さらに32小節を超える中盤〜後半 (`c >= 255`) では、`#8` が **64小節の長距離文脈**（Aメロ・サビの反復構造）を活用して正解率を **`51.56% → 53.79% (+2.23 pt)`** と大きく伸ばす一方、32小節しか見えない `#6` は **`50.83% → 49.72% (-1.11 pt)`** へと失速し、リードが **`+4.07 pt`** まで拡大する。
   - パラメータ配分の観点でも、`#6` は `135M` バックボーンの表現容量の大部分を「チャンク内16ステップの局所コードブック相関の暗記」に消費しているのに対し、`#8` と `#4` は全12層の Attention 容量を **チャンク間の時間的・音楽的展開の学習に100%集中投資** できている。

3. **メカニズム③：推論時のチャンク内 Exposure Bias（16ステップのドミノ倒し誤差伝播）**
   - `#6 (Flat)` は学習時、「直前の `g=0` が100%正解である」という温室条件で `g=1..3` を学習している（表7.2 `Step-level TF` での `L0` 一致率 `60.22%`）。
   - しかし実際の生成 (`Chunk-level TF`) では、音楽の多様な展開ゆえに先頭 `g=0` は約50%の確率で別解を選ぶ。その瞬間、正解 `g=0` に強く依存していた `#6` の残り15ステップがドミノ倒し的に連鎖崩壊（Exposure Bias）を起こし、実推論時の `L0` 一致率は **`60.22% → 51.73% (-8.49 pt)`** へ急落して `#8` (`55.53%`) にも `#4` (`52.93%`) にも逆転される。
   - 対照的に `#4`（**1チャンクわずか2ステップ**）および `#8`（**4ステップ**）はチャンク内の自己回帰段数が極小であるため誤差蓄積が最小化され、最高精度のピアノロール復元を達成した。

---

### 7.4 3つの設計軸に対する最終結論（パレート最適アーキテクチャの確定）

> [!IMPORTANT]
> **MORTM5 アブレーション実験の最終結論（提案手法 `Grouped Soft-Token + Separated` の完全勝利 🏆）**:
>
> 1. **軸3（RVQ残差レイヤー間の語彙空間）：`Separated ≫ Shared`（全指標で完全決着）**
>    - `#8 (Separated)` は `#7 (Shared)` に対して、トークン損失で **`-0.3333 nats`**、物理ピアノロール復元でも **Note F1 `+2.87 pt` (`78.52% vs 75.65%`)・Onset F1 `+3.11 pt` (`71.85% vs 68.74%`)・Velocity MAE `-0.0196`** と全指標で圧倒した。粗い骨格を担う `L0` と微細残差を担う `L1..L3` の語彙埋め込み分離は必須である。
>
> 2. **軸2（系列モデリング構造）：`Grouped Soft-Token` (`#4`, `#8`) が `Flat` (`#6`) をパレート支配（大逆転勝利 🏆）**
>    - 当初のトークン PPL 評価では「精度なら `Flat (#6)`、速度・長尺なら `Grouped (#8)` のトレードオフ」と予想されたが、統一物理ピアノロール検証により **トレードオフではなく `Grouped Soft-Token` の完全上位互換（パレート支配）** であることが実証された。
>    - `#4` / `#8` は `#6 (Flat)` と比較して、**学習計算量が `1/8 〜 1/4`**、**推論ステップ数が `1/8 〜 1/4`（4〜8倍高速）**、**見渡せる楽曲文脈が `2倍`（約64小節）** でありながら、実際の次チャンク予測精度でも **Note F1 `+1.26 〜 +1.32 pt`・Onset F1 `+1.72 〜 +2.87 pt`・Velocity MAE 改善** という全面勝利を収めた。
>
> 3. **軸1（ベース・トークナイザー）：`M3 Grouped (#4)` と `M4 Grouped (#8)` が双璧の SoTA（用途別最適解）**
>    - **`#4` (`MORTM5_M3_Grouped_Separated` — 総合 Note/Onset F1・推論速度 第1位 🏆)**:
>      - **Note F1 `78.58%`（1位）**、**Onset F1 `73.00%`（`#8` 比 `+1.15 pt` で断トツ1位）**、**Velocity MAE `0.2596`（1位）**、3〜4トラック密アンサンブル領域でも最高精度 (`79.44% / 75.67%`) を達成。
>      - しかも **1チャンクわずか `2 steps`（`Flat` の8倍高速、学習 FLOPs `0.70 × 10¹⁸` で `#6` の8分の1）** という驚異的な計算効率を誇り、リズムのキレ（Onset）・ベロシティ表現・リアルタイム生成速度を重視する MORTM5 の第一選択となる。
>    - **`#8` (`MORTM5_M4_Grouped_Separated` — Note Precision・疎アンサンブル・粗層構造予測 同率1位 🏆)**:
>      - **Note F1 `78.52%`（`#4` と `0.06 pt` 差の同率トップ圏）**、**Note Precision `76.41%`（全モデル最高＝不要なミスタッチ発音が最も少ない）**、**`1〜2 Trk` 疎領域 Note F1 `75.74%`（1位）**、**RVQ `L0` 一致率 `55.53%`（1位）** を達成。和声の精緻さとミスタッチ抑制を重視する場合の最適解となる。
>    - なお、3軸の優劣関係が物理空間でも完全に分離・確定したため、下位組み合わせ（`#1, #2, #3, #5`）の追加学習省略の妥当性も改めて裏付けられた。

---

## 8. 各モデルの学習推移ログ（詳細記録）

### 8.1 Ablation `#6` (`MORTM5_M4_Flat_Separated`) — **100% 完走 (`Token PPL 4.70` / `Chunk-TF Note F1 77.26%`)**
- **W&B Run**: [`MORTM5-Ablation / ltbq31qp`](https://wandb.ai/ayato_dayo-nihon-university/MORTM5-Ablation/runs/ltbq31qp)
- **ウィンドウ / ステップ数**: `train = 2,111,255 windows` (`1,031` Outer Batches / `32,989` Steps, `batch_size=16, accum_steps=4`), `val = 12,815 windows`
- **保存先**: Google Drive `/content/drive/MyDrive/MORTM5_checkpoints/6/` & Mac ローカル `out/mortm5_ablation/6/MORTM5.MORTM5_M4_Flat_Separated_muon_cos_latest6.pth`

| Optimizer Step | 学習トークン数 (`axis/tokens`) | `train/loss` | `val/loss` | `val/ppl` | 備考 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `256` | `53.55M` | `9.9393` | `9.3239` | `11202.95` | Warmup 開始（フル語彙 `32,901` Softmax） |
| `2560` | `534.06M` | `4.6966` | `2.6290` | `13.86` | Warmup 完了直後で一気に `PPL < 14` |
| `5120` | `1067.97M` | `2.2820` | `2.1119` | `8.26` | **1B トークン通過**（トークン PPL 上で `#4` 完走値 `8.40` を突破） |
| `8256` | `1721.41M` | `1.9861` | `1.9162` | `6.80` | **25% マイルストーン**（`#8` 完走と同一 FLOPs `1.40e18`） |
| `10560` | `2200.72M` | `1.8838` | `1.8450` | `6.33` | **32% (`2.20B`) 地点でトークン平均損失が `#8` (`1.8458`) に到達** |
| `12864` | `2681.02M` | `1.8221` | `1.7831` | `5.95` | `PPL < 6.0` 突破 |
| `16512` | `~3442M` | `~1.74` | `~1.71` | `~5.53` | **50% マイルストーン** |
| `20096` | `4190.00M` | `1.6838` | `1.6538` | `5.23` | **4B トークン通過** |
| `23680` | `4937.94M` | `1.6368` | `1.6067` | `4.99` | `PPL < 5.0` 突破 (`~4.94B tokens`) |
| `24768` | `5164.59M` | `1.6213` | `1.5953` | `4.93` | **75% マイルストーン** |
| `28864` | `6018.41M` | `1.5837` | `1.5631` | `4.77` | **6B トークン通過** |
| `31680` | `6605.81M` | `1.5673` | `1.5507` | `4.71` | 96.0% 地点（チェックポイントから再開接続） |
| **`32989`** | **`6878.84M`** | **`1.5615`** | **`1.5484`** | **`4.70`** | **100% 完走（トークン PPL 最小・物理復元 3位）** |

---

### 8.2 Ablation `#8` (`MORTM5_M4_Grouped_Separated`) — **100% 完走 (`Token PPL 6.33` / `Chunk-TF Note F1 78.52%` Co-SoTA 🏆)**
- **W&B Run**: [`MORTM5-Ablation / spiwak1h`](https://wandb.ai/ayato_dayo-nihon-university/MORTM5-Ablation/runs/spiwak1h)（初回途中記録: [`845q8js0`](https://wandb.ai/ayato_dayo-nihon-university/MORTM5-Ablation/runs/845q8js0)）
- **ウィンドウ / ステップ数**: `train = 1,318,941 windows` (`645` Outer Batches / `20,609` Steps), `val = 8,069 windows`
- **保存先**: Google Drive `/content/drive/MyDrive/MORTM5_checkpoints/8/` & Mac ローカル `out/mortm5_ablation/8/MORTM5.MORTM5_M4_Grouped_Separated_muon_cos_latest.pth`

| Optimizer Step | 学習トークン数 (`axis/tokens`) | `train/loss` | `val/loss` | `val/ppl` | 備考 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `160` | `53.73M` | `7.5039` | `7.1240` | `1241.36` | Warmup 開始（`DecodeCodebookHead` 導入後） |
| `1600` | `537.33M` | `5.5165` | `3.9067` | `49.74` | PPL 50 突破 |
| `3200` | `1072.28M` | `3.8781` | `3.1175` | `22.59` | 1B トークン通過 |
| `6272` | `2101.02M` | `2.6858` | `2.4312` | `11.37` | 2B トークン通過 |
| `7872` | `2634.58M` | `2.4379` | `2.2403` | `9.40` | `PPL < 10` 突破 |
| `10304` | `3447.45M` | `2.2238` | `2.0709` | `7.93` | **50% マイルストーン** (`0.70e18 FLOPs`) |
| `15648` | `5237.53M` | `2.0171` | `1.8981` | `6.67` | **75% マイルストーン付近** |
| **`20609`** | **`6897.96M`** | **`1.9460`** | **`1.8458`** | **`6.33`** | **100% 完走（物理 Note Precision `76.41%`・`L0` 一致率 `55.53%` 1位 🏆）** |

---

### 8.3 Ablation `#4` (`MORTM5_M3_Grouped_Separated`) — **100% 完走 (`Token PPL 8.40` / `Chunk-TF Note F1 78.58%` 総合1位 🏆)**
- **W&B Run**: [`MORTM5-Ablation / j93yl9si`](https://wandb.ai/ayato_dayo-nihon-university/MORTM5-Ablation/runs/j93yl9si)
- **ウィンドウ / ステップ数**: `train = 1,320,087 windows` (`645` Outer Batches / `20,627` Steps), `val = 8,074 windows`
- **保存先**: Google Drive `/content/drive/MyDrive/MORTM5_checkpoints/4/` & Mac ローカル `out/mortm5_ablation/4/MORTM5.MORTM5_M3_Grouped_Separated_muon_cos_latest.pth`

| Optimizer Step | 学習トークン数 (`axis/tokens`) | `train/loss` | `val/loss` | `val/ppl` | 備考 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `160` | `53.82M` | `7.5034` | `7.1462` | `1269.26` | Warmup 開始 |
| `3200` | `1075.84M` | `4.2711` | `3.4429` | `31.28` | 1B トークン通過 |
| `6304` | `2118.28M` | `2.9727` | `2.7912` | `16.30` | 2B トークン通過 |
| `10976` | `3687.48M` | `2.4740` | `2.3645` | `10.64` | 50% 通過後 |
| `12576` | `4223.89M` | `2.3752` | `2.2889` | `9.86` | `PPL < 10` 突破 |
| `15648` | `5256.80M` | `2.2646` | `2.1919` | `8.95` | 75% マイルストーン付近 |
| **`20627`** | **`6927.47M`** | **`2.1834`** | **`2.1282`** | **`8.40`** | **100% 完走（物理 Note F1 `78.58%`・Onset F1 `73.00%`・速度 総合1位 🏆）** |

---

### 8.4 Ablation `#7` (`MORTM5_M4_Grouped_Shared`) — **100% 完走 (`Token PPL 8.84` / `Chunk-TF Note F1 75.65%`)**
- **W&B Run**: [`MORTM5-Ablation / 6l5e1e90`](https://wandb.ai/ayato_dayo-nihon-university/MORTM5-Ablation/runs/6l5e1e90)
- **ウィンドウ / ステップ数**: `train = 1,318,941 windows` (`645` Outer Batches / `20,609` Steps), `val = 8,069 windows`
- **保存先**: Google Drive `/content/drive/MyDrive/MORTM5_checkpoints/7/` & Mac ローカル `out/mortm5_ablation/7/MORTM5.MORTM5_M4_Grouped_Shared_muon_cos_latest.pth`

| Optimizer Step | 学習トークン数 (`axis/tokens`) | `train/loss` | `val/loss` | `val/ppl` | 備考 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `0` | `0.00M` | — | `7.8550` | `2578.54` | 初期状態 |
| `3104` | `1039.98M` | `3.5520` | `3.3960` | `29.85` | 1B トークン通過 |
| `6112` | `2047.41M` | `3.0286` | `2.9869` | `19.82` | 2B トークン通過 |
| `10272` | `3436.71M` | `2.7233` | `2.5620` | `12.96` | 50% マイルストーン |
| `14752` | `4938.30M` | `2.4465` | `2.2976` | `9.95` | `PPL < 10` 突破 |
| `15392` | `5151.95M` | `2.4265` | `2.2749` | `9.73` | 75% マイルストーン |
| **`20609`** | **`6897.96M`** | **`2.3108`** | **`2.1791`** | **`8.84`** | **100% 完走（Shared ベースライン）** |
