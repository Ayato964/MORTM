# MORTM5 (MidiVision) 量子化粒度・ハイパーパラメータ比較実験計画書

## 1. 背景と目的

MORTM5 の前段モジュールである **`MidiVision`** は、最大4トラック（Drums + 3 Melodic Tracks）のポリフォニックMIDIピアノロールチャンク（形状: `[8, 128, 24]`、1チャンク＝24ステップ＝0.5拍）を、深層 ConvNeXt エンコーダおよび **Grouped Residual Vector Quantization (Grouped-RVQ)** を用いて離散トークン列へ圧縮・再構成するモデルである。

基準モデル（**Model 1: Baseline**）の学習およびアブレーション診断により、以下の重要な事実が判明している：

1. **CNN（Encoder / Decoder）自体の表現力は4トラック同時発音でも極めて高い**
   - 学習済み Model 1 において、RVQ量子化のみをバイパスして正規化潜在ベクトル `z_norm` をそのまま Decoder に入力すると、3〜4トラック同時発音チャンクであっても **`Onset F1 = 97.32%` (`Recall: 98.9%`, `Precision: 95.8%`)**、**`Note F1 = 90.19%` (`Recall: 97.0%`, `Precision: 84.2%`)** に達する。
2. **マルチトラック時の精度低下（「音の滲み・過剰発音」）の原因は RVQ の量子化粒度にある**
   - 1トラック単独入力時は RVQ の量子化残差エラーが `0.7%` に収まり、**`Onset F1 = 98.52%` / `Note F1 = 94.24%`** を達成する。
   - 一方、3〜4トラック同時入力時は、1チャンクあたり `8 tokens (80 bits)`・1グループあたり `128次元` という粗い量子化ボトルネックを全トラックで共有するため、未量子化エネルギーが `49.8%` 残り、複数の和音やSustainが代表ベクトルへ平均化される「量子化ブラー」が発生する。
   - この結果、**Recall（`86.8%`）は高いまま、Precision（`67.8%`）が低下する（＝本来鳴っていない隣接領域まで確率が滲み出て過剰に鳴らしてしまう）** 現象が生じる。

本実験では、論文における定量的・定性的な根拠（Ablation Study）を確立するため、**量子化の次元密度・グループ分割数・残差段数を体系的に変化させた全4モデル** を同一条件で学習・比較する。

---

## 2. 比較対象モデル一覧（Ablation Matrix）

すでに学習完了している **Model 1** を Baseline とし、新たに **Model 2, Model 3, Model 4** の3モデルを学習する。

| モデルID | モデル呼称 | グループ数 `vq_groups` ($G$) | 残差段数 `encoder_layer` ($L$) | 語彙数 `codebook_size` ($K$) | 潜在次元 `encoder_wout` ($D$) | 1グループ次元 `codebook_dim` ($D/G$) | トークン数 / チャンク (1小節あたり) | ビットレート / チャンク | ステータス |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1** | **Baseline** (`G4-L2-D512-K1024`) | `4` | `2` | `1024` ($10\text{ bit}$) | `512` | `128` | **8 tokens** (64 / bar) | `80 bits` | **完了済** (`Val 0.8683`) |
| **Model 2** | **Compact-Dim** (`G4-L2-D128-K2048`) | `4` | `2` | **`2048`** ($11\text{ bit}$) | **`128`** | **`32`** | **8 tokens** (64 / bar) | `88 bits` | 未実施 |
| **Model 3** | **Fine-Group** (`G8-L2-D512-K1024`) | **`8`** | `2` | `1024` ($10\text{ bit}$) | `512` | **`64`** | **16 tokens** (128 / bar) | `160 bits` | 未実施 |
| **Model 4** | **Deep-RVQ** (`G4-L4-D512-K1024`) | `4` | **`4`** | `1024` ($10\text{ bit}$) | `512` | `128` | **16 tokens** (128 / bar) | `160 bits` | 未実施 |

---

## 3. 論文で検証する3つのリサーチクエスチョン (Research Questions)

### RQ1: 系列長（8トークン/チャンク）を固定したまま量子化ブラーを抑えられるか？（Model 1 vs. Model 2）
- **課題**: 後段の自己回帰 Transformer（MORTM5）の計算量とコンテキスト長を守るには、1チャンクあたりのトークン数を `8 tokens`（1小節64トークン）に据え置くことが望ましい。
- **仮説**: Model 1 では 1グループあたり `128次元` の空間を量子化しているため「次元の呪い」によりコードブックが空間を密に埋められていない（予備実験で単純に `128次元` のまま語彙数だけを増やしても改善しないことを確認済み）。ViT-VQGAN / DAC (Descript Audio Codec) の知見に基づき、`encoder_wout` を `512 → 128`（1グループあたり **`32次元`**）へ凝縮しつつ語彙数を `2048` に拡張することで、**Transformer の系列長を一切伸ばさずに量子化密度を高め、過剰発音（Precision低下）を改善できる**。

### RQ2: トークン予算（ビットレート）の倍増はマルチトラック再構成をどこまで引き上げるか？（Model 1 vs. Model 3, 4）
- **課題**: 4トラックが同時に複雑なポリフォニーを奏でる区間では、1チャンク `80 bits`（1トラックあたり平均 `20 bits`）の情報量自体が理論限界に近づく。
- **仮説**: トークン数を `8 → 16 tokens/chunk`（`80 bits → 160 bits`）に倍増させることで、CNN上限値（`Onset F1 = 97.3%`, `Note F1 = 90.2%`）にどこまで漸近できるか、また「系列長2倍のコストに見合う精度向上が得られるか」を定量化する。

### RQ3: 「横の分割（Grouped VQ）」と「縦の深さ（Residual VQ）」のどちらがマルチトラックMIDIに適しているか？（Model 3 vs. Model 4）
- **課題**: 同じ `16 tokens / chunk`（`160 bits`）を実現する方法として、グループ数を増やす方法（Model 3: `G=8, L=2`）と、残差量子化の段数を深くする方法（Model 4: `G=4, L=4`）の2通りが存在する。
- **仮説**:
  - **Model 3 (`G=8, L=2`)** は1グループあたりの次元が `64次元` に半減し、独立した8個の部分空間で異なる楽器・音域を分担できる。
  - **Model 4 (`G=4, L=4`)** は1〜2段目で大局的な和音・ドラム骨格を捉え、3〜4段目の残差で微細なノートやSustainの輪郭を段階的に引き算・補正できる。
  - 両者を同一ビットレートで直接対決させることで、ポリフォニックMIDIトークナイザにおける最適設計指針を明らかにする。

---

## 4. 統制された学習プロトコル（全モデル共通条件）

実験の公平性を担保するため、`vq_groups` / `encoder_layer` / `codebook_size` / `encoder_wout` 以外の全要素を完全に固定する。

1. **データセット**:
   - GigaMIDI Dataset (`GMD`) 全809,180曲（マニフェスト: `configs/train/mortm5/gmd_paths.json`、固定シード `seed=42` による Train/Val 分割）
   - ドラム除外のオンラインランダム転調（`-6 〜 +5` 半音）を適用
2. **ネットワーク骨格**:
   - Encoder / Decoder ともに5段階深層 ConvNeXt 構造（`first_channel=64`, `mid_channel=128`, `max_channel=256`, `bottle_channel=128`）
   - RVQ直前にグループ独立の無アフィン正規化 `nn.LayerNorm(codebook_dim, elementwise_affine=False)` を適用
   - RVQ EMA設定: `decay = 0.999`, `threshold_ema_dead_code = 0.05`, `kmeans_init = True`
3. **損失関数 (`MidiVisionLoss`)**:
   - Note-Active 損失: Weighted BCE (`pos_weight=10.0`) + Soft Dice Loss
   - Onset + Velocity 損失: Continuous Target Weighted BCE + Onset Soft Dice + Sustain-Onset 同期損失（`t >= 1` 立ち上がりエッジ同期 & ゴーストOnset抑制） + Masked Smooth L1 Velocity Loss (`vel_weight=5.0`)
   - Commitment 損失: `commit_weight = 0.25`
4. **最適化スケジュール**:
   - バッチサイズ: `batch_size = 512` / GPU（2 GPU DDP、グローバルバッチ `1,024 chunks/step`）
   - 学習ステップ数: 約 `27,000 steps`（Outer Batch `215 / 1581`、約 `3.5億 tokens` 相当 — Model 1 と同一の収束地点）
   - 学習率 (`AdamW`, `betas=(0.9, 0.98)`, `weight_decay=1e-2`):
     - Phase 1（初期学習: `Outer Batch 1 〜 145` / `~18,000 steps`）: `lr = 4.0e-4`
     - Phase 2（微調整収束: `Outer Batch 146 〜 215` / `~27,000 steps`）: `lr = 1.0e-4`

---

## 5. 評価指標と分析方法

各モデルの学習完了後、検証セット（Unseen GMD Val Set）を用いて以下の指標を算出し、論文用の比較表を作成する。

1. **全体およびトラック密度別（1〜2トラック vs. 3〜4トラック同時発音）の再構成精度**:
   - **Onset F1 / Precision / Recall**（発音タイミングの正確性と過剰発音の少なさ）
   - **Note (Sustain) F1 / Precision / Recall**（音の長さ・和音構成の正確性と量子化ブラーの少なさ）
   - **デコード後 MIDI ノート単位の F1**（`mortm/utils/convert_pianoroll.py` の `_flat_to_quantized_notes` 適用後）
   - **Velocity MAE**（発音ベロシティの平均絶対誤差）
2. **量子化品質の内部指標**:
   - **相対量子化残差エラー (`Relative Quantization Error`)**: $\|z_{\text{norm}} - x_q\|_2^2 / \|z_{\text{norm}}\|_2^2$
   - **コードブック利用率 (`Codebook Usage / Perplexity`)**: 全コードブック中の有効語彙数とエントロピー換算Perplexity
3. **定性評価（`MidiVision Studio` での視覚・聴覚比較）**:
   - フロントエンド（`midivision_studio`）に **「モデル切り替え機能（Model 1 / 2 / 3 / 4）」** を搭載し、同一MIDI楽曲に対して：
     - ピアノロール上の **赤（Onset差分）** および **黄（Sustainはみ出し差分）** の減少具合
     - FluidSynth アコースティック音源での「鳴らし過ぎ（音の濁り）」の解消度合い
   をリアルタイムに切り替えて検証・可視化する。

---

## 6. 設定ファイルと成果物の管理構成

| モデル | モデル設定JSON | 保存先ウェイト (`.pth`) |
| :--- | :--- | :--- |
| **Model 1 (Baseline)** | `configs/models/mortm5/default.json` | `out/models/mortm5/MORTM5.MIDIVision_M1_G4L2_D512_K1024.pth` (既存 `MORTM5.MIDIVision.pth`) |
| **Model 2 (Compact-Dim)** | `configs/models/mortm5/m2_compact_g4l2_d128_k2048.json` | `out/models/mortm5/MORTM5.MIDIVision_M2_G4L2_D128_K2048.pth` |
| **Model 3 (Fine-Group)** | `configs/models/mortm5/m3_finegroup_g8l2_d512_k1024.json` | `out/models/mortm5/MORTM5.MIDIVision_M3_G8L2_D512_K1024.pth` |
| **Model 4 (Deep-RVQ)** | `configs/models/mortm5/m4_deeprvq_g4l4_d512_k1024.json` | `out/models/mortm5/MORTM5.MIDIVision_M4_G4L4_D512_K1024.pth` |

---

## 7. 実行手順（ロードマップ）

1. **Step 1**: Model 2, Model 3, Model 4 用のモデル設定JSONファイルを `configs/models/mortm5/` に作成する。
2. **Step 2**: **Model 2 (`Compact-Dim`)** の学習を実行・保存する（系列長維持での改善効果を最優先で確認）。
3. **Step 3**: **Model 3 (`Fine-Group`)** および **Model 4 (`Deep-RVQ`)** の学習を順次実行・保存する。
4. **Step 4**: 全4モデルの統一ベンチマークスクリプトを実行し、論文掲載用の定量評価テーブル（Markdown / LaTeX表）を出力する。
5. **Step 5**: `MidiVision Studio` に4モデル切り替えUIを追加し、定性比較および論文用図版（Figure）の作成を行う。
