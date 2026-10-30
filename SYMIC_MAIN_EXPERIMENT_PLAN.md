# SyMIc (Symbolic MIDI Codebook) 本実験計画書
### — 16小節固定分割・無制限窓長・163.6M共通バックボーンによる対外パラダイム比較と長尺自由生成検証 —

## 1. 論文全体構成と本実験（Stage 3）の位置づけ

本研究では、マルチトラック記号音楽（MIDI）を2次元ピアノロール多様体上の深層残差ベクトル量子化（Deep Grouped-RVQ）によって固定レート離散コードへ変換する新トークナイザ **`SyMIc` (Symbolic MIDI Codebook)**、および並列グループ埋め込みを分散保持集約して自己回帰生成する **`Grouped Soft-Token Transformer`** を提案する。

論文全体の実験プログラムは以下の **3段階（3-Stage Experimental Program）** で構成され、本計画書はその最終本丸である **Stage 3（本番スケール対外比較・複数小節自由生成評価）** の統制仕様を定めるものである：

1. **Stage 1：Codec 内部設計アブレーション（完了済 — [MIDIVISION_EXPERIMENT_PLAN.md](file:///Users/dayo/My_Projects/MORTM/MIDIVISION_EXPERIMENT_PLAN.md)）**
   - 基準モデル `M1` から `M2 (Compact-Dim)`, `M3 (Fine-Group: G8-L2)`, `M4 (Deep-RVQ: G4-L4)` の4モデルを統一学習・比較。
   - サブ空間次元凝縮（$128 \to 32$ 次元）による「量子化ブラー」の解消と、4段深層残差 `SyMIc-G4L4`（`M4`）による最高再構成精度（**`Note F1 = 93.48%`**, **`Onset F1 = 96.44%`**, **`Velocity MAE = 0.1111`**）を実証。
2. **Stage 2：系列モデリング統制アブレーション @ `85M`（完了済 — [MORTM5_ABLATION_EXPERIMENT_PLAN.md](file:///Users/dayo/My_Projects/MORTM/MORTM5_ABLATION_EXPERIMENT_PLAN.md)）**
   - `84.93M`（Non-Embedding）の小規模バックボーン上で `M3/M4 × Flat/Grouped × Shared/Separated` の OFAT 比較を完走。
   - `Separated ≫ Shared`、および **`Grouped Soft-Token` による `Flat` のパレート支配**（`1/4〜1/8` の計算量・`4〜8倍` の推論速度・`2倍` の文脈長でありながら物理次チャンク予測 F1 でも優位）を実証し、「PPL パラドックス」と「残差キメラ現象」を数理的に解明。
3. **Stage 3：本番スケール対外パラダイム比較 & 複数小節自由生成評価 @ `163.6M`（本計画書）**
   - Stage 2 の統制アブレーションで確立した主役構成 **`SyMIc-G4L4` (`Grouped Soft-Token + Separated`)** を本番サイズ（Non-Embedding **`163.60M`**）へスケールアップし、モデル容量ボトルネックを解消する。
   - その上で、既存の記号音楽トークン化における2大標準パラダイムである **① フラットイベント型（`Multitrack-REMI`）** および **② 音符単位バンドル型（`Octuple-MMT` / Compound Word 系）** を同一データ・同一バックボーン・同一16小節セグメント条件で学習し、**コンテキスト効率・計算効率・複数小節の自由生成品質** における `SyMIc` の本質的優位性を実証する。

---

## 2. なぜ「3パラダイム対決」なのか？（科学的仮説の分離）

もし `Multitrack-REMI` のみを比較対象とした場合、査読者から *「提案手法の高速性やコンテキスト圧縮は、単に1ステップに複数トークンを束ねた効果であり、既存の Compound Word (Hsiao et al., AAAI 2021) や OctupleMIDI / MMT (Zeng et al., ACL 2021; Dong et al., ICASSP 2023) と本質的に同じではないか？」* という疑問が必ず生じる。

本実験では **`Multitrack-REMI`（属性直列）** と **`Octuple-MMT`（音符集約）** の双方をベースラインに据えることで、**「イベント単位の属性バンドル」と「2Dアンサンブル空間の固定時間グリッド Deep-RVQ 集約（`SyMIc`）」の決定的な構造差** を厳密に分離する：

| 比較軸 | **Paradigm 1: Flat Event**<br>`Multitrack-REMI` | **Paradigm 2: Note-Bundled Event**<br>`Octuple-MMT` (CP系) | **Paradigm 3: Frame Deep-RVQ (提案)**<br>**`SyMIc-G4L4`** |
| :--- | :--- | :--- | :--- |
| **1自己回帰ステップの単位** | 1原子イベントトークン<br>(`<BAR>`, `<POS>`, `<SLOT>`, `<PITCH>`, ...) | **1音符タプル（6属性同時集約）**<br>`(bar, pos, slot, pitch, dur, vel)` | **半拍（`0.5 beat`）4トラック全体の1残差層**<br>($G=4$ 並列サブ空間の Soft-Token) |
| **埋め込み集約・出力構造** | 単一 Embedding + 単一 Softmax Head | **6属性 `1/√6` 加算 + 6並列 Head** | **4グループ `1/√4` 加算 + 4並列 Head** |
| **系列長計算量（対音符数 $N_{\text{notes}}$）** | **$O(k \cdot N_{\text{notes}})$** ($k \approx 4.3$) | **$O(N_{\text{notes}})$**（ポリフォニー密度に比例膨張） | **$O(1)$ 完全定数**（音数・編成に完全不変） |
| **同時発音（和音・4楽器）の生成** | 1音・1属性ずつ人工順序で直列展開 | 同時刻の複数音も人工ソート順で1音ずつ直列展開 | 2D ConvNeXt が4楽器の垂直和声・リズムを同時統合し、`L0→L3` で粗密生成 |
| **時間軸・小節構造の進行** | `<BAR>` / `<POS>` の自己回帰予測に依存（長尺で拍ドリフト・密度暴走リスク） | `bar` / `pos` ヘッドの独立予測に依存（同時予測時の属性衝突・順序破綻リスク） | **物理クロック（`1 chunk = 0.5 beat`）にハードロック**（拍ズレ・小節崩壊が原理的にゼロ） |

### 2.1 GigaMIDI 実データにおける「16小節あたり系列ステップ数」の実測分布

[`data/gmd/eval.json`](file:///Users/dayo/My_Projects/MORTM/data/gmd/eval.json) 先頭200曲から切り出した計 `961` 個の16小節セグメント（うち3〜4トラック編成 `777` セグメント）に対し、[`extract_quantized_notes_and_slots`](file:///Users/dayo/My_Projects/MORTM/mortm/utils/convert_pianoroll.py#L931-L973) を用いて実測した **同一の音楽的16小節を表現するのに要する Transformer ステップ数 $S$** は以下の通りである：

| トークン化手法 | 全編成 平均 $S$ (`N=961`) | 3–4Trk 平均 $S$ (`N=777`) | 3–4Trk 中央値 (`p50`) | 密集90%点 (`p90`) | 密集95%点 (`p95`) | 密集99%点 (`p99`) | 最大長 (`max`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`SyMIc-G4L4` (提案)** | **`518`** | **`518`** | **`518`** | **`518`** | **`518`** | **`518`** | **`518` (`1.00×` 定数)** |
| **`Octuple-MMT`** | `472` | `515` | `460` | `964` | `1,198` | `1,492` | `1,925` (**`3.72×` 膨張**) |
| **`Multitrack-REMI`** | `2,041` | `2,220` | `2,018` | `4,121` | `5,098` | `6,245` | `7,954` (**`15.36×` 膨張**) |

> [!IMPORTANT]
> **実測分布が示す決定的な構造差**:
> - `Octuple-MMT` は疎な単旋律区間を含む全体平均こそ `472` ステップであるが、本研究の主眼である **アンサンブル密集領域（`p90〜max`）では `964 → 1,925` ステップ（`SyMIc` の最大 `3.7倍`）へと急膨張** し、さらに1ステップごとに1音ずつしか進めないため同時発音和音を直列分解せざるを得ない。
> - `Multitrack-REMI` に至っては3〜4トラック平均で **`2,220` ステップ（`4.29倍`）**、上位5%では **`5,098〜7,954` ステップ（`10〜15倍`）** に達する。一般的な固定コンテキスト窓（例：`2048` や `4096`）を設けた場合、アンサンブルの盛り上がりの最中でコンテキスト上限に達して楽曲構造が切断される。
> - **`SyMIc-G4L4` は、どれほど重厚な4パート・トゥッティであっても常に厳密に `518` ステップ（`5` prefix + `512` RVQ + `1` EOS）で16小節を完結** させ、パディング率 `0%` で GPU TensorCore を100%稼働させる。

---

## 3. 情報量と前処理の完全対称化（Zero-Confound Information Parity）

トークナイザ間の比較において「前処理の量子化精度やトラック選択の違い」という交絡因子を完全に排除するため、全3モデルの学習・検証データはすべて [`extract_quantized_notes_and_slots`](file:///Users/dayo/My_Projects/MORTM/mortm/utils/convert_pianoroll.py#L931-L973) が出力する **完全に同一の4スロット楽器コード `slot_codes` `[4]` および量子化ノート配列 `(slot, start, end, pitch, velocity)`** を唯一の真値源（Single Source of Truth）として構築する。

### 3.1 共通データ基盤と16小節セグメント境界の一致保証
- **コーパス**: GigaMIDI Dataset (`train`: 有効 **`801,838` 曲** / スキップ `7,342` 曲、`val`: 有効 **`4,950` 曲** / スキップ `50` 曲、[`m4_g4l4/train_state.json`](file:///Users/dayo/My_Projects/MORTM/out/mortm5_tokens/m4_g4l4/train_state.json) と曲単位で1対1完全一致）。
- **時間解像度**: 1小節 ＝ `STEPS_PER_BAR = 192` ステップ（1拍 ＝ `48` ステップ、1チャンク ＝ `CHUNK_STEPS = 24` ステップ ＝ `0.5` 拍）。
- **16小節固定分割（Non-Overlapping 16-Bar Split）**:
  - 各楽曲 $i$（総チャンク数 $C_i$）を先頭から **`128 チャンク`（＝ 16小節 ＝ `3,072` 時間ステップ）単位** の非重複セグメント $[c_0, c_1)$（$c_0 = 128 k, c_1 = \min(C_i, c_0 + 128)$、末尾端数は `min_chunks = 8` ＝ 1小節以上を採用）に分割する。
  - **セグメント境界を跨ぐ持続音（Cross-Segment Sustain）の厳密処理**: 前のセグメント（$< s_0$）で発音し当該セグメント内（$> s_0$）まで持続するノートは、[`rasterize_notes_to_chunks_u8`](file:///Users/dayo/My_Projects/MORTM/mortm/utils/convert_pianoroll.py#L976-L1005) のチャンク切り出しと情報量を一致させるため、`REMI` / `Octuple` でもセグメント先頭 `local_step = 0` に残余長 `min(s_1, end) - s_0` でクリップ配置（または Tie 保持）し、3モデルに与えられる音楽的情報量を1ビットの狂いもなく一致させる。
- **共通プレフィックス（ID `0..132`）**:
  - 3モデルすべてで特殊トークン `<PAD>=0`, `<BOS>=1`, `<EOS>=2` および4スロット楽器条件付け `<INST_PROG_0..127>=3..130`, `<INST_DRUMS>=131`, `<INST_EMPTY>=132`（`INST_RVQ_TOKEN_OFFSET = 133`）を完全に共有する。

### 3.2 各モデルの語彙・符号化仕様

1. **`SyMIc-G4L4`（提案手法：[`AblationTransformer2`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py#L613-L1060)）**
   - 既存の事前トークン化バイナリ `out/mortm5_tokens/m4_g4l4/{train,val}_tokens.bin` をそのまま使用し、[`MORTM5MemmapDataset`](file:///Users/dayo/My_Projects/MORTM/mortm/train/datasets.py#L588-L625) にて `position_length = 518`（`max_chunks = 128`, `stride_chunks = 128`, `min_chunks = 8`）を指定することで再トークン化なしで厳密な16小節分割を得る。
   - **語彙サイズ**: `133 + (4 layers × 4 groups × 2048 codes) = 32,901`。

2. **`Octuple-MMT`（音符単位6属性バンドルベースライン）**
   - 各音符をセグメント内相対座標で正準ソート `(bar, pos, slot, pitch)` し、1音符＝1ステップの6属性タプルとして表現する：
     - `bar` (`0..15`: `16` 種)、`pos` (`0..191`: `192` 種)、`slot` (`0..3`: `4` 種)、`pitch` (`0..127`: `128` 種)、`dur` (`312` 種: `1..192` は1ステップ刻み完全可逆、`216..3072` は半拍24ステップ刻みで最大16小節持続まで網羅；GMD 全音符の `99.40%` が `<=192` のため実質完全可逆)、`vel` (`0..31`: `32` 段階均等量子化)。
   - **構造的公平性**: `SyMIc-G4L4`（[`AblationTransformer2`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py#L685-L704)）と**全く同一の分散保持集約式** $\mathbf{h}_t^{\text{in}} = \frac{1}{\sqrt{6}} \sum_{m=0}^{5} \mathbf{e}(a_{t,m})$ で6属性の独立 Embedding を加算し、出力層では6個の独立線形ヘッドから各属性の Cross-Entropy 損失の平均を最小化する。プレフィックス5ステップ（`<BOS>` + 4 `<INST>`）および終端 `<EOS>` も `AblationTransformer2` と同型に付与する。
   - **語彙サイズ**: `133` (共通特殊・楽器) `+ (16 + 192 + 4 + 128 + 312 + 32) = 817`。

3. **`Multitrack-REMI`（フラットイベントベースライン）**
   - [`AblationTransformer1`](file:///Users/dayo/My_Projects/MORTM/mortm/models/mortm5.py#L364-L610) と完全同型の1次元自己回帰モデル。
   - プレフィックス `<BOS> <INST_0..3>` に続き、セグメント内の各小節 $b \in \{0, \dots, B_{\text{seg}}-1\}$ ごとに小節境界 `<BAR>` を出力（休符小節でも出力し小節カウントを厳密保持）。小節内の発音時刻ごとに `<POS_0..191>` を1回出力し、同時刻の音符を `(slot, pitch)` 順に `<SLOT_0..3> <PITCH_0..127> <DUR_0..311> <VEL_0..31>`（4トークン/音）として直列展開し、末尾に `<EOS>` を付す。
   - **語彙サイズ**: `133` (共通特殊・楽器) `+ 1 (<BAR>) + 192 (<POS>) + 4 (<SLOT>) + 128 (<PITCH>) + 312 (<DUR>) + 32 (<VEL>) = 802`。

---

## 4. 統制されたモデルアーキテクチャと学習プロトコル

### 4.1 窓の最大長「無制限」ルール（Zero-Truncation Guarantee）
- **最大系列長による切り詰め（Truncation）は全モデルで一切行わない。**
  - `Multitrack-REMI` で `8,000` トークンを超える超高密度16小節セグメントであっても、1音たりとも切り捨てず全文脈をそのままモデルに入力する。
  - 位置符号化 `RoPE`（$\theta = 10000.0$）は学習可能パラメータを持たないため、[`attention.py`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/attention.py#L318-L350) の `cos/sin` キャッシュを入力バッチの最大系列長に応じてオンデマンドで自動拡張する仕組みを導入し、**パラメータ数を1個も変えずに任意長系列へ厳密対応**させる。
  - 可変長バッチにおける `<PAD>` トークンは [`unpad_input` および `flash_attn_varlen_qkvpacked_func`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/attention.py#L37-L90) により Attention 前後で物理的に除去し、パディングに起因する計算量・VRAM 浪費および勾配汚染を完全にゼロにする。

### 4.2 3モデル完全共通バックボーン仕様（`163.60M` Non-Embedding）

| 項目 | **`SyMIc-G4L4` (提案)** | **`Octuple-MMT` (音符集約)** | **`Multitrack-REMI` (Flat)** |
| :--- | :---: | :---: | :---: |
| **設定JSON（予定）** | `configs/models/mortm5/symic_main/symic_g4l4_160m.json` | `configs/models/mortm5/symic_main/octuple_160m.json` | `configs/models/mortm5/symic_main/remi_160m.json` |
| **Transformer 層数 (`d_layer`)** | `12` | `12` | `12` |
| **隠れ層次元 (`d_model`)** | `1024` | `1024` | `1024` |
| **Attention ヘッド数 (`num_heads`)** | `16` (`head_dim = 64`) | `16` (`head_dim = 64`) | `16` (`head_dim = 64`) |
| **SwiGLU 中間次元 (`dim_feedforward`)** | `3072` (`use_bias = False`) | `3072` (`use_bias = False`) | `3072` (`use_bias = False`) |
| **正規化 / 位置符号化 / Dropout** | `RMSNorm` / 動的長 `RoPE` / `0.1` | `RMSNorm` / 動的長 `RoPE` / `0.1` | `RMSNorm` / 動的長 `RoPE` / `0.1` |
| **Non-Embedding パラメータ数** | **`163,603,456` (`163.60M`)** | **`163,603,456` (`163.60M`)** | **`163,603,456` (`163.60M`)** |
| **Embedding + Output Head パラメータ** | `67,414,149` (`67.41M`) | `~1,673,216` (`1.67M`) | `1,642,496` (`1.64M`) |
| **Total パラメータ数** | `231,017,605` (`231.02M`) | `~165,276,672` (`165.28M`) | `165,245,952` (`165.25M`) |
| **1Epoch 推定アクティブステップ数** | `~1.72B` (完全固定) | `~1.55B` (可変) | `~6.80B` (可変) |
| **1Epoch 推定総 FLOPs (`6 N_non_emb S`)** | **`~1.69 × 10¹⁸` FLOPs** | **`~1.52 × 10¹⁸` FLOPs** | **`~6.67 × 10¹⁸` FLOPs** |

### 4.3 最適化・バッチ・チェックポイント統制条件
- **オプティマイザ（[`HybridMuonOptimizer`](file:///Users/dayo/My_Projects/MORTM/mortm/train/muon.py)）**:
  - 2D 隠れ層重み行列（`163.58M` params）: `torch.optim.Muon` (`lr = 1.0e-3`, `momentum = 0.95`, `nesterov = True`, `ns_steps = 5`, `weight_decay = 0.1`, `adjust_lr_fn = "match_rms_adamw"`)
  - Embedding / Output Head / 1D RMSNorm: `torch.optim.AdamW` (`lr = 1.0e-3`, `betas = (0.9, 0.95)`, `weight_decay = 0.1` [1Dは `0.0`])
- **学習率スケジューラ（WSD: Warmup–Stable–Decay）**:
  - **1-Epoch 基準ラン（完走済 — [`symic_main_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/symic_main_160m.json), W&B `of4woumg`）**: `"type": "wsd"`, `warmup_ratio = 0.05`（`730` steps 線形ウォームアップ `0 → 1.0e-3`）＋ `stable_ratio = 0.88`（`12,856` steps 定数プラトー `1.0e-3`）＋ 最終 `7%`（`1,024` steps）Cosine クールダウン（`5.0e-5`）にて全 `14,610` steps 完走し、**`val_loss = 1.89362` (`val_ppl = 6.643`, Backbone FLOPs `1.704 × 10¹⁸`)** を記録。ただし128種のMIDIプログラムに勾配が分散する旋律スロットは単一エポック（`~1.74B` AR steps）では Chinchilla 最適トークン予算（`~4.6B+` steps）に対して未収束（`train_loss = 2.03 > val_loss = 1.89`）であることが判明した。
  - **4-Epoch 本学習ラン（[`symic_main_160m_4ep.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/symic_main_160m_4ep.json)）**: 1-Epoch ランとの100%厳密なデータセット統制（全 `801,838` 曲を一切変更せず維持）および初期軌道一致を保ったまま `num_epochs = 4`（全 **`58,440` steps**, `~6.94B` AR steps, `~6.82 × 10¹⁸` Backbone FLOPs）へ拡張する。`warmup_steps = 730`（1-Epoch と完全同一）＋ `stable_ratio = 0.90`（`52,596` steps 定数プラトー `1.0e-3`）＋ 最終 `5,114` steps Cosine クールダウン（`min_lr_ratio = 0.02` → `2.0e-5`）とし、`checkpoint_percents = [25, 50, 75, 100]` により **Epoch 1 / 2 / 3 / 4 各終端のチェックポイントを自動保存** する。
- **有効バッチサイズとステップ数の完全一致**:
  - `big_batch_size = 16,384` セグメント（全 `3,740,025` train windows = `229` BigBatches）、`batch_size = 128`, `accumulation_steps = 2` により、1 optimizer step あたり **`256` 個の16小節セグメント（＝ `4,096 小節 / step`、全 `14,610` steps / epoch、Stage 2 の `64 windows × 64 bars` と音楽量が完全一致）** を3モデル共通で処理する。
  - 全3モデルでセグメントのシャッフル順（`seed = 42`）を固定するため、**各ステップ $t$ でモデルが学習する256個の16小節セグメントは3モデル間で100%同一** となる。
  - `Multitrack-REMI` 等で長系列セグメントを含むマイクロバッチが来る場合は、動的トークン予算マイクロバッチ分割（勾配累積で厳密に256セグメントの平均勾配を計算）により VRAM 溢れを防ぎつつ数学的に同一の更新を行う。
- **2つの公平比較断面（Iso-FLOPs & Iso-Corpus）の同時確保**:
  1. **Iso-Corpus（同一楽曲露出量 = 1 Epoch 100% 完走点）**: 全3モデルが同一の80.2万曲・全16小節セグメントを1周学習し、WSD の最終 `1,024` steps クールダウンを完了した時点での直接比較。
  2. **Iso-FLOPs（同一計算量予算 = $C^* \approx 1.69 \times 10^{18}$ FLOPs 断面）**: WSD スケジューラの特性（Hägele et al., NeurIPS 2024）を活かし、`Multitrack-REMI` が `SyMIc-G4L4` の 1 Epoch 完走 FLOPs $C^*$ に到達するタイミングに合わせて同率（`7%`）のクールダウン分岐を実行し、「同一 GPU 計算予算・同一クールダウン条件下での到達性能」も一切の不公平なく比較する（Dagan et al., ICML 2024 の2軸報告原則に準拠）。

---

## 5. 多面的評価スイート（すべてトークン空間非依存の物理・音楽空間で評価）

語彙数も1音あたりのトークン数も異なる3パラダイム間では、トークン単位の Perplexity 比較は無意味である。したがって本実験の主評価はすべて **デコード後の物理ピアノロール `[N_chunks, 8, 128, 24]` および出力 MIDI 空間** にて実施する。

### 5.1 【評価1】コンテキスト効率・推論システムプロファイル（Systems & Context Scaling）
未見 GMD テスト曲（1〜4トラック、小節内音符密度 `1〜120+ notes/bar`）を対象に、アンサンブル密度に対するスケーリング特性を実測する：
1. **同時発音密度（Notes/Bar・アクティブトラック数）vs 自己回帰ステップ数 & 実測生成速度 (`bars/sec`, `ms/bar`)**
2. **16小節生成時の KV キャッシュメモリ消費量（MB）とレイテンシ推移**（`SyMIc` のフラットな定数直線 vs `Octuple` / `REMI` の密度比例悪化カーブ）
3. **固定ステップ予算（例：`512` / `1024` / `2048` AR steps）を与えた際に到達できる生成小節数分布**

### 5.2 【評価2】複数小節の自由生成評価（Multi-Bar Prompted Continuation）
本実験の中核評価。未見テストセット（`150` 曲以上、3〜4トラック編成を含む）の各16小節セグメントに対し、**先頭4小節をプロンプトとして与え、残り12小節を自由自己回帰生成（4-to-16 Bar Continuation）** させる。さらにスライディングウィンドウ推論により **8小節プロンプト → 24〜32小節の長尺継続生成** も実施し、以下の客観音楽指標を算出する：

1. **プロンプト継承性・音楽分布一致度（Fidelity & Style Consistency）**:
   - **Pitch-Class Histogram JS Divergence**（調性・スケール分布の真値継続区間との一致度）
   - **Inter-Onset Interval (IOI) / Groove Similarity**（リズム・グルーヴパターンの維持度）
   - **Note Duration & Velocity Wasserstein Distance ($W_1$)**（音価・ダイナミクス分布の自然さ）
   - **Seam Continuity（接続境界4小節目→5小節目の滑らかさ）**: [`mortm/eval/metrics.py`](file:///Users/dayo/My_Projects/MORTM/mortm/eval/metrics.py) 準拠の境界音程・密度段差スコア。
2. **マルチトラック垂直アンサンブル整合性（Vertical Ensemble Coherence）**:
   - **Inter-Track Harmonic Consonance / Pitch-Class Overlap**（メロディ・コード・ベースの3旋律トラック間で不協和衝突を起こさず和声進行が噛み合っているか）
   - **Drum–Bass Rhythmic Locking**（Slot 0 キック/スネア強拍と低音トラック Onset の同期率）
3. **長尺構造安定性・破綻耐性（Structural Stability & Failure Rate）**:
   - **Bar-Level Self-Similarity Matrix (SSM) Consistency**（Aメロ反復・4/8小節周期フレーズ構造の再現度）
   - **Track Survival & Density Drift Rate**（生成後半で特定パートが無音消滅したり、逆に音符密度が爆発・ループ固着したりする異常率）
   - **Metric / Bar-Grid Integrity**（`REMI` / `Octuple` で発生する小節長不整合・構文エラー率 vs `SyMIc` の物理クロック保証 `0.00%`）

### 5.3 【評価3】物理次チャンク予測のスケーリング検証（`85M` → `163.6M`）
- [`mortm/eval/mortm5_pianoroll.py`](file:///Users/dayo/My_Projects/MORTM/mortm/eval/mortm5_pianoroll.py) を用い、未見150曲（16小節セグメント群）において Stage 2 の `#8 (85M)` と Stage 3 の `SyMIc-G4L4 (163.6M)` を直接比較し、バックボーン容量倍増による Note F1 / Onset F1 / Velocity MAE の純粋なスケーリング向上幅を定量化する（併せて `REMI` / `Octuple` についても半拍境界プロンプトからの次半拍ロールアウト物理 F1 を計測可能）。

### 5.4 【評価4】再学習不要のメカニズム解析 & ブラインド主観聴取評価
1. **SyMIc-G4L4 レイヤー別累積デコード解析（`L0` → `L0..1` → `L0..2` → `L0..3`）**:
   - 未見 `87,523` チャンクに対し、残差レイヤーを1段ずつ足した際の Onset F1 / Note F1 / Velocity MAE の推移を定量化し、[`render_m4_codebook_video.py`](file:///Users/dayo/My_Projects/MORTM/mortm/video/render_m4_codebook_video.py) の定性可視化と対応づける。
2. **`Soft-Token` 部分空間直交性・線形復元プローブ**:
   - 学習済み `SyMIc-G4L4`（および `Octuple-MMT`）のグループ別 Embedding 部分空間の主角度・コサイン類似度と、加算後の単一ベクトル $\mathbb{R}^{1024}$ から元の各グループ ID を線形分類器で何%復元できるかを検証する。
3. **ブラインド主観聴取評価（Listening Study）**:
   - 同一の4小節プロンプトから `SyMIc-G4L4` / `Octuple-MMT` / `Multitrack-REMI`（すべて `163.6M`）が生成した続き12小節の4パートアンサンブル音源をランダム順提示し、①全体の音楽的自然さ（Overall Musicality）、②パート間のアンサンブル調和（Multi-Track Coherence）、③プロンプトからの展開・構造一貫性（Structural Development）を評価する。

---

## 6. 実装・実行ロードマップ

- [ ] **Step 1：動的長 `RoPE` & 共通 `163.60M` モデル設定の整備**
  - [`mortm/models/modules/attention.py`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/attention.py) の `RoPE` キャッシュを任意系列長へ動的拡張可能にし、`SyMIc-G4L4` (`position_length=518`, 16小節分割)・`Octuple-MMT`・`Multitrack-REMI` の3モデル構成クラスと JSON 設定ファイル（`configs/models/mortm5/symic_main/`）を作成する。
- [ ] **Step 2：`Multitrack-REMI` & `Octuple-MMT` トークナイザ・16小節アラインメント変換器の実装と検証**
  - [`extract_quantized_notes_and_slots`](file:///Users/dayo/My_Projects/MORTM/mortm/utils/convert_pianoroll.py#L931-L973) 共通基盤上で両ベースラインのエンコーダ・デコーダを実装し、単体テスト（往復可逆性テスト・`m4_g4l4` との曲/16小節境界1対1一致テスト）を通過させる。
  - Mac マルチコア処理で GMD 全 `801,838` train 曲 / `4,950` val 曲を一括変換し、memmap バイナリを出力する。
- [ ] **Step 3：256セグメント同期データローダ & 動的マイクロバッチ学習ランナーの整備**
  - 同一シャッフル順・`256 segments/step` 厳密一致・無制限窓長（`flash_attn_varlen` パディング除去）および `Iso-FLOPs` チェックポイント自動保存に対応した学習設定・ランナーを整備し、E2E スモークテストを実施する。
- [ ] **Step 4：再学習不要のメカニズム解析（§5.4-1, §5.4-2）の先行実行**
  - 本学習の並列期間中または前段で、`M4` 累積レイヤー定量表と `Soft-Token` 直交復元プローブを実行・記録する。
- [ ] **Step 5：RTX PRO 6000 Blackwell における `163.60M` 3モデルの1 Epoch 本学習**
  - `SyMIc-G4L4` → `Octuple-MMT` → `Multitrack-REMI`（途中 $C^*$ 断面保存付き）を完走させる。
- [ ] **Step 6：多面的評価スイート（§5.1〜5.3）の一括実行・図表出力・論文執筆**
  - 効率プロファイル、4→16小節＆長尺自由生成ベンチマーク、スケーリング比較表、聴取用サンプル生成を完了し、論文『SyMIc: Symbolic MIDI Codebook』の原稿へ統合する。
