# SyMIc (Symbolic MIDI Codebook) 本実験計画書
### — 16小節固定分割・無制限窓長・163.6M共通バックボーン（＋313.3M上位スケール）による対外3パラダイム比較と4→16小節音楽継続評価 —

## 1. 論文全体構成と本実験（Stage 3）の位置づけ

本研究では、マルチトラック記号音楽（MIDI）を2次元ピアノロール多様体上の深層残差ベクトル量子化（Deep Grouped-RVQ）によって固定レート離散コードへ変換する新トークナイザ **`SyMIc` (Symbolic MIDI Codebook)**、および並列グループ埋め込みを分散保持集約して自己回帰生成する **`Grouped Soft-Token Transformer`** を提案する。

論文全体の実験プログラムは以下の **3段階（3-Stage Experimental Program）** で構成され、本計画書はその最終本丸である **Stage 3（本番スケール対外比較・4→16小節音楽継続評価）** の統制仕様を定めるものである：

1. **Stage 1：Codec 内部設計アブレーション（完了済 — [MIDIVISION_EXPERIMENT_PLAN.md](file:///Users/dayo/My_Projects/MORTM/MIDIVISION_EXPERIMENT_PLAN.md)）**
   - 基準モデル `M1` から `M2 (Compact-Dim)`, `M3 (Fine-Group: G8-L2)`, `M4 (Deep-RVQ: G4-L4)` の4モデルを統一学習・比較。
   - サブ空間次元凝縮（$128 \to 32$ 次元）による「量子化ブラー」の解消と、4段深層残差 `SyMIc-G4L4`（`M4`）による最高再構成精度（**`Note F1 = 93.48%`**, **`Onset F1 = 96.44%`**, **`Velocity MAE = 0.1111`**）を実証。
2. **Stage 2：系列モデリング統制アブレーション @ `85M`（完了済 — [MORTM5_ABLATION_EXPERIMENT_PLAN.md](file:///Users/dayo/My_Projects/MORTM/MORTM5_ABLATION_EXPERIMENT_PLAN.md)）**
   - `84.93M`（Non-Embedding）の小規模バックボーン上で `M3/M4 × Flat/Grouped × Shared/Separated` の OFAT 比較を完走。
   - `Separated ≫ Shared`、および **`Grouped Soft-Token` による `Flat` のパレート支配**（`1/4〜1/8` の計算量・`4〜8倍` の推論速度・`2倍` の文脈長でありながら物理次チャンク予測 F1 でも優位）を実証し、「PPL パラドックス」と「残差キメラ現象」を数理的に解明。
3. **Stage 3：本番スケール対外パラダイム比較 & 4→16小節音楽継続評価 @ `163.6M` / `313.3M`（学習完了・評価フェーズ）**
   - Stage 2 の統制アブレーションで確立した主役構成 **`SyMIc-G4L4` (`Grouped Soft-Token + Separated`)** と既存2大標準パラダイム（`Octuple-MMT`・`Multitrack-REMI`）の **全5モデル**（`163.60M` 3パラダイム 1-Epoch Iso-Corpus ＋ `SyMIc-G4L4 163.60M` 4-Epoch High-Compute Iso-FLOPs ＋ `SyMIc-G4L4 313.30M` [`320M` 級] 4-Epoch 上位スケーリング）の学習をすべて完了した。
   - 本段階では以下の3本柱により、`SyMIc` の本質的優位性を実証する：
     1. **コンテキスト・推論システム効率（§5.1）**
     2. **4→16小節プロンプト音楽継続の国際標準客観評価（§5.2: `FMD [CLaMP 2]` + 曲単位ペア比較 `MusPy` 3指標）**
     3. **ブラインド主観聴取評価（§5.3: 同一音源WAV・5段階MOS）**

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

### 2.2 GigaMIDI 全コーパス（`801,838` train 曲 = `3,740,025` セグメント / `4,950` val 曲 = `22,607` セグメント）における厳密実測トークン規模

全コーパス変換済みの `out/mortm5_tokens/{m4_g4l4,octuple_mmt}` バイナリ（総音符数 `train`: `1,480,353,804` 音 / `val`: `10,341,117` 音、末尾短尺端数セグメントを含む全 `3,740,025` セグメント）から厳密計測した自己回帰ステップ数 $S_{\text{AR}} = L - 1$（`seq[:-1]` 入力長）およびバックボーン計算量 $C = 6 N_{\text{non-emb}} S_{\text{AR}}$（$N_{\text{non-emb}} = 163,603,456$）は以下の通りである：

| 指標（GMD Full `train`: `3,740,025` segs） | **`SyMIc-G4L4` (提案)** | **`Octuple-MMT` (音符集約)** | **`Multitrack-REMI` (Flat)** |
| :--- | :---: | :---: | :---: |
| **1Epoch 総 AR ステップ数 $\sum S_{\text{AR}}$** | **`1,735,919,853` (`1.000×`)** | `1,499,053,929` (`0.864×`) | **`6,561,226,916` (`3.780×`)** |
| **Validation 総 AR ステップ数（`22,607` segs）** | `10,427,187` (`1.000×`) | `10,454,152` (`1.003×`) | `45,382,530` (`4.352×`) |
| **セグメント平均 $S_{\text{AR}}$ / 中央値 `p50`** | `464.15` / `517` | `400.81` / `320` | `1,754.33` / `1,435` |
| **密集分位点 `p90` / `p95` / `p99` / `p99.9`** | `517` / `517` / `517` / `517` | `828` / `1,021` / `1,451` / `2,175` | `3,561` / `4,338` / `6,163` / `9,033` |
| **最大セグメント長 `max`（Zero-Truncation）** | **`517`（完全有界）** | `79,991`（`154.7×` 膨張） | **`322,271`（`623.3×` 膨張）** |
| **1Epoch 総 Backbone FLOPs ($6 N_{\text{non-emb}} S_{\text{AR}}$)** | **`1.70401 × 10¹⁸` FLOPs** | **`1.47150 × 10¹⁸` FLOPs** | **`6.44064 × 10¹⁸` FLOPs** |
| **High-Compute Iso-FLOPs 断面 ($\sim 6.6 \times 10^{18}$ FLOPs)** | **`58,440` steps (`4.00 Ep`, `6.816 × 10¹⁸` FLOPs)** | *(1Ep `1.472 × 10¹⁸` で収束済)* | **`14,610` steps (`1.00 Ep`, `6.441 × 10¹⁸` FLOPs)** |

`Multitrack-REMI` は 1Epoch あたりの自己回帰ステップ数および Backbone FLOPs が `SyMIc-G4L4` の **`3.780倍`**（`6.441 × 10¹⁸` vs `1.704 × 10¹⁸`）に達するため、**`SyMIc-G4L4` は `REMI` がわずか 1Epoch を完走するのとほぼ同一の GPU 計算量予算（比率 `1.058 : 1.000`）で 4Epoch 分のフル学習（`58,440` steps / `6.816 × 10¹⁸` FLOPs）を完了できる**。

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

## 4. 統制されたモデルアーキテクチャ・学習プロトコル・完走済みモデル一覧

### 4.1 窓の最大長「無制限」ルール（Zero-Truncation Guarantee）
- **最大系列長による切り詰め（Truncation）は全モデルで一切行わない。**
  - `Multitrack-REMI` で `8,000` トークンを超える超高密度16小節セグメントであっても、1音たりとも切り捨てず全文脈をそのままモデルに入力する。
  - 位置符号化 `RoPE`（$\theta = 10000.0$）は学習可能パラメータを持たないため、[`attention.py`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/attention.py#L318-L350) の `cos/sin` キャッシュを入力バッチの最大系列長に応じてオンデマンドで自動拡張する仕組みを導入し、**パラメータ数を1個も変えずに任意長系列へ厳密対応**させる。
  - 可変長バッチにおける `<PAD>` トークンは [`unpad_input` および `flash_attn_varlen_qkvpacked_func`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/attention.py#L37-L90) により Attention 前後で物理的に除去し、パディングに起因する計算量・VRAM 浪費および勾配汚染を完全にゼロにする。

### 4.2 モデルバックボーン仕様（`163.60M` 3パラダイム共通 ＋ `313.30M` 上位スケール）

| 項目 | **`SyMIc-G4L4` (提案 `160M`)** | **`Octuple-MMT` (`160M`)** | **`Multitrack-REMI` (`160M`)** | **`SyMIc-G4L4` (上位 `320M` 級)** |
| :--- | :---: | :---: | :---: | :---: |
| **設定JSON** | [`symic_g4l4_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/symic_g4l4_160m.json) | [`octuple_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/octuple_160m.json) | [`remi_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/remi_160m.json) | [`symic_g4l4_320m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/symic_g4l4_320m.json) |
| **Transformer 層数 (`d_layer`)** | `12` | `12` | `12` | `16` |
| **隠れ層次元 (`d_model`)** | `1024` | `1024` | `1024` | `1280` |
| **Attention ヘッド数 (`num_heads`)** | `16` (`head_dim = 64`) | `16` (`head_dim = 64`) | `16` (`head_dim = 64`) | `20` (`head_dim = 64`) |
| **SwiGLU 中間次元 (`dim_feedforward`)** | `3072` (`use_bias = False`) | `3072` (`use_bias = False`) | `3072` (`use_bias = False`) | `3392` (`use_bias = False`) |
| **正規化 / 位置符号化 / Dropout** | `RMSNorm` / 動的長 `RoPE` / `0.1` | `RMSNorm` / 動的長 `RoPE` / `0.1` | `RMSNorm` / 動的長 `RoPE` / `0.1` | `RMSNorm` / 動的長 `RoPE` / `0.0` |
| **Non-Embedding Decoder パラメータ数** | **`163,603,456` (`163.60M`)** | **`163,603,456` (`163.60M`)** | **`163,603,456` (`163.60M`)** | **`313,304,320` (`313.30M`)** |
| **Embedding + Output Head パラメータ** | `67,429,524` (`67.43M`) | `1,673,216` (`1.67M`) | `1,642,496` (`1.64M`) | `84,278,676` (`84.28M`) |
| **Total パラメータ数** | `231,032,980` (`231.03M`) | `165,276,672` (`165.28M`) | `165,245,952` (`165.25M`) | `397,582,996` (`397.58M`) |
| **1Epoch 実測 AR ステップ数 (`train`)** | **`1,735,919,853` (`1.74B`)** | **`1,499,053,929` (`1.50B`)** | **`6,561,226,916` (`6.56B`)** | **`1,735,919,853` (`1.74B`)** |
| **1Epoch 実測総 FLOPs ($6 N_{\text{non-emb}} S_{\text{AR}}$)** | **`1.704 × 10¹⁸` FLOPs** | **`1.472 × 10¹⁸` FLOPs** | **`6.441 × 10¹⁸` FLOPs** | **`3.263 × 10¹⁸` FLOPs** |

### 4.3 最適化・バッチ統制と3つの公平比較断面
- **オプティマイザ（[`HybridMuonOptimizer`](file:///Users/dayo/My_Projects/MORTM/mortm/train/muon.py)）**:
  - 2D 隠れ層重み行列: `torch.optim.Muon` (`lr = 1.0e-3`, `momentum = 0.95`, `nesterov = True`, `ns_steps = 5`, `weight_decay = 0.1`, `adjust_lr_fn = "match_rms_adamw"`)
  - Embedding / Output Head / 1D RMSNorm: `torch.optim.AdamW` (`lr = 1.0e-3`, `betas = (0.9, 0.95)`, `weight_decay = 0.1` [1Dは `0.0`])
- **有効バッチサイズとデータ順の完全一致**:
  - `big_batch_size = 16,384` セグメント（全 `3,740,025` train windows = `229` BigBatches）、1 optimizer step あたり **`256` 個の16小節セグメント（＝ `4,096 小節 / step`、全 `14,610` steps / epoch）** を全モデル共通で処理。
  - セグメントのシャッフル順（`seed = 42`）を固定し、長系列を含むマイクロバッチも動的トークン予算分割で数学的に同一の256セグメント平均勾配を更新する。
- **3つの統制比較断面（Three Controlled Comparison Axes）**:
  1. **比較軸 A — 1-Epoch Iso-Corpus 3パラダイム直接比較**:
     - `SyMIc-G4L4 (160M, 1ep)` vs `Octuple-MMT (160M, 1ep)` vs `Multitrack-REMI (160M, 1ep)`
     - 同一 `163.60M` バックボーン・同一 `801,838` 曲 1周・同一 `14,610` optimizer steps・同一 WSD クールダウン（5% warmup `730` steps ＋ 88% stable `12,856` steps ＋ 7% cosine decay `1,024` steps）下での公平比較。なお、`SyMIc` は語彙・サブ空間数が大きく `1.74B` AR steps（1 Epoch）では Chinchilla 最適トークン予算に対して未収束（`train/loss = 2.03 > val/loss = 1.89`）となるため、下記の比較軸 B・C を併設する。
  2. **比較軸 B — High-Compute Iso-FLOPs 直接比較（$\sim 6.6 \times 10^{18}$ FLOPs）**:
     - `SyMIc-G4L4 (160M, 4ep: 6.816 × 10¹⁸ FLOPs)` vs `Multitrack-REMI (160M, 1ep: 6.441 × 10¹⁸ FLOPs)`
     - ほぼ同一の GPU 計算量予算（比率 `1.058 : 1.000`）のもとで、`SyMIc` が4倍の楽曲反復学習（`6.94B` AR steps / `58,440` steps）により 1-Epoch 時の未収束を解消して到達する性能優位を実証する。
  3. **比較軸 C — `SyMIc` 容量・データスケーリング検証**:
     - `SyMIc 160M (1ep)` → `SyMIc 160M (4ep)` → **`SyMIc 320M (4ep nodrop)`**（Non-Embedding `313.30M` / Total `397.58M`）
     - バックボーン容量倍増と `4Epoch` 学習による検証損失（`val/ppl: 6.643 → 5.299 → 4.734`）および実生成品質の上限スケーリングを検証する。

### 4.4 完走済み全5モデルの学習結果サマリ（Completed Stage 3 Checkpoints & W&B Logs）

Stage 3 で使用する全5モデルの学習はすべて完走し、ローカルチェックポイントおよび W&B 計測ログが確定している：

| モデル | モデル / 学習設定JSON | Non-Emb パラメータ | 完了 Steps (Epochs) | 総 AR ステップ数 | 総 Backbone FLOPs | W&B Run ID | 最終 `train/loss` / `val/loss` (`val/ppl`) | 保存チェックポイント (`.pth`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`SyMIc-G4L4 (160M, 1ep)`** | [`symic_g4l4_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/symic_g4l4_160m.json)<br>[`symic_main_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/symic_main_160m.json) | `163.60M` | `14,610` (`1 Ep`) | `1.74B` | `1.704 × 10¹⁸` | `of4woumg` | `2.0308` / **`1.8936`** (`ppl 6.643`) | [`MORTM5.SyMIc_G4L4_160M_16bar_muon_wsd_latest.pth`](file:///Users/dayo/My_Projects/MORTM/out/symic_main/symic_g4l4_160m_wsd/MORTM5.SyMIc_G4L4_160M_16bar_muon_wsd_latest.pth) |
| **`Octuple-MMT (160M, 1ep)`** | [`octuple_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/octuple_160m.json)<br>[`octuple_main_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/octuple_main_160m.json) | `163.60M` | `14,610` (`1 Ep`) | `1.50B` | `1.472 × 10¹⁸` | `0r4g9moh` | `0.4780` / **`0.4306`** (`ppl 1.538`) | [`MORTM5.Octuple_MMT_160M_16bar_muon_wsd_latest.pth`](file:///Users/dayo/My_Projects/MORTM/out/symic_main/octuple_160m/MORTM5.Octuple_MMT_160M_16bar_muon_wsd_latest.pth) |
| **`Multitrack-REMI (160M, 1ep)`** | [`remi_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/remi_160m.json)<br>[`remi_main_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/remi_main_160m.json) | `163.60M` | `14,610` (`1 Ep`) | `6.56B` | `6.441 × 10¹⁸` | `zuh1cwr1` | `0.4910` / **`0.4398`** (`ppl 1.552`) | [`MORTM5.Multitrack_REMI_160M_16bar_muon_wsd_latest.pth`](file:///Users/dayo/My_Projects/MORTM/out/symic_main/remi_160m/MORTM5.Multitrack_REMI_160M_16bar_muon_wsd_latest.pth) |
| **`SyMIc-G4L4 (160M, 4ep)`** | [`symic_g4l4_160m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/symic_g4l4_160m.json)<br>[`symic_main_160m_4ep.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/symic_main_160m_4ep.json) | `163.60M` | `58,440` (`4 Ep`) | `6.94B` | `6.816 × 10¹⁸` | `kffwyk7y` | `1.6998` / **`1.6674`** (`ppl 5.299`) | [`MORTM5.SyMIc_G4L4_160M_16bar_4ep_muon_wsd_latest.pth`](file:///Users/dayo/My_Projects/MORTM/out/symic_main/symic_4ep/MORTM5.SyMIc_G4L4_160M_16bar_4ep_muon_wsd_latest.pth) |
| **`SyMIc-G4L4 (320M, 4ep nodrop)`** | [`symic_g4l4_320m.json`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/symic_g4l4_320m.json)<br>[`symic_main_320m_2ep_from_pct50.json`](file:///Users/dayo/My_Projects/MORTM/configs/train/mortm5/symic_main_320m_2ep_from_pct50.json) | `313.30M` | `58,440` (`4 Ep`) | `6.94B` | `1.305 × 10¹⁹` | `1sn8ycqs` → `oc7pfcrl` | `1.4712` / **`1.5548`** (`ppl 4.734`) | [`MORTM5.SyMIc_G4L4_320M_16bar_4ep_muon_wsd_nodrop_resume_latest.pth`](file:///Users/dayo/My_Projects/MORTM/out/mortm5_ablation/symic_g4l4_320m_nodrop_4ep/MORTM5.SyMIc_G4L4_320M_16bar_4ep_muon_wsd_nodrop_resume_latest.pth) |

> [!NOTE]
> **`val/ppl` の解釈範囲について**:
> `SyMIc-G4L4` の3モデル間ではトークナイザと語彙（`32,901` 語彙 / 4並列グループ）が完全同一であるため、`val/ppl` の改善（**`6.643` [`160M 1ep`] → `5.299` [`160M 4ep`] → `4.734` [`320M 4ep`]**）を直接スケーリング指標として比較できる。一方、`SyMIc`・`Octuple-MMT`・`Multitrack-REMI` のパラダイム間では語彙数も1音あたりのトークン数も異なるため、Cross-Entropy / PPL を直接比較せず、すべて **§5 で定める物理 MIDI 空間および計算効率プロファイル** 上で評価する。

---

## 5. 最終確定・3本柱評価スイート（すべてトークン空間非依存の物理・音楽空間で評価）

語彙数も1音あたりのステップ数も異なる3パラダイムを完全に公平な土俵で比較するため、すべての生成結果・推論プロファイルは **デコード後の標準マルチトラック MIDI（`.mid`）空間および実測システムコスト** で評価する。なお、Stage 2 で役割を完了した 0.5拍次チャンク教師強制 F1・先行研究非標準の独自ヒューリスティック指標（ドラム×ベース同期率・独自不協和音率・SSM等）・ならびに `M4` 内部解析 2種（累積デコード表 / `Soft-Token` プローブ）は計画を研ぎ澄ますため廃止する。

### 5.1 【評価1】コンテキスト効率・推論システムプロファイル（Systems & Context Efficiency Scaling）
- **対象モデル**: `163.60M` 共通バックボーンの3パラダイム（`SyMIc-G4L4`, `Octuple-MMT`, `Multitrack-REMI`）。
- **評価対象データ**: 未見 GMD テスト曲（`1〜4` トラック、小節内音符密度 `Notes/Bar` の疎領域〜高密度トゥッティ領域までを網羅）。
- **計測項目**:
  1. **同時発音密度（`Notes/Bar`）vs 16小節の自己回帰ステップ数 $S$ および実測生成レイテンシ（`ms/bar`, `bars/sec`）**
  2. **16小節生成時の KV キャッシュメモリ消費量（MB）推移**（音数・編成に不変な `SyMIc` の $O(1)$ 完全定数水平線 vs `Octuple` / `REMI` の密度比例膨張カーブ）
- **実行環境要件**: 単一の共通プロファイルスクリプトとして実装し、**Apple Silicon (`MPS`) 上での即時検証** と **NVIDIA GPU (`CUDA`) 上での論文最終数値計測** の双方に1コマンドで対応させる。

### 5.2 【評価2】4→16小節プロンプト音楽継続の客観評価（4-to-16 Bar Prompted Continuation）

#### A. タスク統制（ネイティブ16小節窓の1本勝負）
- [`data/gmd/eval.json`](file:///Users/dayo/My_Projects/MORTM/data/gmd/eval.json) の未見テスト曲から抽出した **3〜4トラック・アンサンブル 16小節セグメント群（`150〜200` 曲）** を使用し、**先頭4小節（`32` チャンク）をプロンプトとして与え、続く12小節（`96` チャンク、合計16小節）を自由自己回帰生成** させる。
- モデルの学習コンテキスト窓（16小節）を超える長尺スライディングウィンドウ評価（旧8→24/32小節）は交絡排除のため実施しない。

#### B. 完全均一サンプリング規約（Zero-Favoritism Sampling Protocol）
- 全5モデル・全トークン・全出力ヘッド共通で以下の同一条件に固定し、特定パラダイムに有利な特別チューニングを一切行わない：
  - **Temperature**: **`temperature = 1.05`（全モデル・全トークン・全出力ヘッドで完全均一。`fine_temperature = 1.05` とし、残差層や属性別の温度低下を行わない）**
  - **Nucleus Sampling**: **`top_p = 0.95`**, **`top_k = 0`（無制限）**
  - **反復抑制（Repetition Penalty）**: **`repetition_penalty = 1.0`（完全OFF）**
- **構造妥当性ガード**: `Octuple-MMT` および `Multitrack-REMI` には文法状態遷移マスク（FSM）・同一時刻内の正準順序制約・アクティブ楽器スロットマスクを適用し、ベースラインが構文崩壊で自滅しない最強状態で公平に比較する。
- 全モデルの出力を同一量子化グリッドの標準マルチトラック MIDI ファイル（`.mid`）として書き出した上で、共通の評価プログラムへ入力する。

#### C. 国際会議標準に基づく4大客観評価指標
1. **`FMD`（Fréchet Music Distance via `CLaMP 2`）↓**（Retkowski et al., *AAAI 2025 Workshop on AI for Music*, `arXiv:2412.07948`; エンコーダ `CLaMP 2`: Wu et al., *NAACL 2025 Findings*）
   - 公式パッケージ `frechet-music-distance`（`--model clamp2`）を用い、実人間による16小節参照MIDI群と生成16小節MIDI群（プロンプト4小節＋生成12小節）の `CLaMP 2` 深層音楽特徴分布間のフレシェ距離を算出する。プロンプト4小節からの調性・音価・4楽器アンサンブル構造の継承と人間らしさを単一総合スコアで評価する。
2. **1曲ごとの正解継続区間とのペア誤差で測る `MusPy` 3大指標（Dong et al., *ISMIR 2020*; *Multitrack Music Transformer*, *ICASSP 2023* 準拠）**
   - 各テスト曲 $i$ の継続12小節（5〜16小節目）に対し、実人間の正解継続区間 $\text{GT}_i$ とモデル生成継続区間 $\text{Gen}_i$ それぞれの指標値 $M(\cdot)$ を算出し、**1曲単位のペア絶対誤差 $\frac{1}{N}\sum_{i=1}^{N} |M(\text{Gen}_i) - M(\text{GT}_i)|$ ↓** を主評価値とする（併せてコーパス平均値および `Ground Truth` 基準値を横並びで表記する）：
     - **Pitch Class Entropy ($H_{PC}$)**: 12半音クラス分布のシャノン・エントロピー $-\sum_{c=0}^{11} p_c \log_2 p_c$（ドラム除外）。ループ崩壊（過小）や無秩序な乱打（過大）を検出し、原曲固有の和声的多様性との一致度を測る。
     - **Scale Consistency ($SC$)**: 全24種の長調・短調に対する最大スケール内音符率 $\max_{\text{root, mode}} \text{Pitch-in-Scale Rate}$（ドラム除外）。原曲固有の調性・スケール収まり具合との一致度を測る。
     - **Groove Consistency ($GC$)**: 隣接小節間の二値発音オンセットベクトル $G_b, G_{b+1}$ の正規化ハミング類似度平均 $1 - \frac{1}{T-1}\sum_{b=1}^{T-1} d(G_b, G_{b+1})$。原曲固有の小節間リズム・グルーヴ安定度との一致度を測る。
   - ※ `Empty-Bar Rate`（無音小節率）は音楽的に自然なブレイク・休符とパート消滅を区別できないため不採用とする。

### 5.3 【評価3】ブラインド主観聴取評価（Blind Human Listening Study — 5-Point MOS）
- *MMT (ICASSP 2023)*・*FIGARO (ICLR 2023)*・*Anticipatory Music Transformer (TMLR 2024)* 等の国際会議標準プロトコルに準拠し、最終ステップとして実施する。
- 未見テストプロンプト群（10〜15曲）に対する各モデルの `4→16小節` 生成MIDI（および `Ground Truth`）を、完全に同一の General MIDI SoundFont 音源で WAV 音声へ統一レンダリングし、音色差の交絡を排除する。
- モデル名を伏せたランダム順ブラインド提示により、以下3観点の **5段階 MOS（Mean Opinion Score: 1〜5点）** を評価する：
  1. **全体の音楽的な自然さ・心地よさ（Overall Musicality）**
  2. **4楽器アンサンブルの調和・噛み合い（Multi-Track Ensemble Coherence）**
  3. **プロンプト4小節からの繋がり・展開の一貫性（Continuation & Structural Coherence）**

---

## 6. 実装・実行ロードマップ

- [x] **Step 1：動的長 `RoPE` & 共通 `163.60M` / 上位 `313.30M` モデル設定の整備（完了）**
  - [`attention.py`](file:///Users/dayo/My_Projects/MORTM/mortm/models/modules/attention.py) の `RoPE` キャッシュを任意系列長へ動的拡張可能にし、`SyMIc-G4L4` (`position_length=518`, 16小節分割)・`Octuple-MMT`・`Multitrack-REMI` のモデル構成クラスと JSON 設定ファイル（[`configs/models/mortm5/symic_main/`](file:///Users/dayo/My_Projects/MORTM/configs/models/mortm5/symic_main/)）を作成済み。
- [x] **Step 2：`Multitrack-REMI` & `Octuple-MMT` トークナイザ・16小節アラインメント変換器の実装と検証（完了）**
  - [`extract_quantized_notes_and_slots`](file:///Users/dayo/My_Projects/MORTM/mortm/utils/convert_pianoroll.py#L931-L973) 共通基盤上で両ベースラインのエンコーダ・デコーダを実装し、単体テスト（往復可逆性・`m4_g4l4` との曲/16小節境界1対1一致・実GMD `val` 22,607セグメント整合テスト）を全件通過済み。
  - GMD 全 `801,838` train 曲 / `4,950` val 曲を一括変換して `out/mortm5_tokens/octuple_mmt` memmap バイナリを構築し、さらに `Multitrack-REMI` のベクトル化 On-the-Fly 変換およびオフライン高速変換を実装済み。
- [x] **Step 3：256セグメント同期データローダ & 動的マイクロバッチ学習ランナーの整備（完了）**
  - 同一シャッフル順・`256 segments/step` 厳密一致・無制限窓長（`flash_attn_varlen` パディング除去 + `grad_checkpoint_tokens = 16384` 自動 Activation Checkpointing）に対応した学習設定・ランナーを整備し、自動検証スイートを通過済み。
- [x] **Step 4：Stage 3 全5モデルの学習完走 & Studio インタラクティブ検証環境の統合（完了）**
  - `SyMIc 160M (1ep)`・`Octuple-MMT 160M (1ep)`・`Multitrack-REMI 160M (1ep)`・`SyMIc 160M (4ep)`・`SyMIc 320M (4ep nodrop)` 全5チェックポイントの学習完走、および [`generation_engine.py`](file:///Users/dayo/My_Projects/MORTM/demo/backend/generation_engine.py) への全パラダイム実機ストリーミング統合を完了。
- [ ] **Step 5：【評価1】コンテキスト効率・推論システムプロファイル（§5.1）の実装と計測**
  - `Notes/Bar` 密度に対する16小節生成ステップ数・実測レイテンシ（`ms/bar`）・KV キャッシュメモリ消費量の統一計測スクリプトを実装し、Mac (`MPS`) で即時検証（必要に応じて GPU `CUDA` で論文最終数値を計測）する。
- [ ] **Step 6：【評価2】`4→16小節` プロンプト音楽継続ベンチマーク（§5.2）の実装・一括実行**
  - 完全均一サンプリング（`temperature = 1.05`, `top_p = 0.95`, `repetition_penalty = 1.0`）による全5モデルの未見アンサンブル継続生成・MIDI 書き出しパイプライン、および `FMD (CLaMP 2)` ＋ 曲単位ペア `MusPy` 3指標（`PCE`, `SC`, `GC`）自動集計器を実装・実行する。
- [ ] **Step 7：【評価3】ブラインド主観聴取評価キット生成・MOS 実験・論文図表統合（§5.3）**
  - 同一 SoundFont による WAV 一括レンダリングとブラインド MOS 採点画面・集計スクリプトを整備し、聴取評価結果を統合して論文『SyMIc: Symbolic MIDI Codebook』を完成させる。
