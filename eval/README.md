# MORTM 評価・テスト環境ガイド (`eval/`)

本ディレクトリには、MORTM 基盤モデル（Foundation Model）およびファインチューニングモデルの性能を測定・検証・分析するためのすべてのテストスクリプトが集約されています。

---

## 1. スクリプト一覧と対応タスク

### A. 言語モデル・尤度評価 (NLL)
- **`eval_nll.py`**:
  - 時系列予測テストセット（`TEST-SEQ`）および双方向コンテキスト（`continuation`）における負の対数尤度 (NLL) を測定。
  - 各アーム間の相対差（$\Delta$ NLL）も自動算出。
  - 実行例:
    ```bash
    python eval/eval_nll.py --arms BR chrono --scale 10M --budget 400M --n 1000
    python eval/eval_nll.py --arms BR chrono --scale 80M --budget 3.2B --seeds ""
    ```

- **`eval_boundary.py` / `eval_e2_boundary.py`**:
  - 補完タスク（Infill）における境界部（first1, first3, interior, last1）の pitch NLL および top-5 正解率を測定。
  - 実行例:
    ```bash
    python eval/eval_boundary.py --arms BR chrono perm del --scale 10M --budget 400M
    python eval/eval_e2_boundary.py 80M 500
    ```

---

### B. 音楽生成品質・客観軸評価 (5軸)
- **`eval_genaxes.py`**:
  - 凍結 `TEST-TASK`（infill, continuation, condgen, uncond）の各タスクで実際に自己回帰生成を行い、5つの客観評価軸で群レベルの品質を測定。
    - **Axis 0 (Gate)**: 構文妥当性・文法違反率
    - **Axis 1**: キースケール一致度
    - **Axis 2**: ピッチクラス・音価・単語分布の JS ダイバージェンス
    - **Axis 3**: リズムグルーヴ・フェーズ一致度
    - **Axis 4**: 音域・語彙多様性
  - 実行例:
    ```bash
    python eval/eval_genaxes.py 80M 200
    ```

- **`eval_generation_metrics.py`**:
  - 生成された旋律と参照旋律（ref_const）の文法違反率、音高/音長JS、音符数比、境界密度段差を直接評価。
  - 実行例:
    ```bash
    python eval/eval_generation_metrics.py 80M 200
    ```

---

### C. ゼロショット音楽分析能力
- **`eval_analysis.py` / `eval_analysis_acc.py`**:
  - 事前学習済み基盤モデルが、音楽トークン列からメタ情報（調 Key、音符密度 Density、ジャンル Genre）をゼロショットでどれだけ推論できるかを測定。
  - Top-1 正解率 (ACC), Top-5 正解率, Macro-F1, MIREX スコアを出力。
  - 実行例:
    ```bash
    # 複数アーム・シード集計
    python eval/eval_analysis.py --arms BR chrono --scale 10M --budget 400M
    # 単一モデル直接評価
    python eval/eval_analysis_acc.py 80M
    ```

---

### D. 表現学習・プローブ評価 & ファインチューニング適応
- **`bench_probe.py` / `bench_prep.py`**:
  - Moonbeam系外部ベンチマーク（EMOPIA感情分類、Pianist8作曲家分類、MidiKong奏者分類）に対する評価。
  - 基盤モデルを凍結（Frozen backbone）し、Mean-pooling、Attention-pooling (AttnPool)、または PMAヘッドを学習させて汎化性能を検証。
  - 実行例:
    ```bash
    # データ準備 (キャッシュ作成)
    python eval/bench_prep.py --dataset emopia
    # プローブ学習・評価
    python eval/bench_probe.py --dataset emopia --backbone 80M --pool pma --head mlp
    ```

- **`eval_probe.py` / `train_head.py`**:
  - TEST-TASK に対する分類ヘッドの学習と評価。
  - 実行例:
    ```bash
    python eval/eval_probe.py
    ```

- **`eval_ft_cost.py`**:
  - ファインチューニングのトークン量に対する補完性能の学習曲線・適応コスト（LoRA微調整時の収束速度）を検証。
  - 実行例:
    ```bash
    python eval/eval_ft_cost.py noaug_lr1.0 200
    ```

---

### E. 個別生成テスト & 定性可視化
- **`foundation_generate_test.py`**:
  - `<EOS>` をプロンプトとして完全ゼロショット生成を実行。
  - 生成された小節・ブロック構造を解析し、MIDIファイル（`.mid`）として保存。
  - 実行例:
    ```bash
    python eval/foundation_generate_test.py
    ```

- **`eval_e2_samples.py`**:
  - 補完タスク（Infill）の各モデル生成結果を 1 旋律に繋ぎ、継ぎ目を聴覚確認するための MIDI を一括出力。
  - 実行例:
    ```bash
    python eval/eval_e2_samples.py 6
    ```

- **`foundation_attention_vis.py`**:
  - 生成推論中の全層・全ヘッドの Attention 重みを追跡し、ヒートマップを画像出力。
  - 実行例:
    ```bash
    python eval/foundation_attention_vis.py
    ```

---

## 2. 共通設定とパス解決

- **`paths.py`**:
  - データディレクトリ（`~/data/paper`）、モデル重み出力先（`~/out/models/paper`）、モデル config のパスを集約管理。
  - 環境変数（`MORTM_DATA`, `MORTM_MODELS`, `MORTM_TESTTASK` など）で上書き可能。
- **`common.py`**:
  - トークナイザの初期化、チェックポイント自動探索（`find_ckpt`）、モデルロード（`load_model`）、系列読み込み（`load_seqs`）、平均・標準偏差集計（`mean_sd`）などの共通処理を提供。
