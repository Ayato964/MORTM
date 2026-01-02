<div align="center">
  <img src="asset/title (1).png" width="100%" alt="MORTM Structure"/>

  <h1>MORTM: Metric-Oriented Rhythmic Transformer for Music Generation</h1>

  <p>
    <b>名越 崇晃(Takaaki Nagoshi)</b>
  </p>
  <p>
    <em>Project.MORTM Research Group</em>
  </p>

  <a href="https://github.com/Ayato964/mortm/blob/master/LICENSE">
    <img alt="License" src="https://img.shields.io/badge/License-MIT-blue?style=flat-square">
  </a>
  <img alt="Version" src="https://img.shields.io/badge/Version-4.5-orange?style=flat-square">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch">
  
  <br><br>
  <a href="./README_ja.md"><img src="https://img.shields.io/badge/ドキュメント-日本語-white?style=for-the-badge&logo=ja" alt="Japanese"/></a>
  <a href="./README.md"><img src="https://img.shields.io/badge/Document-English-blue?style=for-the-badge&logo=en" alt="English"/></a>
</div>

---

## 概要 (Abstract)

Transformerアーキテクチャに基づく自己回帰モデルは、記号的音楽生成（Symbolic Music Generation）において顕著な成果を上げています。しかし、標準的なトークン化手法では音楽の時間的階層構造が軽視されがちであり、長期間にわたる構造的一貫性やリズムの整合性を維持することは依然として重要な課題です。

我々は、**MORTM (Metric-Oriented Rhythmic Transformer for Music)** を提案します。これは、小節単位のトークン化戦略を通じて拍節構造（Metric Structure）を明示的にモデル化する新しいフレームワークです。バージョン4.5では、拡張可能な **Sparse Mixture of Experts (MoE)** アーキテクチャと **FlashAttention-2** の統合により、長大なコンテキストにおける効率的な学習を実現しました。さらに、BERTベースの報酬モデル（BERTM）によって定義された様式的目的に生成器を適合させる、PPOを用いた **Reinforcement Learning from Music Feedback (RLMF)** パイプラインを導入しています。

---

## 1. 主な貢献 (Key Contributions)

* **拍節指向トークナイゼーション (Metric-Oriented Tokenization)**:
    音楽イベントをメトリックグリッド内にカプセル化する独自の語彙およびエンコーディングスキームを採用し、小節レベルの構造的整合性を強制します。
* **Sparse Mixture of Experts (MoE)**:
    Top-2 ゲーティングを備えたMoE層の実装により、推論コストを増大させることなくモデル容量（パラメータ数）を大幅にスケールさせることを可能にしました。
* **効率的な長距離モデリング**:
    **FlashAttention-2** と相対位置エンコーディング（**ALiBi/RoPE**）の統合により、メモリ計算量を線形に抑えつつ、長大な音楽シーケンスを処理します。
* **強化学習によるアライメント (RL Alignment)**:
    双方向エンコーダ（BERTM）から導出される報酬を用いて自己回帰ポリシーをファインチューニングする、完全なPPOベースのRLパイプラインを提供します。
* **マルチモーダル拡張**:
    オーディオスペクトログラムモデリング (**V_MORTM**) およびピアノロール視覚処理 (**MORTM Live**) への拡張をサポートしています。

---

## 2. アーキテクチャ (Architecture)

MORTMは、記号的音楽データの特性に最適化されたDecoder-only Transformerを基盤としています。

### 2.1 Sparse Mixture of Experts (MoE)
計算コストを抑制しつつ表現力を向上させるため、特定のブロックにおいて標準的なFFNをMoE層に置換しています。
- **ルーティング機構**: 学習可能なゲートネットワークが、各トークンをTop-$k$（デフォルト $k=2$）のエキスパートにルーティングします。
- **エキスパートの専門化**: これにより、異なるエキスパートが異なる音楽的テクスチャ（例: リズミカルな伴奏 vs 旋律的フレーズ）に特化することが可能になります。

### 2.2 注意機構 (Attention Mechanism)
Attention計算の高速化には **FlashAttention-2** を採用しています。
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
これに **Rotary Positional Embeddings (RoPE)** を組み合わせることで、数千トークンに及ぶシーケンス内の相対的なタイミング依存関係を効果的に捉えます。

### 2.3 報酬モデリング (Reward Modeling)
**BERTM (Bidirectional Encoder Representations for Music)** がCritic（評価者）として機能します。マスク化言語モデリング（MLM）で事前学習され、ジャンルや品質の分類タスクでファインチューニングされたBERTMは、PPO学習フェーズを導くスカラー報酬を提供します。

---

## 3. インストールと前提条件

本研究コードはPyTorchで実装されています。特にFlashAttention-2の性能を最大限に引き出すため、NVIDIA GPU（Ampereアーキテクチャ以降）の使用を推奨します。

```bash
# リポジトリのクローン
git clone [https://github.com/Ayato964/mortm.git](https://github.com/Ayato964/mortm.git)
cd mortm

# コアライブラリのインストール
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
pip install flash-attn --no-build-isolation

# プロジェクト依存関係のインストール
pip install -r requirements.txt
