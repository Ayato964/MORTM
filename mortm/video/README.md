# MidiVision M4 Codebook Layer-wise Decoding Video Generator

`MidiVision` **Model 4 (`Deep-RVQ: G=4, L=4, D=128, K=2048`)** の各 RVQ コードブックレイヤー（Layer 1〜Layer 4）が保持している音楽的情報を、冒頭 8 小節（64 チャンク / 1,536 ステップ）の CNN (`ResNetDecoder`) デコード過程として可視化するアニメーション生成パッケージです。

## アーキテクチャ構成

- `domain.py` — 不変値オブジェクト（`Resolution`, `PitchWindow`, `BarWindow`, `QuantizedNote`, `StageMetrics`, `DecodedFrameState`, `AnimationConfig`）
- `decoder_engine.py` — `LatentDecodeService` 抽象と `TorchMidiVisionDecodeService`（M4 RVQ レイヤー分解および複数フレーム一括 GPU CNN デコード）
- `strategies.py` — `TrajectoryStrategy` と `TrajectoryStrategyRegistry` による潜在空間トラバーサル戦略
  - `noise_to_fine`（既定）: 無音潜在表現 → 未量子化球面ノイズ → Layer 1 → Layer 1+2 → Layer 1+2+3 → Layer 1+2+3+4
  - `origin_to_fine`: 無音潜在表現 → 原点 $z=0$ → Layer 1 → ... → Layer 1..4
  - `fine_to_coarse`: 無音潜在表現 → Layer 4 単体（微細残差）→ Layer 3+4 → Layer 2+3+4 → 全層統合
  - `isolated_layers`: Layer 1 単体 → Layer 2 単体 → Layer 3 単体 → Layer 4 単体 → 全層和
- `renderer.py` — `FrameRenderer` 抽象と `PianoRollDashboardRenderer`（90/10 カラーバジェット・余白主導タイポグラフィ・1920x1080 高視認性ピアノロール描画）
- `exporters.py` — `FrameStreamSink` 抽象と $O(1)$ メモリストリーミング出力（macOS `AVFoundation` ハードウェア H.264 MP4・GIF・各ステージの `.png` / `.mid` 書き出し）
- `pipeline.py` — チャンクバッチ推論とストリーミングエンコードを統合する `CodebookAnimationPipeline`
- `render_m4_codebook_video.py` — CLI エントリポイント

## 使い方

```bash
# デフォルト実行（4トラック GMD サンプル冒頭8小節 -> MP4 + 各レイヤーの静止画/MIDI出力）
.venv/bin/python -m mortm.video.render_m4_codebook_video

# 任意の MIDI ファイルを指定して MP4 + GIF を生成
.venv/bin/python -m mortm.video.render_m4_codebook_video \
    --midi data/generate/Sample2.mid \
    --out out/video/m4_sample2_layers.mp4 \
    --gif out/video/m4_sample2_layers.gif \
    --artifacts-dir out/video/sample2_stages

# 逆順（微細残差 Layer 4 から大局骨格 Layer 1 へ合流）モードで生成
.venv/bin/python -m mortm.video.render_m4_codebook_video \
    --mode fine_to_coarse \
    --out out/video/m4_fine_to_coarse.mp4
```
