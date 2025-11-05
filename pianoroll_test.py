import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mortm.train.train import VisionTrainSet
from mortm.models.modules.config import MORTM_LIVE_Args
from mortm.utils.pianoroll_convert import get_pianoroll, pianoroll_to_midi
from mortm.models.modules.progress import _DefaultLearningProgress

def visualize_pianoroll(pianoroll, title, output_path):
    """
    ピアノロールを画像として保存する。
    Args:
        pianoroll (np.ndarray): ピアノロールデータ
        title (str): 画像のタイトル
        output_path (str): 画像の保存先パス
    """
    plt.figure(figsize=(20, 10))
    # チャンネルが複数ある場合、最初のチャンネル（ピッチ）のみ表示
    if pianoroll.ndim == 3:
        pianoroll_display = pianoroll[:, :, 0]
    else:
        pianoroll_display = pianoroll

    plt.imshow(pianoroll_display.T, aspect='auto', origin='lower', cmap='magma')
    plt.title(title)
    plt.xlabel("Time (ticks)")
    plt.ylabel("Pitch")
    plt.colorbar(label='Velocity')
    plt.savefig(output_path)
    plt.close()

def test_vision_model(model_path, config_path, test_midi_path, output_dir):
    """
    学習済みのVisionモデルをテストし、結果をMIDIファイルとピアノロール画像として保存する。

    Args:
        model_path (str): 学習済みモデルのパス (.pth)
        config_path (str): モデル設定ファイルのパス (.json)
        test_midi_path (str): テストに使用するMIDIファイルのパス
        output_dir (str): 結果を保存するディレクトリ
    """
    midi_output_dir = os.path.join(output_dir, "midi")
    image_output_dir = os.path.join(output_dir, "image")
    os.makedirs(midi_output_dir, exist_ok=True)
    os.makedirs(image_output_dir, exist_ok=True)

    # 1. モデルと設定の読み込み
    args = MORTM_LIVE_Args(config_path)
    progress = _DefaultLearningProgress()
    trainer = VisionTrainSet(args, progress, load_directory=model_path)
    model = trainer.model
    model.eval()

    # 2. テストデータの準備
    original_pianoroll = get_pianoroll(test_midi_path, args.ticks_per_measure, args.inst_list)
    if original_pianoroll.size == 0:
        print(f"'{test_midi_path}' の読み込みに失敗しました。")
        return

    # 元のピアノロールを可視化
    visualize_pianoroll(
        original_pianoroll, 
        f"Original Pianoroll - {os.path.basename(test_midi_path)}", 
        os.path.join(image_output_dir, f"original_{os.path.basename(test_midi_path)}.png")
    )

    # 3. ピアノロールを1024のチャンクに分割して推論を実行
    chunk_size = 1024  # 学習時と同じチャンクサイズ
    total_ticks = original_pianoroll.shape[0]
    reconstructed_chunks = []

    model_device = progress.get_device()

    for i in range(0, total_ticks, chunk_size):
        chunk = original_pianoroll[i:i + chunk_size]
        actual_chunk_size = chunk.shape[0]

        # 最後のチャンクが小さければパディング
        if actual_chunk_size < chunk_size:
            padding_size = chunk_size - actual_chunk_size
            pad = np.zeros((padding_size, chunk.shape[1], chunk.shape[2]), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)

        input_tensor = torch.from_numpy(chunk).unsqueeze(0).to(model_device)

        with torch.no_grad():
            reconstructed_chunk_tensor, _, _ = model(input_tensor)
        
        reconstructed_chunk = reconstructed_chunk_tensor.squeeze(0).cpu().numpy()

        # パディングした分を削除
        if actual_chunk_size < chunk_size:
            reconstructed_chunk = reconstructed_chunk[:actual_chunk_size]
            
        reconstructed_chunks.append(reconstructed_chunk)

    # 4. 結果を結合し、MIDIファイルと画像として保存
    reconstructed_pianoroll = np.concatenate(reconstructed_chunks, axis=0)
    reconstructed_pianoroll = reconstructed_pianoroll[:total_ticks] # 元の長さに戻す

    output_midi_filename = f"reconstructed_{os.path.basename(test_midi_path)}"
    output_midi_path = os.path.join(midi_output_dir, output_midi_filename)
    
    pianoroll_to_midi(
        reconstructed_pianoroll,
        output_midi_path,
        args.inst_list,
        args.ticks_per_measure
    )

    # 復元されたピアノロールを可視化
    visualize_pianoroll(
        reconstructed_pianoroll, 
        f"Reconstructed Pianoroll - {os.path.basename(test_midi_path)}", 
        os.path.join(image_output_dir, f"reconstructed_{os.path.basename(test_midi_path)}.png")
    )

if __name__ == '__main__':
    # --- 設定 ---
    # 学習済みモデルのパス (pianoroll_train.pyのsave_directoryで指定したパス)
    MODEL_PATH = "out/model/mortm_live/MORTM_LIVE.train.0.3.5144.pth" # Vision.pthではなく、バージョン名のディレクトリを指定してください
    
    # モデル設定ファイルのパス (pianoroll_train.pyで使用したもの)
    CONFIG_PATH = "configs/models/live/A.json"
    
    # テスト用MIDIファイルのパス
    TEST_MIDI_PATH = "data/other/A-Beautiful-Friendship.mid" # ★★★ テストしたいMIDIファイルのパスを指定してください ★★★
    
    # 出力先ディレクトリ
    OUTPUT_DIR = "out/test/vision"
    
    # --- 実行 ---
    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイルが見つかりません: {MODEL_PATH}")
        print("MODEL_PATHを、学習済みモデルが保存されている正しいパスに修正してください。")
    elif not os.path.exists(TEST_MIDI_PATH):
        print(f"エラー: テスト用MIDIファイルが見つかりません: {TEST_MIDI_PATH}")
        print("TEST_MIDI_PATHを、存在するMIDIファイルのパスに修正してください。")
    else:
        test_vision_model(MODEL_PATH, CONFIG_PATH, TEST_MIDI_PATH, OUTPUT_DIR)
        print(f"テストが完了しました。出力は {OUTPUT_DIR} を確認してください。")
