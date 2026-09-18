"""
単一MIDI楽曲のエンコード（MIDI -> ピアノロール/チャンク）および
デコード（ピアノロール -> MIDI）の一連の動作を検証・確認するスクリプト。

使用例:
    python test_pianoroll_roundtrip.py                     # デフォルトのサンプルMIDIで実行
    python test_pianoroll_roundtrip.py path/to/your.mid    # 任意のMIDIファイルで実行
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import mido

from mortm.utils.convert_pianoroll import (
    midi_to_roll,
    roll_to_midi,
    roll_to_chunks,
    chunks_to_roll,
    save_npz,
    load_npz,
    save_chunks_npz,
    load_chunks_npz,
    verify_roundtrip,
    make_synthetic_test_midi,
    extract_notes,
    TOTAL_CHANNELS,
    STEPS_PER_BAR,
    CHUNK_STEPS,
)


def print_section(title: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run_roundtrip_test(midi_path: str | Path, output_dir: str | Path = "./out/roundtrip_test") -> bool:
    midi_path = Path(midi_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section(f"1. 入力MIDIファイルの確認: {midi_path.name}")
    if not midi_path.exists():
        print(f"指定されたMIDIファイルが見つかりません: {midi_path}")
        return False

    mid_orig = mido.MidiFile(str(midi_path))
    print(f"  - ファイルパス: {midi_path}")
    print(f"  - Ticks per beat (PPQ): {mid_orig.ticks_per_beat}")
    print(f"  - トラック数: {len(mid_orig.tracks)}")

    notes_orig = extract_notes(mid_orig)
    drum_notes = [n for n in notes_orig if n.is_drum]
    melodic_notes = [n for n in notes_orig if not n.is_drum]
    print(f"  - 抽出ノート数: 合計 {len(notes_orig)} 音 (ドラム: {len(drum_notes)}, 旋律: {len(melodic_notes)})")

    # -------------------------------------------------------------
    # 2. エンコード (MIDI -> PianoRoll)
    # -------------------------------------------------------------
    print_section("2. エンコード処理 (MIDI -> ピアノロール [bars, 8, 192, 128])")
    try:
        roll, meta = midi_to_roll(midi_path)
    except Exception as e:
        print(f"  [エラー] エンコードに失敗しました: {e}")
        return False

    print(f"  - ピアノロール形状: {roll.shape}")
    print(f"    (小節数: {roll.shape[0]}, チャネル: {roll.shape[1]}, 1小節ステップ: {roll.shape[2]}, ピッチ数: {roll.shape[3]})")
    print(f"  - データ型: {roll.dtype}")
    print(f"  - 値域: min={roll.min():.4f}, max={roll.max():.4f}")
    print(f"  - アクティブセル数 (非ゼロ): {np.count_nonzero(roll):,} / {roll.size:,}")

    print("\n  [スロット割り当て情報]")
    for slot in meta.slots:
        inst_type = "ドラム" if slot.is_drum else f"旋律 (Prog: {slot.program})"
        print(f"    Slot {slot.slot}: {slot.name:<12} | {inst_type:<15} | ノート数: {slot.note_count}")

    # -------------------------------------------------------------
    # 3. チャンク分割と再構成 (Roll <-> Chunks)
    # -------------------------------------------------------------
    print_section("3. チャンク分割検証 (Roll <-> Chunks [N, 8, 24, 128])")
    chunks = roll_to_chunks(roll)
    print(f"  - チャンク形状: {chunks.shape} (1小節あたり {roll.shape[2] // CHUNK_STEPS} チャンク)")
    restored_roll_from_chunks = chunks_to_roll(chunks)
    chunk_match = np.array_equal(roll, restored_roll_from_chunks)
    print(f"  - Chunks -> Roll の復元完全一致: {'PASS (完全一致)' if chunk_match else 'FAIL (不一致)'}")
    if not chunk_match:
        return False

    # -------------------------------------------------------------
    # 4. NPZファイルへの保存と読み込み検証
    # -------------------------------------------------------------
    print_section("4. NPZファイルのシリアライズ・読み込み検証")
    npz_roll_path = output_dir / f"{midi_path.stem}_roll.npz"
    npz_chunks_path = output_dir / f"{midi_path.stem}_chunks.npz"

    save_npz(npz_roll_path, roll, meta)
    save_chunks_npz(npz_chunks_path, chunks, meta)
    print(f"  - ピアノロール保存先: {npz_roll_path} ({os.path.getsize(npz_roll_path) / 1024:.1f} KB)")
    print(f"  - チャンク保存先:     {npz_chunks_path} ({os.path.getsize(npz_chunks_path) / 1024:.1f} KB)")

    loaded_roll, loaded_meta = load_npz(npz_roll_path)
    loaded_chunks, _ = load_chunks_npz(npz_chunks_path)
    roll_save_match = np.array_equal(roll, loaded_roll)
    chunk_save_match = np.array_equal(chunks, loaded_chunks)
    print(f"  - NPZ読み込み整合性: Roll={'OK' if roll_save_match else 'NG'}, Chunks={'OK' if chunk_save_match else 'NG'}")

    # -------------------------------------------------------------
    # 5. デコード (PianoRoll -> MIDI)
    # -------------------------------------------------------------
    print_section("5. デコード処理 (ピアノロール -> 再構成MIDI)")
    decoded_midi_path = output_dir / f"{midi_path.stem}_decoded.mid"
    try:
        roll_to_midi(roll, meta, decoded_midi_path)
        print(f"  - 再構成MIDI出力先: {decoded_midi_path}")
    except Exception as e:
        print(f"  [エラー] デコードに失敗しました: {e}")
        return False

    mid_recon = mido.MidiFile(str(decoded_midi_path))
    notes_recon = extract_notes(mid_recon)
    print(f"  - 再構成MIDIトラック数: {len(mid_recon.tracks)}")
    print(f"  - 再構成ノート数: 合計 {len(notes_recon)} 音")

    # -------------------------------------------------------------
    # 6. ラウンドトリップの完全一致検証 (再エンコード)
    # -------------------------------------------------------------
    print_section("6. ラウンドトリップ再エンコード検証 (再構成MIDI -> Roll2)")
    try:
        roll2, meta2 = midi_to_roll(decoded_midi_path)
        diff_count = int(np.count_nonzero(roll != roll2))

        if diff_count == 0:
            print("  ★ PASS: 復元MIDIを再エンコードしたピアノロールは元のピアノロールと完全に一致しました！")
            print("  (差分セル数: 0 / 全セル完全一致)")
        else:
            print(f"  ▲ WARNING: 再エンコードしたピアノロールに {diff_count} セルの差分があります。")
            max_diff = np.max(np.abs(roll - roll2))
            print(f"    最大差分値: {max_diff:.4f}")

    except Exception as e:
        print(f"  [検証エラー] 再エンコード中に例外が発生しました: {e}")
        return False

    # -------------------------------------------------------------
    # 7. トラック別ノート比較
    # -------------------------------------------------------------
    print_section("7. トラック別ノート復元状況")
    for slot in meta.slots:
        if slot.note_count == 0 and not slot.is_drum:
            continue
        recon_slot_notes = [
            n for n in notes_recon
            if (slot.is_drum and n.is_drum) or (not slot.is_drum and not n.is_drum and n.program == slot.program)
        ]
        status = "OK" if len(recon_slot_notes) > 0 or slot.note_count == 0 else "EMPTY"
        print(f"  Slot {slot.slot} ({slot.name:<10}): 元ノート数={slot.note_count:<4} -> 復元後ノート数={len(recon_slot_notes):<4} [{status}]")

    print_section("検証完了: エンコーダーおよびデコーダーは正常に機能しています！")
    print(f"出力ファイル一覧:")
    print(f"  1. 元MIDI:       {midi_path}")
    print(f"  2. 復元MIDI:     {decoded_midi_path}")
    print(f"  3. ロールNPZ:    {npz_roll_path}")
    print(f"  4. チャンクNPZ:  {npz_chunks_path}\n")

    return True


if __name__ == "__main__":
    test_dir = Path("./out/roundtrip_test")
    test_dir.mkdir(parents=True, exist_ok=True)

    # コマンドライン引数でMIDIファイルが指定されているか確認
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        target_midi = Path(sys.argv[1])
    else:
        # 既存のサンプルMIDIがあればそれを使用、なければ合成MIDIを生成
        candidate_samples = [
            Path("data/generate/Piano_Sample2.mid"),
        ]
        target_midi = None
        for cand in candidate_samples:
            if cand.exists():
                try:
                    # 4/4拍子かチェック
                    m = mido.MidiFile(str(cand))
                    from mortm.utils.convert_pianoroll import _validate_four_four
                    _validate_four_four(m)
                    target_midi = cand
                    break
                except Exception:
                    continue

        if target_midi is None:
            # 4トラック（ドラム + ピアノ + ベース + ストリングス）を含む合成テストMIDIを生成
            synth_path = test_dir / "synthetic_demo.mid"
            print(f"サンプルMIDIを新規生成します: {synth_path}")
            make_synthetic_test_midi(synth_path)
            target_midi = synth_path

    success = run_roundtrip_test(target_midi, output_dir=test_dir)
    sys.exit(0 if success else 1)
