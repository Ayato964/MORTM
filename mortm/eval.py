import torch
import torch.nn.functional as F
import pretty_midi
import numpy as np
import matplotlib.pyplot as plt

def compute_perplexity(logits, targets, ignore_index=0):
    """
    perplexityを計算する
    logits: Tensor of shape [B, T, VocabSize]
    targets: Tensor of shape [B, T] (ground truth token IDs)
    ignore_index: index to ignore in loss (e.g., padding)
    """
    B, T, V = logits.shape
    logits = logits.view(-1, V)
    targets = targets.view(-1)

    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='mean')
    perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()


def plot_generated_pitch_analysis(generated_midi_paths, bins=12):
    """
    自己回帰生成されたMIDIファイルのみを対象に
    - ピッチクラスヒストグラム
    - キー推定（K-S法）
    - スケール逸脱率
    を一つの図にまとめて可視化する関数
    :param generated_midi_paths: list of generated MIDI file paths
    :param bins: ピッチクラス数（デフォルト12）
    """
    # Krumhansl-Schmuckler 用のプロファイル
    KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    SCALE_INTERVALS = {
        'major': [0,2,4,5,7,9,11],
        'minor': [0,2,3,5,7,8,10]
    }
    NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    # ピッチヒストグラム計算
    def midi_histogram(paths):
        all_hist = []
        for path in paths:
            midi = pretty_midi.PrettyMIDI(path)
            pitches = [note.pitch % bins for inst in midi.instruments for note in inst.notes]
            hist = np.bincount(pitches, minlength=bins).astype(np.float32)
            all_hist.append(hist / (hist.sum() + 1e-8))
        return np.mean(all_hist, axis=0)
    hist = midi_histogram(generated_midi_paths)
    # キー推定
    best_corr, best_root, best_mode = -np.inf, 0, 'major'
    norms = hist / (hist.sum() + 1e-8)
    for mode, profile in [('major', KRUMHANSL_MAJOR), ('minor', KRUMHANSL_MINOR)]:
        prof_norm = profile / profile.sum()
        for root in range(bins):
            corr = np.corrcoef(norms, np.roll(prof_norm, root))[0,1]
            if corr > best_corr:
                best_corr, best_root, best_mode = corr, root, mode
    # スケール逸脱率
    intervals = SCALE_INTERVALS[best_mode]
    total_notes = 0
    in_scale = 0
    for path in generated_midi_paths:
        midi = pretty_midi.PrettyMIDI(path)
        for inst in midi.instruments:
            for note in inst.notes:
                p = note.pitch % bins
                total_notes += 1
                if ((p - best_root) % bins) in intervals:
                    in_scale += 1
    scale_consistency = in_scale / total_notes if total_notes > 0 else 0

    # プロット
    labels = ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#']
    order = [9,10,11,0,1,2,3,4,5,6,7,8]
    x = np.arange(bins)
    plt.figure(figsize=(10,5))
    plt.bar(x, hist[order], width=0.6, alpha=0.8)
    plt.xticks(x, labels)
    plt.ylabel('Normalized Frequency')
    plt.title(
        f'Generated Pitch-Class Histogram & Tonal Analysis\n'
        f'Key: {labels[best_root]} {best_mode}, Scale Consistency: {scale_consistency:.2f}'
    )
    plt.tight_layout()
    plt.show()

# 使い方例:
# generated = ["gen1.mid", "gen2.mid", ...]
# reference = ["ref1.mid", "ref2.mid", ...]
# plot_pitch_class_histograms_A_to_G_midi(generated, reference)

def compute_rhythm_density_from_midi(midi_path, bins=64):
    """
    MIDIファイルからリズム密度（ノートオンセット分布）を計算する
    :param midi_path: MIDIファイルのパス
    :param bins: 1小節あたりのビン数（デフォルト64）
    :return: 正規化されたビン数次元の密度ヒストグラム (numpy array)
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    resolution = midi.resolution  # ticks per quarter note
    ticks_per_bar = resolution * 4  # 4/4 拍子を想定

    # ノートオンセット位置をビンに割り当て
    onsets = []
    for inst in midi.instruments:
        for note in inst.notes:
            tick = midi.time_to_tick(note.start)
            tick_in_bar = tick % ticks_per_bar
            bin_idx = int(tick_in_bar * bins / ticks_per_bar)
            onsets.append(bin_idx)
    # ヒストグラム計算と正規化
    counts = np.bincount(onsets, minlength=bins).astype(np.float32)
    return counts / (counts.sum() + 1e-8)

def compute_average_rhythm_density(midi_paths, bins=96):
    """
    複数MIDIファイルから平均リズム密度を計算する
    :param midi_paths: MIDIファイルパスのリスト
    :param bins: ビン数
    :return: 平均化された密度ヒストグラム (numpy array)
    """
    densities = [compute_rhythm_density_from_midi(p, bins) for p in midi_paths]
    return np.mean(densities, axis=0)

def plot_rhythm_pattern_density_midi(generated_midi_paths, reference_midi_paths, bins=96):
    """
    生成と参照のMIDIリストを受け取り、リズム密度分布をプロットする
    :param generated_midi_paths: 生成MIDIファイルパスのリスト
    :param reference_midi_paths: 参照MIDIファイルパスのリスト
    :param bins: ビン数（デフォルト64）
    """
    gen_density = compute_average_rhythm_density(generated_midi_paths, bins)
    ref_density = compute_average_rhythm_density(reference_midi_paths, bins)

    x = np.arange(bins)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, ref_density, label='Reference', linewidth=2)
    ax.plot(x, gen_density, label='Generated', linewidth=2)
    ax.set_xlabel('Tick Position (0–{})'.format(bins-1))
    ax.set_ylabel('Normalized Density')
    ax.set_title('Rhythm Pattern Density')
    ax.set_xticks([0, bins//4, bins//2, 3*bins//4, bins-1])
    ax.set_xticklabels([0, bins//4, bins//2, 3*bins//4, bins-1])
    ax.legend()
    plt.tight_layout()
    plt.show()

# 使用例:
# generated = ["gen1.mid", "gen2.mid", ...]
# reference = ["ref1.mid", "ref2.mid", ...]
# plot_rhythm_pattern_density_midi(generated, reference)
