import os
import numpy as np

from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from mortm.utils.de_convert import ct_token_to_midi

"""
How to play?

Step1: run next command in terminal
fluidsynth -a pulseaudio -m alsa_seq -s /usr/share/sounds/sf2/FluidR3_GM.sf2

Step2:
run next command in terminal too
aplaymidi --port 128:0 ./out/midi/sample1.midi
"""
# --- 設定 -----------------------------------------------------------
TEST_MIDI            = "0ed647e138608cb582dd2fadf873a56a.mid"
MIDI_DIR             = "/media/takaaki-nagoshi/C8DCDBF4DCDBDB30/MIDIdatasets/GMD/training/all-instruments-with-drums/1/"
#MIDI_DIR            = "./data/other/"
OUT_DIR              = "./out/"
LOG_FILE             = f"{OUT_DIR}token_log.txt"
DISPLAY_SAMPLES      = 3   # stdout / ログに表示するサンプル数
RECONSTRUCT_SAMPLE_IDX = 1   # MIDI 再構成するサンプル番号 (1 始まり)
BLANK_MEASURES       = 8   # CONST 欠損時に挿入する空小節数
# -------------------------------------------------------------------

_BLOCK_STARTS = {"<EOS>", "<PAST_M>", "<CONST_M>", "<FUTURE_M>"}


# ==================================================================
# トークン列の可視化
# ==================================================================

def format_sequence(tokenizer, arr: np.ndarray) -> str:
    """トークンID列を人間が読める記号列に変換する。"""
    lines = []
    cur = []

    for token_id in arr:
        sym = tokenizer.rev_get(int(token_id))

        if sym in _BLOCK_STARTS:
            if cur:
                lines.append("  " + " ".join(cur))
                cur = []
            lines.append("")
            cur = [sym]

        elif sym.startswith("<INST_"):
            if cur:
                lines.append("  " + " ".join(cur))
                cur = []
            cur = [sym]

        elif sym in ("<TAG_END>", "<ESEQ>"):
            cur.append(sym)
            lines.append("  " + " ".join(cur))
            cur = []

        else:
            cur.append(sym)

    if cur:
        lines.append("  " + " ".join(cur))

    return "\n".join(lines).strip()


# ==================================================================
# MIDI 再構成
# ==================================================================

def parse_melody_blocks(tokenizer, arr: np.ndarray) -> dict:
    """
    サンプル配列を解析し PAST_M / CONST_M / FUTURE_M ブロックの
    楽器ごとのトークン列を返す。
    戻り値: {"PAST_M": {"PIANO": np.ndarray, ...}, "CONST_M": {...}, ...}
    """
    block_start_ids = {
        tokenizer.get("<PAST_M>"): "PAST_M",
        tokenizer.get("<CONST_M>"): "CONST_M",
        tokenizer.get("<FUTURE_M>"): "FUTURE_M",
    }
    tag_end_id = tokenizer.get("<TAG_END>")
    eseq_id    = tokenizer.get("<ESEQ>")

    blocks = {}
    current_block  = None
    current_inst   = None
    current_tokens = []

    for token_id in arr:
        token_id = int(token_id)
        sym = tokenizer.rev_get(token_id)

        if token_id in block_start_ids:
            current_block  = block_start_ids[token_id]
            blocks[current_block] = {}
            current_inst   = None
            current_tokens = []

        elif current_block is not None:
            if token_id == tag_end_id:
                if current_inst is not None and current_tokens:
                    blocks[current_block][current_inst] = np.array(current_tokens, dtype=int)
                current_block  = None
                current_inst   = None
                current_tokens = []

            elif token_id == eseq_id:
                if current_inst is not None and current_tokens:
                    blocks[current_block][current_inst] = np.array(current_tokens, dtype=int)
                current_inst   = None
                current_tokens = []

            elif sym.startswith("<INST_"):
                current_inst   = sym[6:-1]   # "<INST_PIANO>" → "PIANO"
                current_tokens = []

            elif current_inst is not None:
                current_tokens.append(token_id)

    return blocks


def reconstruct_melody(tokenizer, arr: np.ndarray, blank_measures: int = 8) -> np.ndarray:
    """
    PAST → CONST → FUTURE の順で全楽器の旋律を再構成し、
    ct_token_to_midi に渡せる単一シーケンスを返す。

    フォーマット: [dummy(0), <INST_A>, seq_A, <INST_B>, seq_B, ...]
      - 先頭の 0 は ct_token_to_midi の seq[1:] による先頭スキップに対応するダミー。
      - <INST_X> を含めることで ct_token_to_midi が楽器を正しく初期化できる。
      - CONST が欠けている場合は blank_measures 小節分の <SME><BLANK> で補完する。
    """
    sme_id     = tokenizer.get("<SME>")
    blank_id   = tokenizer.get("<BLANK>")
    blank_fill = [sme_id, blank_id] * blank_measures

    blocks = parse_melody_blocks(tokenizer, arr)

    # 全ブロックに登場する楽器名を収集
    all_insts = set()
    for block_data in blocks.values():
        all_insts.update(block_data.keys())

    if not all_insts:
        return np.array([], dtype=int)

    # ダミー先頭トークン (ct_token_to_midi が seq[1:] で捨てる)
    combined = [0]

    for inst in sorted(all_insts):
        # <INST_X> を先頭に付けて ct_token_to_midi に楽器を認識させる
        combined.append(tokenizer.get(f"<INST_{inst}>"))

        if "PAST_M" in blocks and inst in blocks["PAST_M"]:
            combined.extend(blocks["PAST_M"][inst].tolist())

        if "CONST_M" in blocks and inst in blocks["CONST_M"]:
            combined.extend(blocks["CONST_M"][inst].tolist())
        else:
            combined.extend(blank_fill)   # CONST 欠損時の補完

        if "FUTURE_M" in blocks and inst in blocks["FUTURE_M"]:
            combined.extend(blocks["FUTURE_M"][inst].tolist())

    return np.array(combined, dtype=int)


# ==================================================================
# メイン
# ==================================================================

tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))

# Step 1: MIDI → トークン列
converter = MIDIConverter(
    tokenizer, MIDI_DIR, TEST_MIDI,
    program_list=["SAX", "PIANO"],
    use_midi2seq=True,
)
converter.convert()

# Step 2: 事前学習サンプル生成
foundation = FoundationDataMaker(converter, min_measure=1, max_measure=8)
foundation.convert()

ok, msg = foundation.save(OUT_DIR)
print(ok, msg)

if ok:
    tokenizer.save(OUT_DIR + "vocab/")
    tokenizer.mode()   # rev_tokens を構築 (ID → 記号)

    node = np.load(f"{OUT_DIR}{TEST_MIDI}.npz")
    n_samples = len(node) - 1
    print(f"生成サンプル数: {n_samples}")

    # --- トークン列の可視化 ---
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        for i in range(min(n_samples, DISPLAY_SAMPLES)):
            arr = node[f"array{i + 1}"]
            header = f"\n{'=' * 60}\nsample{i + 1}  (len={len(arr)})\n{'=' * 60}"
            body   = format_sequence(tokenizer, arr)
            print(header)
            print(body)
            log.write(header + "\n")
            log.write(body   + "\n")
    print(f"\nログ保存: {LOG_FILE}")

    # --- MIDI 再構成 ---
    if 1 <= RECONSTRUCT_SAMPLE_IDX <= n_samples:
        arr = node[f"array{RECONSTRUCT_SAMPLE_IDX}"]
        seq = reconstruct_melody(tokenizer, arr, blank_measures=BLANK_MEASURES)

        if len(seq) > 1:
            midi_dir = f"{OUT_DIR}midi/"
            os.makedirs(midi_dir, exist_ok=True)
            out_path = f"{midi_dir}sample{RECONSTRUCT_SAMPLE_IDX}.midi"
            ct_token_to_midi(tokenizer, seq, out_path, tempo=120)
            print(f"  MIDI 保存: {out_path}  (len={len(seq)})")
        else:
            print("  再構成可能な旋律ブロックが見つかりませんでした。")
    else:
        print(f"  RECONSTRUCT_SAMPLE_IDX={RECONSTRUCT_SAMPLE_IDX} はサンプル範囲外です。")
