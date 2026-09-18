import os
import sys

# RTX 5070 Ti のみを使用
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch

from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN, TO_MUSIC
from mortm.utils.convert import MIDIConverter
from mortm.utils.de_convert import ct_token_to_midi
from eval.foundation_generate_test import reset_kv_cache, format_sequence, parse_melody_blocks

MIDI_INPUT_PATH = "data/generate/Piano_Sample.mid"
MODEL_ARGS_PATH = "configs/models/mortm/foundation/A80M_E64.json"
MODEL_WEIGHT    = "out/models/4_5/MORTM.4.5E-A80M-E64.pth"
OUT_DIR         = "out/moe_test"

os.makedirs(OUT_DIR, exist_ok=True)

print("=== 1. トークナイザー & イントロ MIDI の読み込み ===")
tok = Tokenizer(get_token_converter_pro(TO_TOKEN))

# MIDI をトークン化
conv = MIDIConverter(tok, os.path.dirname(MIDI_INPUT_PATH), os.path.basename(MIDI_INPUT_PATH), program_list=['PIANO'])
conv.convert()

if conv.is_error:
    print(f"Error converting MIDI: {conv.error_reason}")
    sys.exit(1)

past_raw_tokens = list(conv.midi2seq.aya_node['PIANO'])
# 末尾の <TE> や終了記号があれば除外
tok.mode(to=TO_MUSIC)
te_id = tok.get("<TE>")
eseq_id = tok.get("<ESEQ>")
past_notes_tokens = [t for t in past_raw_tokens if t not in (te_id, eseq_id)]

print(f"Intro MIDI: {MIDI_INPUT_PATH}")
print(f"Intro tokens count: {len(past_notes_tokens)}")

# キーの判定 (G minor -> k_Gm)
key_name = "Gm"
if conv.key and isinstance(conv.key, dict) and "global_key" in conv.key:
    gk = conv.key["global_key"]
    # 例: "G minor" -> "Gm", "C major" -> "CM"
    sp = gk.split()
    if len(sp) == 2:
        mode = "m" if "minor" in sp[1].lower() else "M"
        key_name = f"{sp[0]}{mode}"
print(f"Detected Key: {key_name}")

# === 2. プロンプト構築 ===
# <EOS> <SYSTEM> <INST_PIANO> <NOTE_DENSE_4> k_Gm <TAG_END>
# <PAST_M> <INST_PIANO> [past_notes] <ESEQ> <TAG_END>
prompt = [
    tok.get("<EOS>"),
    tok.get("<SYSTEM>"),
    tok.get("<INST_PIANO>"),
    tok.get("<NOTE_DENSE_4>"),
    tok.get(f"k_{key_name}"),
    tok.get("<TAG_END>"),
    tok.get("<PAST_M>"),
    tok.get("<INST_PIANO>"),
] + past_notes_tokens + [
    tok.get("<ESEQ>"),
    tok.get("<TAG_END>"),
    tok.get("<CONST_M>"),  # ここから生成を開始させる
]

prompt_arr = np.array(prompt, dtype=int)
print(f"\n=== 2. プロンプト構築完了 ===")
print(f"Total prompt tokens: {len(prompt_arr)}")

# === 3. モデルロード ===
print(f"\n=== 3. モデルロード (RTX 5070 Ti) ===")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
progress = _DefaultLearningProgress()
args = MORTMArgs(MODEL_ARGS_PATH)
model = MORTM(args=args, progress=progress)

sd = torch.load(MODEL_WEIGHT, map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=True)
model.to(device=device, dtype=torch.bfloat16)
model.eval()
print("Model loaded successfully into GPU!")

# === 4. 後続生成 ===
print(f"\n=== 4. 後続生成開始 (temperature=1.0, top_p=0.95) ===")
reset_kv_cache(model)
src = torch.tensor(prompt_arr, device=device, dtype=torch.long).unsqueeze(0)

generated = []
const_m_id  = tok.get("<CONST_M>")
future_m_id = tok.get("<FUTURE_M>")
past_m_id   = tok.get("<PAST_M>")
tag_end_id  = tok.get("<TAG_END>")
sme_id      = tok.get("<SME>")
max_steps   = 1500

with torch.inference_mode():
    padding_mask = (src != model.embedding.padding_idx)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward(src, padding_mask=padding_mask, is_causal=True, is_save_cache=True)
    next_token = model.top_p_sampling(logits[:, -1, :], p=0.95, temperature=1.0)

    for step in range(max_steps):
        t = next_token.item()
        generated.append(t)

        if t == tag_end_id:
            print(f"  [{step}] <TAG_END> 到達 (CONST_M ブロック完了)")
            break
        elif t == te_id:
            print(f"  [{step}] <TE> 到達")
            break

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{max_steps} ... (生成中)")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(next_token, padding_mask=None, is_causal=True, is_save_cache=True)
        next_token = model.top_p_sampling(logits.squeeze(1), p=0.95, temperature=1.0)

gen_tokens = np.array(generated, dtype=int)
print(f"\n生成トークン数: {len(gen_tokens)}")

# === 5. イントロと続きの結合 & MIDI 出力 ===
print(f"\n=== 5. MIDI 再構成 (イントロ + 続き) ===")

# 生成された CONST_M からピアノの音符列を抽出
# CONST_M は <INST_PIANO> [音符列] <ESEQ> と展開される
const_seq = []
in_inst = False
for t in gen_tokens:
    sym = tok.rev_get(int(t))
    if sym == "<INST_PIANO>":
        in_inst = True
        continue
    elif sym in ("<ESEQ>", "<TAG_END>"):
        in_inst = False
        break
    if in_inst:
        const_seq.append(int(t))

if not const_seq:
    # もし <INST_PIANO> なしで直接音符が出た場合
    const_seq = [int(t) for t in gen_tokens if t not in (te_id, eseq_id, tag_end_id)]

print(f"Extracted continuation tokens: {len(const_seq)}")
n_sme_gen = sum(1 for t in const_seq if t == sme_id)
print(f"Continuation measures: {n_sme_gen} 小節")

# 1. イントロ単体 MIDI (確認用)
intro_midi_out = os.path.join(OUT_DIR, "moe_continuation_intro.mid")
intro_seq = [0, tok.get("<INST_PIANO>")] + past_notes_tokens
ct_token_to_midi(tok, torch.tensor(intro_seq), intro_midi_out, tempo=120)
print(f"★ イントロ MIDI: {intro_midi_out}")

# 2. 続き単体 MIDI
continuation_only_out = os.path.join(OUT_DIR, "moe_continuation_generated_only.mid")
if const_seq:
    ct_token_to_midi(tok, torch.tensor([0, tok.get("<INST_PIANO>")] + const_seq), continuation_only_out, tempo=120)
    print(f"★ 続き単体 MIDI: {continuation_only_out}")

# 3. イントロ + 続きの合体 MIDI (完全版)
full_song_out = os.path.join(OUT_DIR, "moe_continuation_full.mid")
full_song_seq = [0, tok.get("<INST_PIANO>")] + past_notes_tokens + const_seq
ct_token_to_midi(tok, torch.tensor(full_song_seq), full_song_out, tempo=120)
print(f"★ 合体版 MIDI (イントロ→続き): {full_song_out} ({os.path.getsize(full_song_out)} bytes)")

print("\n=== 完了 ===")
