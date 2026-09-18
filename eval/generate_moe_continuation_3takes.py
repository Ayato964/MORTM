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
from eval.foundation_generate_test import reset_kv_cache

MIDI_INPUT_PATH = "data/generate/Piano_Sample.mid"
MODEL_ARGS_PATH = "configs/models/mortm/foundation/A80M_E64.json"
MODEL_WEIGHT    = "out/models/4_5/MORTM.4.5E-A80M-E64.pth"
OUT_DIR         = "out/moe_test"

os.makedirs(OUT_DIR, exist_ok=True)

print("=== 1. トークナイザー & イントロ MIDI の読み込み ===")
tok = Tokenizer(get_token_converter_pro(TO_TOKEN))
conv = MIDIConverter(tok, os.path.dirname(MIDI_INPUT_PATH), os.path.basename(MIDI_INPUT_PATH), program_list=['PIANO'])
conv.convert()

past_raw_tokens = list(conv.midi2seq.aya_node['PIANO'])
tok.mode(to=TO_MUSIC)
te_id = tok.get("<TE>")
eseq_id = tok.get("<ESEQ>")
tag_end_id = tok.get("<TAG_END>")
sme_id = tok.get("<SME>")
past_notes_tokens = [t for t in past_raw_tokens if t not in (te_id, eseq_id)]

key_name = "Gm"
if conv.key and isinstance(conv.key, dict) and "global_key" in conv.key:
    gk = conv.key["global_key"]
    sp = gk.split()
    if len(sp) == 2:
        mode = "m" if "minor" in sp[1].lower() else "M"
        key_name = f"{sp[0]}{mode}"

# プロンプト
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
    tok.get("<CONST_M>"),
]
prompt_arr = np.array(prompt, dtype=int)
src_base = torch.tensor(prompt_arr, device="cuda:0", dtype=torch.long).unsqueeze(0)

# モデルロード
print("=== 2. モデルロード (RTX 5070 Ti) ===")
progress = _DefaultLearningProgress()
args = MORTMArgs(MODEL_ARGS_PATH)
model = MORTM(args=args, progress=progress)
sd = torch.load(MODEL_WEIGHT, map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=True)
model.to(device="cuda:0", dtype=torch.bfloat16)
model.eval()
print("Model loaded successfully!")

takes = [
    ("Take 1 (端正・安定)", 0.95, 42),
    ("Take 2 (標準・バランス)", 1.00, 100),
    ("Take 3 (表情豊か)", 1.05, 2026),
]

output_files = []

for idx, (label, temp, seed) in enumerate(takes, 1):
    print(f"\n==========================================")
    print(f"=== {label}: Temp={temp}, Seed={seed} ===")
    print(f"==========================================")
    
    torch.manual_seed(seed)
    reset_kv_cache(model)
    src = src_base.clone()
    generated = []
    max_steps = 1500

    with torch.inference_mode():
        padding_mask = (src != model.embedding.padding_idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(src, padding_mask=padding_mask, is_causal=True, is_save_cache=True)
        next_token = model.top_p_sampling(logits[:, -1, :], p=0.95, temperature=temp)

        for step in range(max_steps):
            t = next_token.item()
            generated.append(t)

            if t == tag_end_id:
                print(f"  [{step}] <TAG_END> 到達")
                break
            elif t == te_id:
                print(f"  [{step}] <TE> 到達")
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward(next_token, padding_mask=None, is_causal=True, is_save_cache=True)
            next_token = model.top_p_sampling(logits.squeeze(1), p=0.95, temperature=temp)

    # 音符抽出
    const_seq = []
    in_inst = False
    for t in generated:
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
        const_seq = [int(t) for t in generated if t not in (te_id, eseq_id, tag_end_id)]

    n_sme = sum(1 for t in const_seq if t == sme_id)
    print(f"  生成トークン数: {len(const_seq)}, 小節数: {n_sme} 小節")

    # 合体 MIDI 出力
    out_mid = os.path.join(OUT_DIR, f"moe_continuation_take{idx}.mid")
    full_seq = [0, tok.get("<INST_PIANO>")] + past_notes_tokens + const_seq
    ct_token_to_midi(tok, torch.tensor(full_seq), out_mid, tempo=120)
    size = os.path.getsize(out_mid)
    print(f"  ★ 保存完了: {out_mid} ({size} bytes)")
    output_files.append((label, out_mid, size, n_sme))

print("\n==========================================")
print("=== 3 パターン全ての生成が完了しました！ ===")
for label, path, size, measures in output_files:
    print(f"  {label}: {path} ({size} bytes, 展開部 {measures} 小節)")
print("==========================================")
