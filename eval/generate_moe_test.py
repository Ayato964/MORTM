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
from mortm.utils.de_convert import ct_token_to_midi
from eval.foundation_generate_test import (
    reset_kv_cache, format_sequence, parse_melody_blocks, reconstruct_melody
)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--density", type=int, default=4, help="Note density (1-10)")
parser.add_argument("--key", type=str, default="Fm", help="Music key (e.g. Fm, CM)")
parser.add_argument("--inst", type=str, default="PIANO", help="Instrument (e.g. PIANO, SAX)")
parser.add_argument("--temp", type=float, default=1.0, help="Temperature")
parser.add_argument("--top_p", type=float, default=0.95, help="Top-p")
args_cli = parser.parse_args()

DENSITY_NUM = args_cli.density
KEY_NAME    = args_cli.key
INST_NAME   = args_cli.inst
TEMP        = args_cli.temp
TOP_P       = args_cli.top_p

MODEL_ARGS_PATH = "configs/models/mortm/foundation/A80M_E64.json"
MODEL_WEIGHT    = "out/models/4_5/MORTM.4.5E-A80M-E64.pth"
OUT_DIR         = "out/moe_test"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"=== 1. モデルロード (Density={DENSITY_NUM}, Key={KEY_NAME}, Inst={INST_NAME}) ===")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {torch.cuda.get_device_name(0)}")

tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
tokenizer.mode(to=TO_MUSIC)
progress = _DefaultLearningProgress()
args = MORTMArgs(MODEL_ARGS_PATH)
model = MORTM(args=args, progress=progress)

sd = torch.load(MODEL_WEIGHT, map_location="cpu", weights_only=True)
model.load_state_dict(sd, strict=True)
model.to(device=device, dtype=torch.bfloat16)
model.eval()
print(f"Model loaded successfully from {MODEL_WEIGHT}")

# プロンプト構築: Density, Key, Inst
prompt_tokens = [
    tokenizer.get("<EOS>"),
    tokenizer.get("<SYSTEM>"),
    tokenizer.get(f"<INST_{INST_NAME}>"),
    tokenizer.get(f"<NOTE_DENSE_{DENSITY_NUM}>"),
    tokenizer.get(f"k_{KEY_NAME}"),
    tokenizer.get("<TAG_END>")
]
prompt = np.array(prompt_tokens, dtype=int)
print(f"\n=== 2. プロンプト ===")
print("Prompt tokens:", prompt_tokens)
print("Symbols:", [tokenizer.rev_get(t) for t in prompt_tokens])

# デコード
print(f"\n=== 3. 生成開始 (temperature=1.0, top_p=0.95) ===")
reset_kv_cache(model)
src = torch.tensor(prompt, device=device, dtype=torch.long).unsqueeze(0)

generated = []
future_m_id = tokenizer.get("<FUTURE_M>")
const_m_id  = tokenizer.get("<CONST_M>")
past_m_id   = tokenizer.get("<PAST_M>")
tag_end_id  = tokenizer.get("<TAG_END>")
te_id       = tokenizer.get("<TE>")
sme_id      = tokenizer.get("<SME>")

current_block_marker = None
max_steps = 1500

with torch.inference_mode():
    padding_mask = (src != model.embedding.padding_idx)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model.forward(src, padding_mask=padding_mask, is_causal=True, is_save_cache=True)
    next_token = model.top_p_sampling(logits[:, -1, :], p=0.95, temperature=1.0)

    for step in range(max_steps):
        tok = next_token.item()
        generated.append(tok)

        if tok in (past_m_id, const_m_id, future_m_id):
            current_block_marker = tok
            name = {past_m_id: "PAST_M", const_m_id: "CONST_M", future_m_id: "FUTURE_M"}[tok]
            print(f"  [{step}] <{name}> 開始")
        elif tok == tag_end_id:
            # 音楽ブロックの終了
            if current_block_marker is not None:
                name = {past_m_id: "PAST_M", const_m_id: "CONST_M", future_m_id: "FUTURE_M"}.get(current_block_marker, "BLOCK")
                print(f"  [{step}] <{name}> 完了")
                current_block_marker = None
        elif tok == te_id:
            print(f"  [{step}] <TE> 到達 -> 終了")
            break

        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{max_steps} ... (生成中)")

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(next_token, padding_mask=None, is_causal=True, is_save_cache=True)
        next_token = model.top_p_sampling(logits.squeeze(1), p=TOP_P, temperature=TEMP)

gen_tokens = np.array(generated, dtype=int)
full_seq = np.concatenate([prompt, gen_tokens])
print(f"\n生成トークン総数: {len(gen_tokens)}")

# ブロック解析
print(f"\n=== 4. ブロック解析 ===")
blocks = parse_melody_blocks(tokenizer, full_seq)
for bname, inst_data in blocks.items():
    for inst, tokens in inst_data.items():
        n_sme = int(np.sum(tokens == sme_id))
        print(f"  {bname}/{inst}: {len(tokens)} tokens, {n_sme} 小節")

# MIDI 再構成
print(f"\n=== 5. MIDI 再構成 ===")
seq = reconstruct_melody(tokenizer, full_seq)
out_mid = os.path.join(OUT_DIR, f"moe_{KEY_NAME}_density{DENSITY_NUM}.mid")
if len(seq) > 1:
    tokenizer.mode(to=TO_MUSIC)
    try:
        ct_token_to_midi(tokenizer, torch.tensor(seq), out_mid, tempo=120)
        size = os.path.getsize(out_mid)
        print(f"★ MIDI 保存成功: {out_mid} ({size} bytes)")
    except Exception as e:
        print(f"MIDI 保存エラー: {e}")
else:
    print("再構成可能な旋律がありませんでした。")

# トークンログ保存
log_file = os.path.join(OUT_DIR, f"moe_{KEY_NAME}_density{DENSITY_NUM}_tokens.txt")
tokenizer.mode(to=TO_MUSIC)
with open(log_file, "w", encoding="utf-8") as f:
    f.write(format_sequence(tokenizer, full_seq) + "\n")
print(f"トークンログ保存: {log_file}")
print("\n=== 完了 ===")
