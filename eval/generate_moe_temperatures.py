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

MODEL_ARGS_PATH = "configs/models/mortm/foundation/A80M_E64.json"
MODEL_WEIGHT    = "out/models/4_5/MORTM.4.5E-A80M-E64.pth"
OUT_DIR         = "out/moe_test"

os.makedirs(OUT_DIR, exist_ok=True)

print("=== 1. モデルロード ===")
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

# プロンプト: Density 4, Key Fm, PIANO
prompt_tokens = [
    tokenizer.get("<EOS>"),
    tokenizer.get("<SYSTEM>"),
    tokenizer.get("<INST_PIANO>"),
    tokenizer.get("<NOTE_DENSE_4>"),
    tokenizer.get("k_Fm"),
    tokenizer.get("<TAG_END>")
]
prompt = np.array(prompt_tokens, dtype=int)
src_base = torch.tensor(prompt, device=device, dtype=torch.long).unsqueeze(0)

future_m_id = tokenizer.get("<FUTURE_M>")
const_m_id  = tokenizer.get("<CONST_M>")
past_m_id   = tokenizer.get("<PAST_M>")
tag_end_id  = tokenizer.get("<TAG_END>")
te_id       = tokenizer.get("<TE>")
sme_id      = tokenizer.get("<SME>")

temperatures = [0.9, 1.0, 1.1, 1.2]
results = []

for temp in temperatures:
    print(f"\n==========================================")
    print(f"=== 生成開始: Temperature = {temp} (top_p=0.95) ===")
    print(f"==========================================")
    
    reset_kv_cache(model)
    src = src_base.clone()
    generated = []
    current_block_marker = None
    max_steps = 1500

    with torch.inference_mode():
        padding_mask = (src != model.embedding.padding_idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(src, padding_mask=padding_mask, is_causal=True, is_save_cache=True)
        next_token = model.top_p_sampling(logits[:, -1, :], p=0.95, temperature=temp)

        for step in range(max_steps):
            tok = next_token.item()
            generated.append(tok)

            if tok in (past_m_id, const_m_id, future_m_id):
                current_block_marker = tok
            elif tok == tag_end_id:
                current_block_marker = None
            elif tok == te_id:
                print(f"  [{step}] <TE> 到達 -> 終了")
                break

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward(next_token, padding_mask=None, is_causal=True, is_save_cache=True)
            next_token = model.top_p_sampling(logits.squeeze(1), p=0.95, temperature=temp)

    gen_tokens = np.array(generated, dtype=int)
    full_seq = np.concatenate([prompt, gen_tokens])
    print(f"  生成トークン数: {len(gen_tokens)}")

    # ブロック解析
    blocks = parse_melody_blocks(tokenizer, full_seq)
    for bname, inst_data in blocks.items():
        for inst, tokens in inst_data.items():
            n_sme = int(np.sum(tokens == sme_id))
            print(f"  ブロック {bname}/{inst}: {len(tokens)} tokens, {n_sme} 小節")

    # MIDI 保存
    seq = reconstruct_melody(tokenizer, full_seq)
    out_mid = os.path.join(OUT_DIR, f"moe_Fm_d4_t{temp:.1f}.mid")
    if len(seq) > 1:
        tokenizer.mode(to=TO_MUSIC)
        try:
            ct_token_to_midi(tokenizer, torch.tensor(seq), out_mid, tempo=120)
            size = os.path.getsize(out_mid)
            print(f"  ★ 保存完了: {out_mid} ({size} bytes)")
            results.append((temp, out_mid, size, len(gen_tokens)))
        except Exception as e:
            print(f"  MIDI 保存エラー: {e}")
    else:
        print("  再構成可能な旋律がありませんでした。")

    # ログ保存
    log_file = os.path.join(OUT_DIR, f"moe_Fm_d4_t{temp:.1f}_tokens.txt")
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(format_sequence(tokenizer, full_seq) + "\n")

print("\n==========================================")
print("=== 全温度の生成が完了しました！ ===")
for temp, path, size, tokens in results:
    print(f"  Temp {temp:.1f}: {path} ({size} bytes, {tokens} tokens)")
print("==========================================")
