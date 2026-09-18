import os
import numpy as np
import torch
from typing import Dict, List, Tuple

from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN, TO_MUSIC
from mortm.utils.de_convert import ct_token_to_midi

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
MODEL_ARGS_PATH = "configs/models/mortm/foundation/80M.json"
MODEL_WEIGHT    = "out/models/mortm/4_5/MORTM.4.5D-80M_1.1424479484558105.pth"
TEMPERATURE     = 1.15
TOP_P           = 0.95
MAX_GEN_STEPS   = 4000
OUT_DIR         = "out/foundation_test/"


# ---------------------------------------------------------------------------
# KV キャッシュリセット
# ---------------------------------------------------------------------------
def reset_kv_cache(model: MORTM):
    for module in model.modules():
        if hasattr(module, "kv_cache") and hasattr(module, "cache_seqlens"):
            module.kv_cache = None
            module.cache_seqlens = None


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------
def foundation_generate(model: MORTM, tokenizer: Tokenizer,
                         prompt: np.ndarray,
                         temperature: float = 1.0, p: float = 0.95,
                         max_steps: int = 4000) -> np.ndarray:
    """
    <EOS> のみをプロンプトとして完全ランダム生成。
    FUTURE_M ブロックが TAG_END で閉じられたとき停止。
    """
    device = model.progress.get_device()
    reset_kv_cache(model)

    future_m_id = tokenizer.get("<FUTURE_M>")
    const_m_id  = tokenizer.get("<CONST_M>")
    past_m_id   = tokenizer.get("<PAST_M>")
    tag_end_id  = tokenizer.get("<TAG_END>")
    te_id       = tokenizer.get("<TE>")

    src = torch.tensor(prompt, device=device, dtype=torch.long).unsqueeze(0)

    model.eval()
    generated = []
    current_block_marker = None

    with torch.inference_mode():
        padding_mask = (src != model.embedding.padding_idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(src, padding_mask=padding_mask,
                                   is_causal=True, is_save_cache=True)
        next_token = model.top_p_sampling(logits[:, -1, :], p=p, temperature=temperature)

        print("--- Decoding ---")
        for step in range(max_steps):
            tok = next_token.item()
            generated.append(tok)

            if tok in (past_m_id, const_m_id, future_m_id):
                current_block_marker = tok
                name = {past_m_id: "PAST_M", const_m_id: "CONST_M", future_m_id: "FUTURE_M"}[tok]
                print(f"\n  [{step}] <{name}> 開始")
            elif tok == tag_end_id:
                if current_block_marker == future_m_id:
                    print(f"\n  [{step}] <FUTURE_M> 完了 → 停止")
                    break
                current_block_marker = None
            elif tok == te_id:
                print(f"\n  [{step}] <TE> → 停止")
                break

            if step % 100 == 0 and step > 0:
                print(f"\r  Step {step}/{max_steps}", end="", flush=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward(next_token, padding_mask=None,
                                       is_causal=True, is_save_cache=True)
            next_token = model.top_p_sampling(logits.squeeze(1), p=p, temperature=temperature)

        else:
            print(f"\n  最大ステップ {max_steps} 到達")

    return np.array(generated, dtype=int)


# ---------------------------------------------------------------------------
# トークン列の可視化
# ---------------------------------------------------------------------------
_BLOCK_STARTS = {"<EOS>", "<PAST_M>", "<CONST_M>", "<FUTURE_M>", "<SYSTEM>"}

def format_sequence(tokenizer: Tokenizer, arr: np.ndarray) -> str:
    lines, cur = [], []
    for tid in arr:
        sym = tokenizer.rev_get(int(tid))
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


# ---------------------------------------------------------------------------
# ブロック解析・MIDI 再構成
# ---------------------------------------------------------------------------
def parse_melody_blocks(tokenizer: Tokenizer,
                         arr: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    block_ids = {
        tokenizer.get("<PAST_M>"): "PAST_M",
        tokenizer.get("<CONST_M>"): "CONST_M",
        tokenizer.get("<FUTURE_M>"): "FUTURE_M",
    }
    tag_end_id = tokenizer.get("<TAG_END>")
    eseq_id    = tokenizer.get("<ESEQ>")

    blocks: Dict[str, Dict[str, np.ndarray]] = {}
    cur_block = cur_inst = None
    cur_tokens: List[int] = []

    for tid in arr:
        tid = int(tid)
        sym = tokenizer.rev_get(tid)
        if tid in block_ids:
            cur_block = block_ids[tid]
            blocks[cur_block] = {}
            cur_inst = None
            cur_tokens = []
        elif cur_block is not None:
            if tid == tag_end_id:
                if cur_inst and cur_tokens:
                    blocks[cur_block][cur_inst] = np.array(cur_tokens, dtype=int)
                cur_block = cur_inst = None
                cur_tokens = []
            elif tid == eseq_id:
                if cur_inst and cur_tokens:
                    blocks[cur_block][cur_inst] = np.array(cur_tokens, dtype=int)
                cur_inst = None
                cur_tokens = []
            elif sym.startswith("<INST_"):
                cur_inst = sym[6:-1]
                cur_tokens = []
            elif cur_inst is not None:
                cur_tokens.append(tid)

    return blocks


def reconstruct_melody(tokenizer: Tokenizer, arr: np.ndarray,
                        blank_measures: int = 8) -> np.ndarray:
    sme_id     = tokenizer.get("<SME>")
    blank_id   = tokenizer.get("<BLANK>")
    blank_fill = [sme_id, blank_id] * blank_measures

    blocks = parse_melody_blocks(tokenizer, arr)
    all_insts: set = set()
    for bd in blocks.values():
        all_insts.update(bd.keys())

    if not all_insts:
        return np.array([], dtype=int)

    combined = [0]
    for inst in sorted(all_insts):
        combined.append(tokenizer.get(f"<INST_{inst}>"))
        for bname in ("PAST_M", "CONST_M", "FUTURE_M"):
            if bname in blocks and inst in blocks[bname]:
                combined.extend(blocks[bname][inst].tolist())
            elif bname == "CONST_M":
                combined.extend(blank_fill)

    return np.array(combined, dtype=int)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    os.makedirs(os.path.join(OUT_DIR, "midi"), exist_ok=True)

    print("=== モデルロード ===")
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    progress  = _DefaultLearningProgress()
    args      = MORTMArgs(MODEL_ARGS_PATH)
    model     = MORTM(args=args, progress=progress)
    model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=progress.get_device()))
    model.to(progress.get_device())
    print(f"  {MODEL_WEIGHT}")

    # プロンプト: <EOS> のみ
    prompt = np.array([tokenizer.get("<EOS>")], dtype=int)
    print(f"\n=== 生成開始 (temperature={TEMPERATURE}, top_p={TOP_P}) ===")

    gen_tokens = foundation_generate(
        model, tokenizer, prompt,
        temperature=TEMPERATURE, p=TOP_P, max_steps=MAX_GEN_STEPS
    )
    print(f"\n  生成トークン数: {len(gen_tokens)}")

    full_seq = np.concatenate([prompt, gen_tokens])

    tokenizer.mode(to=TO_MUSIC)

    print("\n=== 生成トークン列 ===")
    print(format_sequence(tokenizer, full_seq))

    print("\n=== ブロック解析 ===")
    sme_id = tokenizer.get("<SME>")
    blocks = parse_melody_blocks(tokenizer, full_seq)
    for bname, inst_data in blocks.items():
        for inst, tokens in inst_data.items():
            print(f"  {bname}/{inst}: {len(tokens)} tokens, "
                  f"{int(np.sum(tokens == sme_id))} 小節")

    print("\n=== MIDI 再構成 ===")
    seq = reconstruct_melody(tokenizer, full_seq)
    if len(seq) > 1:
        out_path = os.path.join(OUT_DIR, "midi", "foundation_random.mid")
        try:
            ct_token_to_midi(tokenizer, torch.tensor(seq), out_path, tempo=120)
            print(f"  保存: {out_path}")
        except Exception as e:
            print(f"  MIDI 保存エラー: {e}")
    else:
        print("  再構成可能な旋律なし")

    log_path = os.path.join(OUT_DIR, "token_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(format_sequence(tokenizer, full_seq) + "\n")
    print(f"  ログ: {log_path}")


if __name__ == "__main__":
    main()
