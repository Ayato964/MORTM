"""CoTトリガー(<thinking>)の対照実験。
同一条件で <thinking> あり/なし を切替え、<MGEN>直後にフルmeta(<SYSTEM>..<TAG_END>)が
出るかを複数サンプルで比較。thinking が CoT のトリガーとして機能しているかを検証。
"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN, TO_MUSIC

MODEL_ARGS = "configs/models/mortm/foundation/80M.json"
SFT_WEIGHT = "out/models/mortm/sft/generation/MORTM.4.5D-Lite-SFT-gen.pth"
N = 12
TEMP, TOPP, MAXS = 1.0, 0.92, 1500


def reset_kv(model):
    for m in model.modules():
        if hasattr(m, "kv_cache") and hasattr(m, "cache_seqlens"):
            m.kv_cache = None; m.cache_seqlens = None


def load_model():
    pr = _DefaultLearningProgress(); a = MORTMArgs(MODEL_ARGS)
    a.use_attn_lora = True; a.use_ffn_lora = True; a.use_gate_lora = False; a.lora_r = 4
    m = MORTM(args=a, progress=pr)
    sd = torch.load(SFT_WEIGHT, map_location=pr.get_device())
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    mi, un = m.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(mi)} unexpected={len(un)}")
    return m.to(pr.get_device()).eval(), pr.get_device()


@torch.inference_mode()
def gen(model, device, ids, max_steps=MAXS):
    reset_kv(model); te = TK.get("<TE>")
    src = torch.tensor(ids, device=device, dtype=torch.long).unsqueeze(0)
    out = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        lo = model.forward(src, padding_mask=(src != model.embedding.padding_idx),
                           is_causal=True, is_save_cache=True)
    nx = model.top_p_sampling(lo[:, -1, :], p=TOPP, temperature=TEMP)
    for _ in range(max_steps):
        t = nx.item()
        if t == te: break
        out.append(t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = model.forward(nx, padding_mask=None, is_causal=True, is_save_cache=True)
        nx = model.top_p_sampling(lo.squeeze(1), p=TOPP, temperature=TEMP)
    return out


def build_past_prompt(thinking: bool, past_block):
    sys_blk = [TK.get("<EOS>"), TK.get("<SYSTEM>"), TK.get("<INST_PIANO>"),
               TK.get("<GEN_MEASURE_COUNT_4>"), TK.get("<GENRE_jazz>"), TK.get("k_CM")]
    if thinking:
        sys_blk.append(TK.get("<thinking>"))
    sys_blk.append(TK.get("<TAG_END>"))
    return np.array(sys_blk + past_block + [TK.get("<MGEN>")], dtype=int)


def has_cot_meta(gen_ids):
    """生成の冒頭に <SYSTEM>..<TAG_END> のメタブロックが旋律より先に出ているか。"""
    if not gen_ids:
        return False
    sys_id, tag_end, sme = TK.get("<SYSTEM>"), TK.get("<TAG_END>"), TK.get("<SME>")
    if gen_ids[0] != sys_id:
        return False
    # 最初の <TAG_END> が 最初の <SME>(旋律開始) より前にあるか
    te_pos = gen_ids.index(tag_end) if tag_end in gen_ids else 1e9
    sme_pos = gen_ids.index(sme) if sme in gen_ids else 1e9
    return te_pos < sme_pos


def main():
    global TK
    TK = Tokenizer(get_token_converter_pro(TO_TOKEN))
    model, device = load_model()
    TK.mode(TO_MUSIC)

    # 実 PAST 旋律を借りる
    ev = json.load(open("/home/takaaki-nagoshi/data/sft/generation/eval.json"))
    past_block = None
    for path in ev[:80]:
        z = np.load(path, allow_pickle=True)
        for k in z.keys():
            a = z[k]
            if a.shape == (): continue
            toks = a.tolist()
            if TK.get("<PAST_M>") in toks and TK.get("<MGEN>") in toks:
                mi = toks.index(TK.get("<MGEN>")); ps = toks.index(TK.get("<PAST_M>"))
                past_block = toks[ps:mi]; break
        if past_block: break

    print(f"\n=== CoT対照実験 (同一条件 PIANO/GMC4/jazz/k_CM, N={N}/条件) ===\n")
    for thinking in (False, True):
        hits = 0; first_examples = []
        for i in range(N):
            g = gen(model, device, build_past_prompt(thinking, past_block))
            ok = has_cot_meta(g)
            hits += ok
            if len(first_examples) < 2:
                head = " ".join(TK.rev_get(int(t)) for t in g[:12])
                first_examples.append(f"      [{'CoT' if ok else '直接'}] {head}")
        label = "<thinking> あり" if thinking else "<thinking> なし"
        print(f"  {label:<16}: metaブロック先行出力 = {hits}/{N} ({hits/N*100:.0f}%)")
        for ex in first_examples:
            print(ex)
        print()


if __name__ == "__main__":
    main()
