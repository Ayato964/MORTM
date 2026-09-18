"""CoT版 生成SFTモデルの制御性テスト。
検証軸:
  A) 長さ制御   : <GEN_MEASURE_COUNT_n> を変えて生成CONST内の<SME>小節数が n に追従するか
  B) 密度制御   : <NOTE_DENSE_x> を低/高で変えて生成CONSTの実音符密度が単調に変わるか
  C) CoT動作    : systemタグ末尾<thinking>付きmeta_pastで、<MGEN>直後にフルmeta(<SYSTEM>..<TAG_END>)を
                  書き出してから旋律を生成するか
  D) 試聴MIDI   : 代表サンプルをMIDI保存
ベース重みではなくSFT(LoRA r4 + Emb/Wout)チェックポイントを使用。
"""
import os, sys, json, numpy as np, torch
from collections import defaultdict
# eval/ から実行すると installed版mortm を拾うため、学習に使った repo版を最優先にする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN, TO_MUSIC
from mortm.utils.de_convert import ct_token_to_midi

MODEL_ARGS = "configs/models/mortm/foundation/80M.json"
SFT_WEIGHT = "out/models/mortm/sft/generation/MORTM.4.5D-Lite-SFT-gen.pth"  # CoT版 (6/29 18:13, val0.9605)
OUT_DIR    = "out/sft_gen_test"
TEMPERATURE = 1.0
TOP_P       = 0.92
MAX_STEPS   = 1200
N_PER_COND  = 8          # 各条件のサンプル数


def reset_kv_cache(model):
    for m in model.modules():
        if hasattr(m, "kv_cache") and hasattr(m, "cache_seqlens"):
            m.kv_cache = None
            m.cache_seqlens = None


def load_model():
    progress = _DefaultLearningProgress()
    args = MORTMArgs(MODEL_ARGS)
    # SFT時と同じLoRA構成でモデルを建てないと重みが噛み合わない
    args.use_attn_lora = True
    args.use_ffn_lora = True
    args.use_gate_lora = False
    args.lora_r = 4
    model = MORTM(args=args, progress=progress)
    sd = torch.load(SFT_WEIGHT, map_location=progress.get_device())
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(progress.get_device()).eval()
    return model, progress.get_device()


@torch.inference_mode()
def generate(model, device, prompt_ids, max_steps=MAX_STEPS, temperature=TEMPERATURE, p=TOP_P):
    """prompt_ids(末尾は<MGEN>想定)を投入し、<TE>まで生成。生成分のみ返す。"""
    reset_kv_cache(model)
    te_id = TK.get("<TE>")
    src = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
    gen = []
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pad = (src != model.embedding.padding_idx)
        logits = model.forward(src, padding_mask=pad, is_causal=True, is_save_cache=True)
    nxt = model.top_p_sampling(logits[:, -1, :], p=p, temperature=temperature)
    for _ in range(max_steps):
        t = nxt.item()
        if t == te_id:
            break
        gen.append(t)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(nxt, padding_mask=None, is_causal=True, is_save_cache=True)
        nxt = model.top_p_sampling(logits.squeeze(1), p=p, temperature=temperature)
    return np.array(gen, dtype=int)


# --- プロンプト構築 (full meta, CoTなし: meta タスク) ---
def build_meta_prompt(program, dense, gmc, genres, key):
    b = [TK.get("<EOS>"), TK.get("<SYSTEM>"), TK.get(f"<INST_{program}>")]
    if dense is not None:
        b.append(TK.get(f"<NOTE_DENSE_{dense}>"))
    if gmc is not None:
        b.append(TK.get(f"<GEN_MEASURE_COUNT_{gmc}>"))
    for g in genres:
        b.append(TK.get(f"<GENRE_{g}>"))
    b.append(TK.get(f"k_{key}"))
    b.append(TK.get("<TAG_END>"))
    b.append(TK.get("<MGEN>"))
    return np.array(b, dtype=int)


def count_measures(gen):
    return int(np.sum(gen == TK.get("<SME>")))


def count_notes(gen):
    lo, hi = TK.get_length_tuple("s")
    return int(np.sum((gen >= lo) & (gen < hi)))


def main():
    os.makedirs(os.path.join(OUT_DIR, "midi"), exist_ok=True)
    global TK
    TK = Tokenizer(get_token_converter_pro(TO_TOKEN))
    model, device = load_model()
    print(f"[model] {SFT_WEIGHT}\n")

    report = []

    # ============ A) 長さ制御 ============
    print("=" * 60)
    print("A) 長さ制御テスト (<GEN_MEASURE_COUNT_n>, PIANO/dense5/rock/k_CM)")
    print("=" * 60)
    for n in (2, 4, 8):
        meas = []
        for _ in range(N_PER_COND):
            pr = build_meta_prompt("PIANO", 5, n, ["rock"], "CM")
            g = generate(model, device, pr)
            meas.append(count_measures(g))
        arr = np.array(meas)
        print(f"  要求 n={n:>2} → 生成小節数 mean={arr.mean():.2f}  中央={int(np.median(arr))}  "
              f"全={meas}  命中率={(arr==n).mean()*100:.0f}%")
        report.append(("length", n, float(arr.mean()), meas))

    # ============ B) 密度制御 ============
    print("\n" + "=" * 60)
    print("B) 密度制御テスト (<NOTE_DENSE_x>, PIANO/GMC4/rock/k_CM)")
    print("=" * 60)
    for dense in (2, 9):
        npm = []
        for _ in range(N_PER_COND):
            pr = build_meta_prompt("PIANO", dense, 4, ["rock"], "CM")
            g = generate(model, device, pr)
            m = max(count_measures(g), 1)
            npm.append(count_notes(g) / m)
        arr = np.array(npm)
        print(f"  要求 dense={dense} → 実音符/小節 mean={arr.mean():.2f}  全={[round(x,1) for x in npm]}")
        report.append(("density", dense, float(arr.mean()), npm))

    # ============ C) CoT動作 ============
    print("\n" + "=" * 60)
    print("C) CoT動作テスト (meta_past + <thinking>)")
    print("=" * 60)
    TK.mode(TO_MUSIC)  # decode用
    # eval から実 PAST 旋律を借りる
    ev = json.load(open("/home/takaaki-nagoshi/data/sft/generation/eval.json"))
    past_block = None
    for path in ev[:50]:
        z = np.load(path, allow_pickle=True)
        for k in z.keys():
            a = z[k]
            if a.shape == ():
                continue
            toks = a.tolist()
            if TK.get("<PAST_M>") in toks and TK.get("<MGEN>") in toks:
                mgen_i = toks.index(TK.get("<MGEN>"))
                past_start = toks.index(TK.get("<PAST_M>"))
                past_block = toks[past_start:mgen_i]   # <PAST_M>..<TAG_END>
                break
        if past_block:
            break
    # thinking付きの部分system + PASTブロック + <MGEN>
    sys_blk = [TK.get("<EOS>"), TK.get("<SYSTEM>"), TK.get("<INST_PIANO>"),
               TK.get("<GEN_MEASURE_COUNT_4>"), TK.get("<GENRE_jazz>"), TK.get("k_CM"),
               TK.get("<thinking>"), TK.get("<TAG_END>")]
    prompt = np.array(sys_blk + past_block + [TK.get("<MGEN>")], dtype=int)
    g = generate(model, device, prompt, max_steps=1500)
    decoded = [TK.rev_get(int(t)) for t in g]
    # <MGEN>直後にフルmeta(<SYSTEM>..<TAG_END>)が出ているか
    cot_ok = decoded[0] == "<SYSTEM>" and "<TAG_END>" in decoded[:40]
    head = " ".join(decoded[:24])
    print(f"  CoT thinking フェーズ検出: {'YES' if cot_ok else 'NO'}")
    print(f"  生成冒頭24tok: {head}")
    report.append(("cot", cot_ok, head, None))

    # ============ D) 試聴MIDI ============
    print("\n" + "=" * 60)
    print("D) 試聴MIDI生成 (3条件)")
    print("=" * 60)
    samples = [
        ("piano_8m_rock_CM", build_meta_prompt("PIANO", 6, 8, ["rock"], "CM")),
        ("piano_8m_jazz_Am", build_meta_prompt("PIANO", 6, 8, ["jazz"], "Am")),
        ("sax_8m_pop_GM",    build_meta_prompt("SAX",   5, 8, ["pop"],  "GM")),
    ]
    for name, pr in samples:
        g = generate(model, device, pr, max_steps=1500)
        # 生成分は <INST_x> seq <ESEQ> ... 形式。MIDI再構成用に [0,<INST_x>,seq..] を作る
        seq = np.concatenate([[0], g])
        out = os.path.join(OUT_DIR, "midi", f"{name}.mid")
        try:
            ct_token_to_midi(TK, torch.tensor(seq), out, tempo=120)
            print(f"  保存 {out}  (tok={len(g)}, 小節={count_measures(g)}, 音符={count_notes(g)})")
        except Exception as e:
            print(f"  {name}: MIDI保存エラー {e}")

    json.dump([{"kind": r[0], "cond": r[1], "metric": r[2]} for r in report],
              open(os.path.join(OUT_DIR, "report.json"), "w"), ensure_ascii=False, indent=2)
    print(f"\n完了。出力: {OUT_DIR}/")


if __name__ == "__main__":
    main()
