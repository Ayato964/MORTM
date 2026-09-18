"""keyの誤り構造を分類する。A1-sft-800M を 8小節MIDICapsセットで自由生成し、
各サンプルで key位置の logit から: pred(top1)/gt順位/top5内か/gtとの関係(完全・平行・相対・5度・その他)を集計。
「誤りは主に5度/相対のscale保存的な近傍か? それともランダムか?」を直接判定。
"""
import os, sys, json, collections
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mortm.train.tokenizer import Tokenizer, TO_MUSIC, get_token_converter_pro
import eval_e3_analysis as E3A
from eval_analysis_acc import build_class_sets, CFG

DEV = torch.device("cuda:0")
NEW = "/home/takaaki-nagoshi/data/paper/TEST-TASK-MIDICAPS-DENSE-8BAR"
PCn = {'C':0,'B#':0,'C#':1,'D-':1,'Db':1,'D':2,'D#':3,'E-':3,'Eb':3,'E':4,'F-':4,'Fb':4,
       'F':5,'E#':5,'F#':6,'G-':6,'Gb':6,'G':7,'G#':8,'A-':8,'Ab':8,'A':9,'A#':10,'B-':10,'Bb':10,'B':11,'C-':11,'Cb':11}


def pcmode(name):
    if not name.startswith("k_"): return (None, None)
    body = name[2:]; mode = 'm' if body.endswith('m') else 'M'; ton = body[:-1] if body[-1] in 'Mm' else body
    return (PCn.get(ton), mode)


def rel(gt_name, pr_name):
    gp, gm = pcmode(gt_name); pp, pm = pcmode(pr_name)
    if gp is None or pp is None: return "other"
    if gp == pp and gm == pm: return "correct"
    d = (pp - gp) % 12
    if gm == pm and d in (5, 7): return "fifth"          # 属調/下属調(6/7音共通)
    if gm == 'M' and pm == 'm' and d == 9: return "relative"  # 平行調(7/7音共通)
    if gm == 'm' and pm == 'M' and d == 3: return "relative"
    if gp == pp and gm != pm: return "parallel"          # 同主調(音3つ違い)
    return "other"


def main():
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC)); tok.mode(TO_MUSIC)
    ki, di, gi = build_class_sets(tok)
    key_ids = list(int(x) for x in ki); key_set = set(key_ids)
    idname = {int(i): tok.rev_tokens[int(i)] for i in key_ids}
    cfg = CFG["80M"]
    ck = "out/models/paper/E3/A1_sft_800M/MORTM.E3-A1-sft-800M.train.0.0.4846.pth"
    META, SYSTEM = tok.get("<META>"), tok.get("<SYSTEM>")

    lines = [json.loads(l) for l in open(f"{NEW}/analysis_key.jsonl") if json.loads(l).get("gt_key") is not None]
    prompts = []
    for ln in lines:
        p = list(ln["prompt"])
        if p and p[-1] == SYSTEM: p = p[:-1]
        prompts.append(p + [META, SYSTEM])

    model = E3A.load_model(ck, cfg, True)
    gens = E3A.generate(model, prompts, tok)

    ranks = []; taxo = collections.Counter(); top1 = 0; top5 = 0; n = 0
    err_taxo = collections.Counter()
    key_ids_t = torch.tensor(key_ids)
    for ln, (gt, gl) in zip(lines, gens):
        gtv = int(ln["gt_key"])
        found = None
        for tok_i, lg in zip(gt, gl):
            if tok_i in key_set:
                found = (tok_i, lg); break
        if found is None:
            n += 1; ranks.append(len(key_ids)); taxo["no_emit"] += 1; continue
        pred, lg = found
        sub = lg[key_ids_t]                      # key クラスだけに制限
        order = [key_ids[j] for j in torch.argsort(sub, descending=True).tolist()]
        n += 1
        rank = order.index(gtv) + 1 if gtv in order else len(key_ids)
        ranks.append(rank)
        if pred == gtv: top1 += 1
        if gtv in order[:5]: top5 += 1
        r = rel(idname[gtv], idname[pred])
        taxo[r] += 1
        if r != "correct":
            err_taxo[r] += 1
    del model; torch.cuda.empty_cache()

    ranks = np.array(ranks)
    print(f"n={n}  クラス数={len(key_ids)}")
    print(f"top1={top1/n:.3f}  top5={top5/n:.3f}  gt順位: 中央値={np.median(ranks):.0f} 平均={ranks.mean():.1f}")
    print("全予測の関係内訳:", dict(taxo))
    tot_err = sum(err_taxo.values())
    print(f"\n誤り総数={tot_err}  誤りの内訳(scale保存的な近傍か?):")
    for k in ("relative", "fifth", "parallel", "other", "no_emit"):
        if err_taxo.get(k):
            print(f"  {k:9s}: {err_taxo[k]:4d}  ({err_taxo[k]/tot_err*100:.1f}% of errors)")
    near = err_taxo.get("relative",0)+err_taxo.get("fifth",0)+err_taxo.get("parallel",0)
    print(f"\n近傍誤り(relative+fifth+parallel) = {near}/{tot_err} = {near/tot_err*100:.1f}% of errors")
    print(f"真の主音がtop-5内 = {top5/n*100:.1f}%  (chance top5 = {5/len(key_ids)*100:.1f}%)")


if __name__ == "__main__":
    main()
