"""v1.8c 凍結・客観評価軸の正準実装（研究設計書 §6.2）。
軸1 スケール一致率 / 軸2 分布一致率 / 軸3 リズム一致率(R-a,R-b) / 軸4 多様性 / 音域整合。
軸5(NLL/分析ACC)と軸2の分布JS/W1は既存 metrics.py を使用。全て pitch-class/tick ベースで異名同音免疫。
note = (measure, shift(0-95), pitch(0-127 MIDI), dur)  ← metrics.parse_notes の出力形式。
onset_tick = measure*96 + shift, beat=24tick, 96分割/小節。
"""
from __future__ import annotations
import numpy as np
from .metrics import js_divergence, wasserstein1, _hist

# --- キートークン -> (主音pc, is_major) ---
_TONIC_PC = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"F":5,"F#":6,"Gb":6,
             "G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11}
MAJOR_SET = {0,2,4,5,7,9,11}                 # 全音階7音
MINOR_SET = {0,2,3,5,7,8,10,11}              # 自然的短音階 ∪ 導音(11) = 8音

def build_key_map(tok):
    m={}
    for name,tid in tok.tokens.items():
        s=str(name)
        if not s.startswith("k_"): continue
        body=s[2:]
        mode=body[-1]; tonic=body[:-1]
        if tonic in _TONIC_PC:
            m[tid]=(_TONIC_PC[tonic], mode=="M")
    return m

def scale_agreement(notes, key_token, keymap):
    """指定キーのスケール集合に属するピッチクラス比率。GTとの差で評価すること。"""
    if not notes or key_token not in keymap: return float("nan")
    tonic,is_major=keymap[key_token]
    base=MAJOR_SET if is_major else MINOR_SET
    scale={(tonic+x)%12 for x in base}
    inb=sum(1 for _,_,p,_ in notes if (p%12) in scale)
    return inb/len(notes)

def all_scales_agreement(notes):
    """24キー(主音0-11 x 長/短)それぞれのスケール適合率を計算し、(最大適合率, 最大キー名, プロファイル)を返す。"""
    if not notes: return float("nan"), "None", [float("nan")] * 24
    profile = []
    # 24調の適合率を算出
    for tonic in range(12):
        for is_major in [True, False]:
            base = MAJOR_SET if is_major else MINOR_SET
            scale = {(tonic + x) % 12 for x in base}
            inb = sum(1 for _, _, p, _ in notes if (p % 12) in scale)
            profile.append(inb / len(notes))
    # 最大適合率とそのインデックスを特定
    max_idx = int(np.argmax(profile))
    max_score = profile[max_idx]
    
    # キー名の復元
    mode = "M" if max_idx % 2 == 0 else "m"
    tonic_idx = max_idx // 2
    # 逆引き tonic 名
    tonic_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    max_key_name = f"k_{tonic_names[tonic_idx]}{mode}"
    
    return float(max_score), max_key_name, profile


# --- 軸3 R-a: グルーヴ級(even/swing/undetermined) ---
def _onsets(notes):
    return sorted(m*96+s for m,s,_,_ in notes)

def groove_class(notes, even_lo=0.75, even_hi=1.33, swing_lo=1.5, swing_hi=3.0):
    """連続する2つのIOIが1拍(≈24tick)を分割する場合、比 r=ioi1/ioi2 で級判定。
    多数決で {even, swing, undetermined} を返す。閾値はGT分布から凍結する想定(既定は暫定)。"""
    on=_onsets(notes)
    if len(on)<3: return "undetermined"
    iois=[on[i+1]-on[i] for i in range(len(on)-1) if 0<on[i+1]-on[i]<=24]
    even=swing=0
    for i in range(len(iois)-1):
        a,b=iois[i],iois[i+1]
        if 18<=a+b<=30 and b>0:                # 合計≈1拍(24)の8分ペア
            r=a/b
            if even_lo<=r<=even_hi: even+=1
            elif swing_lo<=r<=swing_hi or (1/swing_hi<=r<=1/swing_lo): swing+=1
    if even+swing==0: return "undetermined"
    return "even" if even>=swing else "swing"

def groove_agreement(gen_notes_list, ctx_notes_list, gt_notes_list):
    """R-a: P(級(生成)=級(文脈) | 文脈が判定可) と、GT(真CONST vs 文脈)の同率を返す。"""
    def agree(cands, ctxs):
        num=den=0
        for c,ctx in zip(cands,ctxs):
            gc=groove_class(ctx)
            if gc=="undetermined": continue
            den+=1; num+=int(groove_class(c)==gc)
        return num/den if den else float("nan")
    return agree(gen_notes_list,ctx_notes_list), agree(gt_notes_list,ctx_notes_list)

# --- 軸3 R-b: 拍内位相ヒスト(24) JS(文脈 vs 生成) ---
def intra_beat_phase_hist(notes):
    h=np.zeros(24)
    for m,s,_,_ in notes:
        h[(m*96+s)%24]+=1
    return h/h.sum() if h.sum()>0 else h

def phase_js(gen_notes, ctx_notes):
    return js_divergence(intra_beat_phase_hist(gen_notes), intra_beat_phase_hist(ctx_notes))

# --- 軸4 多様性: 標本間 distinct pitch-interval 3-gram (self-BLEU型) ---
def _interval_ngrams(notes, n=3):
    ps=[p for _,_,p,_ in sorted(notes)]
    iv=[ps[i+1]-ps[i] for i in range(len(ps)-1)]
    return [tuple(iv[i:i+n]) for i in range(len(iv)-n+1)]

def inter_sample_distinct(samples_notes, n=3):
    """全生成標本の pitch-interval n-gram の distinct率(高=多様, mode collapse検出)。"""
    all_g=[];
    for notes in samples_notes:
        all_g+=_interval_ngrams(notes,n)
    if not all_g: return float("nan")
    return len(set(all_g))/len(all_g)

# --- 音域整合 ---
def pitch_range(notes):
    if not notes: return (float("nan"),float("nan"))
    ps=[p for _,_,p,_ in notes]
    return (float(np.mean(ps)), float(max(ps)-min(ps)))

def register_jump(gen_notes, ctx_notes):
    """継ぎ目レジスタ跳躍: |生成平均pitch − 文脈平均pitch|。GTの同値と比較する。"""
    g=pitch_range(gen_notes)[0]; c=pitch_range(ctx_notes)[0]
    return abs(g-c) if (g==g and c==c) else float("nan")
