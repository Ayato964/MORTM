"""MIDICaps真キーでのキー評価セットを作る。各窓:
 prompt(analysis_key, P形式)/ gt_true(MIDICaps train.json 真キー token)/ gt_ks_win(窓PCヒストのK-Sキー token)
真キー=データセット付与ラベル(アルゴリズム非依存)。gt_ks_win=断片から決まる窓レベルアルゴリズム上限の代理。
出力: /home/.../data/paper/MIDICAPS-KEY/eval.jsonl
"""
import json, os, glob, random
import numpy as np
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN
from mortm.utils.convert import MIDIConverter
from mortm.utils.convert_foundation import FoundationDataMaker
from mortm.eval.protocol_builder import ProtocolBuilder

R="/media/takaaki-nagoshi/MIDIdatasets/MIDI_Caps"; LMD=R+"/lmd_full"
OUT="/home/takaaki-nagoshi/data/paper/MIDICAPS-KEY"; PROGRAMS=["PIANO","SAX"]
TARGET=int(os.environ.get("TARGET","400"))
PC={'C':0,'B#':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'Fb':4,'F':5,'E#':5,'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11,'Cb':11}
INT2MAJ={0:'CM',1:'C#M',2:'DM',3:'D#M',4:'EM',5:'FM',6:'F#M',7:'GM',8:'G#M',9:'AM',10:'A#M',11:'BM'}
INT2MIN={0:'Cm',1:'C#m',2:'Dm',3:'D#m',4:'Em',5:'Fm',6:'F#m',7:'Gm',8:'G#m',9:'Am',10:'A#m',11:'Bm'}
# Krumhansl-Kessler profiles
KKmaj=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KKmin=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def ks_key(pchist):
    if pchist.sum()==0: return None
    h=pchist-pchist.mean(); best=None; bc=-2
    for mode,prof in (('M',KKmaj),('m',KKmin)):
        p=prof-prof.mean()
        for tonic in range(12):
            r=np.roll(prof,tonic); r=r-r.mean()
            c=np.dot(h,r)/(np.linalg.norm(h)*np.linalg.norm(r)+1e-9)
            if c>bc: bc=c; best=(tonic,mode)
    return best

def parse_true(s):  # 'F# minor'->token名
    p=s.split(); pc=PC.get(p[0].replace('-','b'));
    if pc is None: return None
    return INT2MIN[pc] if 'min' in p[1].lower() else INT2MAJ[pc]

def main():
    tk=Tokenizer(get_token_converter_pro(TO_TOKEN)); rev={v:k for k,v in tk.tokens.items()}
    pb=ProtocolBuilder(tk,PROGRAMS)
    plo,phi=tk.get_length_tuple("p")
    gm={}
    for line in open(R+"/json/train.json"):
        line=line.strip()
        if not line: continue
        try: d=json.loads(line)
        except: continue
        if d.get("location") and d.get("key"): gm[d["location"]]=d["key"]
    os.makedirs(OUT,exist_ok=True); fo=open(os.path.join(OUT,"eval.jsonl"),"w")
    files=glob.glob(os.path.join(LMD,"*","*.mid")); random.seed(0); random.shuffle(files)
    n=0; proc=0
    for fp in files:
        if n>=TARGET: break
        loc=os.path.relpath(fp,R)
        if loc not in gm: continue
        tk_true=parse_true(gm[loc])
        if tk_true is None: continue
        d,f=os.path.dirname(fp),os.path.basename(fp)
        try:
            con=MIDIConverter(tk,d,f,PROGRAMS); con.convert()
            if con.is_error or con.midi2seq is None: continue
            maker=FoundationDataMaker(con,1,8)
            if maker.is_error: continue
            w=next(pb.windows(d,f),None)
            if w is None: continue
            prompt,_=pb.build(w,maker,"analysis_key")
        except Exception: continue
        proc+=1
        # 窓のPCヒスト(prompt内 p_ トークン)
        pch=np.zeros(12)
        for t in prompt:
            nm=rev.get(int(t),"")
            if nm.startswith("p_"):
                try: pch[int(nm[2:])%12]+=1
                except: pass
        ks=ks_key(pch)
        ks_tok = (INT2MIN if (ks and ks[1]=='m') else INT2MAJ).get(ks[0]) if ks else None
        # 学習ラベルの算法=全曲music21合議キー(con.key_dict[0])。モデルはこれを再現するよう学習。
        m21_tok=None
        try:
            kd=con.key_dict[0] if con.key_dict else None
            if kd and kd.get('tonic') and kd.get('tonic')!='Unknown':
                pc=PC.get(str(kd['tonic']).replace('-','b'))
                if pc is not None:
                    m21_tok=(INT2MIN if kd.get('mode')=='minor' else INT2MAJ)[pc]
        except Exception: pass
        gt_true_id=tk.get(f"k_{tk_true}")
        gt_ks_id=tk.get(f"k_{ks_tok}") if ks_tok else None
        gt_m21_id=tk.get(f"k_{m21_tok}") if m21_tok else None
        fo.write(json.dumps({"song":os.path.splitext(f)[0],"programs":w.programs,
                             "prompt":[int(x) for x in prompt],
                             "gt_key":gt_true_id,               # 真キー(MIDICaps)
                             "gt_key_ks_window":gt_ks_id,       # 窓レベルK-S(断片天井)
                             "gt_key_m21_song":gt_m21_id,       # 全曲music21合議(学習ラベル算法)
                             "gt_dense":{p_:int(dv) for p_,dv in w.info}})+"\n")
        n+=1
        if proc%200==0: print(f"proc {proc}, collected {n}",flush=True)
    fo.close(); print(f"[MIDICAPS-KEY] done: {n} 窓 -> {OUT}/eval.jsonl")

if __name__=="__main__":
    main()
