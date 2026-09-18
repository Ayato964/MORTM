"""評価用 共通ユーティリティ(全eval_*.pyが利用)。

前提: `mortm` は pip 導入済み (`import mortm` が通る)。
"""
import os, sys, glob, json
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mortm.models.mortm import MORTM
from mortm.models.modules.config import MORTMArgs
from mortm.train.train import _DefaultLearningProgress
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_MUSIC

try:
    import paths as P
except ImportError:
    from eval import paths as P

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MINLEN, MAXLEN = 40, 5000


def tokenizer():
    tok = Tokenizer(get_token_converter_pro(TO_MUSIC)); tok.mode(TO_MUSIC); return tok


def find_ckpt(arm_code, scale, budget, seed):
    """学習済み最終ckptを探す。arm_code=A1/A2/A3a/.../A1songmatch。中間ckpt/train.は除外。"""
    base = P.MODEL_DIR_MATCH if arm_code == "A1songmatch" else P.MODEL_DIR
    tags = [f"{arm_code}_{scale}_{budget}_{seed}", f"{arm_code}_{scale}_{budget}"] if seed else [f"{arm_code}_{scale}_{budget}"]
    for tag in tags:
        fs = [f for f in glob.glob(os.path.join(base, tag, "*.pth"))
              if "ckpt" not in os.path.basename(f) and ".train." not in os.path.basename(f)]
        if fs:
            return sorted(fs, key=os.path.getmtime, reverse=True)[0]
    return None


def load_model(ckpt, scale):
    prog = _DefaultLearningProgress()
    try: prog.set_device(DEV)
    except Exception: pass
    m = MORTM(MORTMArgs(P.CFG[scale]), prog).to(DEV)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.eval()
    return m


def load_seqs(manifest, n, seed=0):
    """npzパス列挙(json)から MINLEN<len<MAXLEN の系列を n 本まで返す。"""
    paths = json.load(open(manifest)); rng = np.random.RandomState(seed); rng.shuffle(paths)
    seqs = []
    for p in paths:
        try:
            with np.load(p, allow_pickle=True) as d:
                i = 1
                while f"array{i}" in d.files:
                    a = np.asarray(d[f"array{i}"])
                    if a.ndim > 0 and MINLEN < len(a) < MAXLEN:
                        seqs.append(a.astype(np.int64))
                    i += 1
        except Exception:
            continue
        if len(seqs) >= n: break
    return seqs[:n]


def mean_sd(vals):
    v = np.asarray(vals, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1)) if len(v) > 1 else 0.0)
