"""評価・テスト用パス設定。

プロジェクトルート配下の eval/ から実行されることを前提とし、
環境変数による上書きにも対応。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

# --- データ(リーク無し索引 = npzパス列挙 json) ---
# 構成: {DATA_PAPER}/{arm}/{budget}/train.json, {arm}/val.json, {arm}/test.json
#   arm: A1(=BR)/A2(=chrono)/A3a(=perm)/A3b(=del)/A3c(=metafirst)/A1songmatch(=match)
DATA_PAPER = os.environ.get("MORTM_DATA", os.path.expanduser("~/data/paper"))

# --- 凍結テストセット ---
SPLIT_DIR = os.environ.get("MORTM_SPLITS", os.path.join(_ROOT, "mortm_repro", "data", "splits"))
TESTSEQ        = os.path.join(DATA_PAPER, "A2", "test.json")   # 順方向NLL(chrono held-out)
TESTSEQ_ALLDIR = os.path.join(DATA_PAPER, "A1", "test.json")   # 全方向held-out(continuation)
TEST_TASK      = os.environ.get("MORTM_TESTTASK", os.path.join(DATA_PAPER, "TEST-TASK"))

# --- 学習済みモデル(探索先) ---
# 命名: {MODEL_DIR}/{arm}_{scale}_{budget}[_{seed}]/MORTM.*_{loss}.pth
MODEL_DIR       = os.environ.get("MORTM_MODELS",       os.path.expanduser("~/out/models/paper/E1"))
MODEL_DIR_MATCH = os.environ.get("MORTM_MODELS_MATCH", os.path.expanduser("~/out/models/paper/E5"))

# --- モデルconfig(アーキ) ---
CFG = {
    "10M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "scaling", "10M.json"),
    "80M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "80M.json"),
    "160M": os.path.join(_ROOT, "configs", "models", "mortm", "foundation", "160M.json"),
}

# フォールバック (mortm_repro 内 config)
if not os.path.exists(CFG["10M"]):
    alt_10m = os.path.join(_ROOT, "mortm_repro", "train", "configs", "model", "10M.json")
    if os.path.exists(alt_10m):
        CFG["10M"] = alt_10m

if not os.path.exists(CFG["80M"]):
    alt_80m = os.path.join(_ROOT, "mortm_repro", "train", "configs", "model", "80M.json")
    if os.path.exists(alt_80m):
        CFG["80M"] = alt_80m

# --- 論文アーム名 <-> 内部コード ---
ARM = {"BR": "A1", "chrono": "A2", "perm": "A3a", "del": "A3b", "metafirst": "A3c", "match": "A1songmatch"}
SEEDS = ["s42", "s1337", "s2024"]
