"""CARL の実行に必要なチェックポイント/設定の集約。ここだけ書き換えれば環境を移せる。

3 モデルの役割:
  MORTM-gem      : 生成器。学習対象(偶数ラウンド)。多タスク SFT 済みの重みから始める。
  MORTM-ana      : 分析器 + AI/Human 判別器。学習対象(奇数ラウンド)。
                   基盤 MORTM に LoRA + PMA + OutHead を足して作る。
  MORTM-gem-base : KL の参照方策。gem の初期重みを凍結して保持する(= SFT 直後の gem)。
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _p(*a):
    return os.path.join(ROOT, *a)


# --- 生成器(多タスク SFT 済み) -----------------------------------------
# meta / meta_past / meta_future / infill / inst_comp を解けるモデル。
GEM_CKPT = _p("out/models/mortm/sft/generation/MORTM4.5D-160M-SFT-gen.pth")
GEM_CONFIG = _p("configs/models/mortm/rl/carl_160M_rl.json")

# --- KL 参照(gem の初期重みそのもの) ------------------------------------
# 別ファイルを用意せず GEM_CKPT を凍結ロードするのが既定。学習を再開する場合は
# 「そのラウンド開始時点の gem」ではなく **SFT 直後** を指し続けること
# (参照が動くと KL が「直前の自分との差」になり、報酬ハッキングの検出力を失う)。
BASE_CKPT = GEM_CKPT
BASE_CONFIG = GEM_CONFIG

# --- 分析器の初期重み ---------------------------------------------------
# 基盤 MORTM から始める(分析方向は Any-Order 事前学習で獲得済み)。
# 分析 SFT 済みの重みがあるならそちらを指す方が収束が速い。
ANA_INIT_CKPT = GEM_CKPT
ANA_CONFIG = GEM_CONFIG
# 注: この config は preview4/160M.json に LoRA 設定(attn+ffn, r=4)を足したもの。
#     既定の lora_r=8 / use_attn_lora=False では SFT 済み重みがロードできない。

# --- データ -------------------------------------------------------------
# 奇数ラウンドの人間サンプル(分析 SFT 形式の npz を列挙した json)。
# ※ 生成SFT(sft/generation)の npz は `<META>` トリガを持たない別形式なので使えない。
#    必ず分析SFT(sft/analysis)側を指すこと。
ANALYSIS_MANIFEST = _p("out/models/mortm/rl/analysis_paths.json")

# --- 生成SFTデータ(CARL 本番で使うのはこちら) ---------------------------
# MORTM4.5D-160M-SFT-gen を学習したデータそのもの(81,457 ファイル)。
# meta / meta_past / meta_future / infill / inst_comp の全タスクを含み、
# PAST/CONST/FUTURE を時系列連結すると最大 24 小節の CONST になる。
# 奇数(判別器)・偶数(GRPO)の両フェーズがこの同一データを共有する。
GEN_SFT_MANIFEST = "/home/takaaki-nagoshi/data/sft/generation/train.json"
GEN_SFT_EVAL = "/home/takaaki-nagoshi/data/sft/generation/eval.json"

# --- 出力 ---------------------------------------------------------------
OUT_DIR = _p("out/models/mortm/rl/carl")


def resolve(path: str) -> str:
    """存在確認つきでパスを返す。落ちるなら学習開始前に落とす。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CARL: 見つかりません -> {path}")
    return path


def check_all() -> dict:
    """必要ファイルの存在を一括確認する(学習前の事前チェック用)。"""
    items = {"GEM_CKPT": GEM_CKPT, "GEM_CONFIG": GEM_CONFIG,
             "BASE_CKPT": BASE_CKPT, "ANA_INIT_CKPT": ANA_INIT_CKPT,
             "ANA_CONFIG": ANA_CONFIG, "ANALYSIS_MANIFEST": ANALYSIS_MANIFEST,
             "GEN_SFT_MANIFEST": GEN_SFT_MANIFEST}
    return {k: os.path.exists(v) for k, v in items.items()}
