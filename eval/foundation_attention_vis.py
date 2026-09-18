"""
Foundation モデルの Attention 可視化スクリプト。

生成の各ステップで「どのトークンに注意を向けているか」を
全層・全ヘッドにわたって収集し、ヒートマップで描画する。
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import List

from mortm.models.modules.config import MORTMArgs
from mortm.models.modules.attention import FlashSelfAttentionM
from mortm.models.modules.progress import _DefaultLearningProgress
from mortm.models.mortm import MORTM
from mortm.train.tokenizer import Tokenizer, get_token_converter_pro, TO_TOKEN, TO_MUSIC

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
MODEL_ARGS_PATH = "configs/models/mortm/foundation/80M.json"
MODEL_WEIGHT    = "out/models/mortm/4_5/MORTM.4.5D-80M_1.1424479484558105.pth"
TEMPERATURE     = 1.15
TOP_P           = 0.95
MAX_GEN_STEPS   = 4000
OUT_DIR         = "out/foundation_attention/"

# 可視化設定
# 全ステップ × 全トークンは大きくなりすぎるため、
# 生成ステップを間引いてサンプリングする
VIS_LAYER       = "mean" # 表示するレイヤー番号(int) または "mean"(全層平均)
VIS_HEAD        = "mean"   # 表示するヘッド番号(int) または "mean"(全ヘッド平均)
STEP_STRIDE     = 5        # 何ステップおきに記録するか


# ---------------------------------------------------------------------------
# ブロックカラーマップ (トークン位置への色付け用)
# ---------------------------------------------------------------------------
BLOCK_COLORS = {
    "<EOS>": "#e74c3c",
    "<SYSTEM>": "#e67e22",
    "<PAST_M>": "#2980b9",
    "<CONST_M>": "#27ae60",
    "<FUTURE_M>": "#8e44ad",
    "<TAG_END>": "#95a5a6",
    "<ESEQ>": "#bdc3c7",
}

def _block_color(sym: str) -> str:
    for key, color in BLOCK_COLORS.items():
        if sym == key:
            return color
    if sym.startswith("<INST_"):
        return "#f39c12"
    if sym.startswith("k_"):
        return "#16a085"
    if sym.startswith("s_"):
        return "#85c1e9"
    if sym.startswith("p_"):
        return "#82e0aa"
    if sym.startswith("d_"):
        return "#f9e79f"
    return "#ecf0f1"


def _abbrev(sym: str) -> str:
    """トークン記号の省略表示"""
    if sym.startswith("s_"):  return "s"
    if sym.startswith("p_"):  return "p"
    if sym.startswith("d_"):  return "d"
    if sym.startswith("<NOTE_DENSE_"): return sym[12:-1]+"d"
    return sym


# ---------------------------------------------------------------------------
# KV キャッシュリセット
# ---------------------------------------------------------------------------
def reset_kv_cache(model: MORTM):
    for module in model.modules():
        if hasattr(module, "kv_cache") and hasattr(module, "cache_seqlens"):
            module.kv_cache = None
            module.cache_seqlens = None


# ---------------------------------------------------------------------------
# attention layer 一覧を取得
# ---------------------------------------------------------------------------
def get_attn_layers(model: MORTM) -> List[FlashSelfAttentionM]:
    # self_block はメソッドなので isinstance で直接 FlashSelfAttentionM を探す
    return [m for m in model.modules() if isinstance(m, FlashSelfAttentionM)]


# ---------------------------------------------------------------------------
# 生成 + attention 収集
# ---------------------------------------------------------------------------
DENSE_TRANSITION_STEPS = 30   # ブロック開始直後に密に記録するステップ数


def generate_with_attention(model, tokenizer, prompt, temperature, p, max_steps, stride):
    """
    戻り値:
      generated_tokens : np.ndarray  [T]
      attn_records     : list of dict
          {'step': int, 'token_id': int, 'is_transition': bool,
           'weights': np.ndarray [L, H, S]}  (S はその時点の系列長)
    ブロック開始直後 DENSE_TRANSITION_STEPS ステップは stride に関わらず毎ステップ記録。
    """
    device = model.progress.get_device()
    reset_kv_cache(model)
    attn_layers = get_attn_layers(model)
    print(f"  Attention 観察対象レイヤー数: {len(attn_layers)}")

    future_m_id = tokenizer.get("<FUTURE_M>")
    const_m_id  = tokenizer.get("<CONST_M>")
    past_m_id   = tokenizer.get("<PAST_M>")
    tag_end_id  = tokenizer.get("<TAG_END>")
    te_id       = tokenizer.get("<TE>")

    src = torch.tensor(prompt, device=device, dtype=torch.long).unsqueeze(0)
    model.eval()

    generated         = []
    attn_records      = []
    cur_block         = None
    block_start_step  = -9999   # 現在ブロックが始まった生成ステップ

    with torch.inference_mode():
        # Prefill: キャッシュ構築 + 最初の next_token を一度で取得
        padding_mask = (src != model.embedding.padding_idx)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model.forward(src, padding_mask=padding_mask,
                                   is_causal=True, is_save_cache=True)
        next_token = model.top_p_sampling(logits[:, -1, :], p=p, temperature=temperature)

        print("--- Decoding + Attention 収集 ---")
        for step in range(max_steps):
            tok = next_token.item()
            generated.append(tok)

            # ブロックマーカー追跡 & 停止判定
            if tok in (past_m_id, const_m_id, future_m_id):
                cur_block        = tok
                block_start_step = step
                name = {past_m_id:"PAST_M", const_m_id:"CONST_M", future_m_id:"FUTURE_M"}[tok]
                print(f"\n  [{step}] <{name}> 開始")
            elif tok == tag_end_id:
                if cur_block == future_m_id:
                    print(f"\n  [{step}] <FUTURE_M> 完了 → 停止")
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        model.forward(next_token, padding_mask=None,
                                      is_causal=True, is_save_cache=True)
                    break
                cur_block = None
            elif tok == te_id:
                print(f"\n  [{step}] <TE> → 停止")
                break

            if step % 100 == 0 and step > 0:
                print(f"\r  Step {step}/{max_steps}", end="", flush=True)

            # 次トークン生成: この forward で last_attn_weights がセットされる
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.forward(next_token, padding_mask=None,
                                       is_causal=True, is_save_cache=True)
            next_token = model.top_p_sampling(logits.squeeze(1), p=p, temperature=temperature)

            # ブロック開始直後は密に記録、それ以外は stride 間隔
            is_transition = (step - block_start_step) < DENSE_TRANSITION_STEPS
            if step % stride == 0 or is_transition:
                seq_len = len(prompt) + step + 1
                layer_weights = []
                for la in attn_layers:
                    if hasattr(la, "last_attn_weights"):
                        w = la.last_attn_weights          # Tensor [H, S_cur]
                        H, S_cur = w.shape
                        if S_cur < seq_len:
                            pad = torch.zeros(H, seq_len - S_cur)
                            w = torch.cat([w, pad], dim=1)
                        layer_weights.append(w[:, :seq_len].float().numpy())
                if layer_weights:
                    attn_records.append({
                        "step":          step,
                        "token_id":      tok,
                        "is_transition": is_transition,
                        "weights":       np.stack(layer_weights, axis=0),  # [L, H, S]
                    })

        else:
            print(f"\n  最大ステップ {max_steps} 到達")

    return np.array(generated, dtype=int), attn_records


# ---------------------------------------------------------------------------
# 可視化
# ---------------------------------------------------------------------------
def visualize(tokenizer, prompt, generated, attn_records, out_dir,
              vis_layer, vis_head):

    tokenizer.mode(to=TO_MUSIC)
    full_seq = np.concatenate([prompt, generated])
    syms = [tokenizer.rev_get(int(t)) for t in full_seq]
    total_len = len(full_seq)

    if not attn_records:
        print("  Attention レコードがありません。")
        return

    # ---- 1. ヒートマップ用データ構築 ----
    n_steps = len(attn_records)
    heat    = np.zeros((n_steps, total_len), dtype=np.float32)

    for i, rec in enumerate(attn_records):
        w = rec["weights"]   # [L, H, S]
        S = w.shape[2]

        if vis_layer == "mean":
            w_l = w.mean(axis=0)          # [H, S]
        else:
            w_l = w[vis_layer]            # [H, S]

        if vis_head == "mean":
            w_h = w_l.mean(axis=0)        # [S]
        else:
            w_h = w_l[vis_head]           # [S]

        heat[i, :S] = w_h[:S]

    # ---- 2. トークンごとの背景色リスト ----
    colors = [_block_color(s) for s in syms]

    # ---- 3. ブロック境界の x 位置を検出 ----
    block_markers = {"<PAST_M>", "<CONST_M>", "<FUTURE_M>", "<SYSTEM>", "<EOS>"}
    block_boundaries = [i for i, s in enumerate(syms) if s in block_markers]
    # prompt / generated 境界
    prompt_end = len(prompt)

    # ---- 4. 描画 ----
    fig_w = max(20, total_len * 0.15)
    fig_h = max(8,  n_steps  * 0.15)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h + 3),
                             gridspec_kw={"height_ratios": [1, fig_h]})

    # --- 上段: トークン色帯 ---
    ax_tokens = axes[0]
    for xi, col in enumerate(colors):
        ax_tokens.add_patch(
            mpatches.Rectangle((xi, 0), 1, 1, color=col, lw=0))
    ax_tokens.set_xlim(0, total_len)
    ax_tokens.set_ylim(0, 1)
    ax_tokens.set_xticks(np.arange(total_len) + 0.5)
    labels = [_abbrev(s) for s in syms]
    ax_tokens.set_xticklabels(labels, rotation=90, fontsize=6)
    ax_tokens.set_yticks([])
    ax_tokens.set_title("Token sequence  (colors = block type)", fontsize=9)

    # prompt/generated 境界線
    ax_tokens.axvline(prompt_end, color="red", lw=1.5, label="prompt end")
    for bx in block_boundaries:
        ax_tokens.axvline(bx, color="black", lw=0.5, alpha=0.4)

    # ブロックラベル
    legend_patches = [
        mpatches.Patch(color=c, label=k) for k, c in BLOCK_COLORS.items()
    ] + [mpatches.Patch(color="#f39c12", label="<INST_X>"),
         mpatches.Patch(color="#16a085", label="k_KEY"),
         mpatches.Patch(color="#85c1e9", label="s_N (pos)"),
         mpatches.Patch(color="#82e0aa", label="p_N (pitch)"),
         mpatches.Patch(color="#f9e79f", label="d_N (dur)")]
    ax_tokens.legend(handles=legend_patches, loc="upper right",
                     fontsize=6, ncol=4, framealpha=0.8)

    # --- 下段: Attention ヒートマップ ---
    ax_heat = axes[1]
    im = ax_heat.imshow(heat, aspect="auto", cmap="hot",
                        interpolation="nearest", origin="upper",
                        vmin=0, vmax=heat.max())
    plt.colorbar(im, ax=ax_heat, shrink=0.6, label="attention weight")

    # y軸: 生成ステップ
    step_labels = [str(r["step"]) for r in attn_records]
    tick_every = max(1, n_steps // 30)
    ax_heat.set_yticks(range(0, n_steps, tick_every))
    ax_heat.set_yticklabels(step_labels[::tick_every], fontsize=7)
    ax_heat.set_ylabel("generation step")

    # x軸: トークン位置
    ax_heat.set_xticks(np.arange(total_len))
    ax_heat.set_xticklabels(labels, rotation=90, fontsize=6)
    ax_heat.set_xlabel("token position")

    layer_str = f"layer={vis_layer}" if vis_layer != "mean" else "all-layer-mean"
    head_str  = f"head={vis_head}"  if vis_head  != "mean" else "all-head-mean"
    ax_heat.set_title(f"Attention heatmap  [{layer_str}, {head_str}]  "
                      f"(red line = prompt|generated boundary)", fontsize=9)

    # ブロック境界・prompt境界を縦線で表示
    ax_heat.axvline(prompt_end - 0.5, color="red", lw=1.5)
    for bx in block_boundaries:
        ax_heat.axvline(bx - 0.5, color="cyan", lw=0.8, alpha=0.6)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "attention_heatmap.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  ヒートマップ保存: {out_path}")

    # ---- 5. 各ブロックへの平均注目量を棒グラフで表示 ----
    _plot_block_attention(tokenizer, full_seq, syms, heat, attn_records,
                          prompt_end, out_dir)

    # ---- 6. ブロック遷移時の token-level attention を拡大表示 ----
    _plot_transition_attention(tokenizer, full_seq, syms, attn_records,
                                prompt_end, out_dir,
                                vis_layer=vis_layer, vis_head=vis_head)


def _plot_block_attention(tokenizer, full_seq, syms, heat, attn_records,
                           prompt_end, out_dir):
    """各ブロック領域への平均 attention をステップごとに折れ線プロット"""
    # ブロック領域を検出
    block_ranges = {}
    cur = None
    for i, s in enumerate(syms):
        if s in ("<PAST_M>", "<CONST_M>", "<FUTURE_M>"):
            cur = s
            block_ranges[cur] = [i, i]
        elif cur and s == "<TAG_END>":
            block_ranges[cur][1] = i
            cur = None
        elif cur:
            block_ranges[cur][1] = i

    if not block_ranges:
        return

    n_steps = len(attn_records)
    fig, ax = plt.subplots(figsize=(12, 5))

    block_palette = {
        "<PAST_M>":   "#2980b9",
        "<CONST_M>":  "#27ae60",
        "<FUTURE_M>": "#8e44ad",
    }
    for block_name, (start, end) in block_ranges.items():
        if end <= start:
            continue
        mean_attn = heat[:, start:end+1].mean(axis=1)   # [n_steps]
        steps = [r["step"] for r in attn_records]
        color = block_palette.get(block_name, "gray")
        ax.plot(steps, mean_attn, label=block_name, color=color, lw=1.5)

    ax.set_xlabel("generation step")
    ax.set_ylabel("mean attention weight")
    ax.set_title("Mean attention to each block over generation steps")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "block_attention.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ブロック別注目グラフ保存: {out_path}")


def _plot_transition_attention(tokenizer, full_seq, syms, attn_records,
                                prompt_len, out_dir, vis_layer="mean", vis_head="mean",
                                context_window=50):
    """
    各ブロック生成開始時（最初 DENSE_TRANSITION_STEPS ステップ）の attention を拡大表示。

    X 軸: ブロックマーカー前 context_window トークン + マーカー以降のブロック先頭部分
    Y 軸: そのブロックの最初 DENSE_TRANSITION_STEPS 生成ステップ

    これにより「CONSTの最初のトークンがPASTの末尾に注意を向けているか」を直接確認できる。
    """
    block_markers = {"<PAST_M>": "PAST", "<CONST_M>": "CONST", "<FUTURE_M>": "FUTURE"}
    block_palette = {"PAST": "#2980b9", "CONST": "#27ae60", "FUTURE": "#8e44ad"}

    # 生成順にブロック開始位置（full_seq 上のトークン位置 & 生成ステップ）を検出
    block_starts = []
    for i, s in enumerate(syms):
        if s in block_markers:
            gen_step = i - prompt_len
            block_starts.append((block_markers[s], i, gen_step))

    if not block_starts:
        return

    # step -> record のルックアップ
    step_to_rec = {rec["step"]: rec for rec in attn_records}

    for bname, b_token_pos, b_gen_step in block_starts:
        # このブロックの最初 DENSE_TRANSITION_STEPS ステップのレコードを取得
        trans_recs = [
            rec for rec in attn_records
            if b_gen_step <= rec["step"] < b_gen_step + DENSE_TRANSITION_STEPS
            and rec.get("is_transition", False)
        ]
        if not trans_recs:
            # フォールバック: is_transition フラグなしで範囲検索
            trans_recs = [
                rec for rec in attn_records
                if b_gen_step <= rec["step"] < b_gen_step + DENSE_TRANSITION_STEPS
            ]
        if not trans_recs:
            print(f"  [{bname}] 遷移レコードなし (gen_step={b_gen_step})")
            continue

        # X 軸範囲: ブロックマーカー直前 context_window + マーカー以降
        x_start = max(0, b_token_pos - context_window)
        # 最後の trans_rec の seq_len まで見せる (その時点で参照できた全トークン)
        last_step = trans_recs[-1]["step"]
        x_end = min(len(syms), prompt_len + last_step + 2)
        x_range = list(range(x_start, x_end))
        n_x = len(x_range)
        n_rec = len(trans_recs)

        # attention weight を [n_rec, n_x] に整形
        heat = np.zeros((n_rec, n_x), dtype=np.float32)
        for ri, rec in enumerate(trans_recs):
            w = rec["weights"]  # [L, H, S]
            if vis_layer == "mean":
                w_l = w.mean(axis=0)
            else:
                w_l = w[int(vis_layer)]
            if vis_head == "mean":
                w_h = w_l.mean(axis=0)  # [S]
            else:
                w_h = w_l[int(vis_head)]
            S = len(w_h)
            for ci, xi in enumerate(x_range):
                if xi < S:
                    heat[ri, ci] = float(w_h[xi])

        # ---- 描画 ----
        fig_w = max(14, n_x * 0.22)
        fig_h = max(4, n_rec * 0.28)
        fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h + 2.5),
                                  gridspec_kw={"height_ratios": [0.6, fig_h]})

        # 上段: トークン色帯
        ax_t = axes[0]
        for ci, xi in enumerate(x_range):
            col = _block_color(syms[xi]) if xi < len(syms) else "#ecf0f1"
            ax_t.add_patch(mpatches.Rectangle((ci, 0), 1, 1, color=col, lw=0))
        marker_ci = b_token_pos - x_start
        ax_t.axvline(marker_ci + 0.5, color="red", lw=2, label=f"<{bname}_M>")
        ax_t.set_xlim(0, n_x)
        ax_t.set_ylim(0, 1)
        ax_t.set_xticks(np.arange(n_x) + 0.5)
        xlabels = [_abbrev(syms[xi]) if xi < len(syms) else "" for xi in x_range]
        ax_t.set_xticklabels(xlabels, rotation=90, fontsize=6)
        ax_t.set_yticks([])
        ax_t.set_title(
            f"<{bname}_M> ブロック開始時の token-level attention  "
            f"(赤線 = <{bname}_M> マーカー位置, x軸左={x_start}～右={x_end-1})",
            fontsize=9)
        legend_patches = [
            mpatches.Patch(color=block_palette.get(k, "gray"), label=f"<{k}_M>")
            for k in ("PAST", "CONST", "FUTURE")
        ]
        ax_t.legend(handles=legend_patches, loc="upper left", fontsize=7, framealpha=0.8)

        # 下段: ヒートマップ
        ax_h = axes[1]
        vmax = heat.max() if heat.max() > 1e-8 else 1e-6
        im = ax_h.imshow(heat, aspect="auto", cmap="hot",
                          interpolation="nearest", origin="upper",
                          vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax_h, shrink=0.6, label="attention weight")

        step_labels = [str(rec["step"]) for rec in trans_recs]
        ax_h.set_yticks(range(n_rec))
        ax_h.set_yticklabels(step_labels, fontsize=7)
        ax_h.set_ylabel("generation step")
        ax_h.set_xticks(np.arange(n_x))
        ax_h.set_xticklabels(xlabels, rotation=90, fontsize=6)
        ax_h.set_xlabel("token position in full sequence")
        ax_h.axvline(marker_ci + 0.5, color="cyan", lw=2,
                     label=f"<{bname}_M> marker")
        ax_h.legend(fontsize=7, loc="upper left")

        plt.tight_layout()
        out_path = os.path.join(out_dir, f"transition_{bname}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  遷移 attention 保存: {out_path}  "
              f"({n_rec} steps, token {x_start}~{x_end-1})")


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=== モデルロード ===")
    tokenizer = Tokenizer(get_token_converter_pro(TO_TOKEN))
    progress  = _DefaultLearningProgress()
    args      = MORTMArgs(MODEL_ARGS_PATH)

    # 観察用 Attention を有効化
    args.debug_attention = True

    model = MORTM(args=args, progress=progress)
    model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=progress.get_device()))
    model.to(progress.get_device())
    print(f"  {MODEL_WEIGHT}")
    print(f"  debug_attention = {args.debug_attention}")

    prompt = np.array([tokenizer.get("<EOS>")], dtype=int)

    print(f"\n=== 生成 + Attention 収集 (stride={STEP_STRIDE}) ===")
    generated, attn_records = generate_with_attention(
        model, tokenizer, prompt,
        temperature=TEMPERATURE, p=TOP_P,
        max_steps=MAX_GEN_STEPS, stride=STEP_STRIDE
    )

    print(f"\n  生成トークン数: {len(generated)}")
    print(f"  収集 Attention レコード数: {len(attn_records)}")

    print("\n=== 可視化 ===")
    visualize(tokenizer, prompt, generated, attn_records, OUT_DIR,
              vis_layer=VIS_LAYER, vis_head=VIS_HEAD)


if __name__ == "__main__":
    main()
