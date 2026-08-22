"""CARL が保存した gem チェックポイントを「再マージされない形」へ直す。

背景:
  `CARLDriver._save` は gem が `eval()`(= loralib がマージ済み)の状態で
  `state_dict()` を取っていたため、保存ファイルの `weight` には既に
  `B @ A * scaling` が焼き込まれている。これを通常どおり
  `load_state_dict` -> `eval()` の順で読むと **2回目のマージ**が走り、
  デルタが 2 倍に膨らんだ別のモデルになってしまう
  (実測: 小節数遵守 100% -> 17.7%、空生成 0% -> 74%)。

方針:
  重み側は学習結果そのものなので触らず、`lora_B` をゼロで置き換える。
  デルタは `B @ A` なので B=0 ならマージしても何も足されない。
  結果として **config の lora_alpha が何であっても** 学習時の重みが
  そのまま再現される(alpha 取り違えの影響も同時に断てる)。

  LoRA は元々マージ状態では勾配が流れず一度も更新されていないため、
  ゼロ化で失われる学習成果は無い(SFT 由来の値は既に weight 側に入っている)。
"""

import argparse
import os

import torch


def unmergeable(sd: dict) -> dict:
    """`lora_B` をゼロ化した state_dict を返す。他のテンソルは共有のまま。"""
    return {k: (torch.zeros_like(v) if k.endswith("lora_B") else v)
            for k, v in sd.items()}


def fix_file(src: str, dst: str) -> dict:
    sd = torch.load(src, map_location="cpu")
    n_b = sum(1 for k in sd if k.endswith("lora_B"))
    n_nonzero = sum(1 for k in sd if k.endswith("lora_B") and float(sd[k].abs().sum()) > 0)
    torch.save(unmergeable(sd), dst)
    return {"src": src, "dst": dst, "lora_B": n_b, "zeroed": n_nonzero,
            "mb": os.path.getsize(dst) / 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="+", help="CARL が保存した gem の .pth")
    ap.add_argument("--suffix", default=".fixed", help="出力ファイル名に付ける接尾辞")
    args = ap.parse_args()

    for src in args.src:
        root, ext = os.path.splitext(src)
        st = fix_file(src, f"{root}{args.suffix}{ext}")
        print(f"[fix] {os.path.basename(st['src'])} -> {os.path.basename(st['dst'])}  "
              f"lora_B {st['zeroed']}/{st['lora_B']} 個をゼロ化  ({st['mb']:.1f}MB)")


if __name__ == "__main__":
    main()
