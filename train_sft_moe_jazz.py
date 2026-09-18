"""後方互換ラッパー: 汎用スクリプト train_sft_moe.py を --dataset jazz で実行します。"""
import sys
from train_sft_moe import main

if __name__ == "__main__":
    if "--dataset" not in sys.argv:
        sys.argv.extend(["--dataset", "jazz"])
    main()
