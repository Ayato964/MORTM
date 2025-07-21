import torch
import torch.nn.functional as F
import numpy as np
import os # ファイルパス操作のためにインポート

# mortmライブラリからのインポート（ユーザー提供のコードと同様）
from mortm.models.bertm import BERTM, MORTMArgs
from mortm.train.tokenizer import Tokenizer, get_token_converter, TO_MUSIC
from mortm.models.modules.progress import _DefaultLearningProgress
# これで scikit-learn を使って評価指標を計算できる
from sklearn.metrics import classification_report, roc_auc_score


# 1. モデルとトークナイザーの準備（ユーザー提供のコードと同様）
# ----------------------------------------------------------------
tokenizer: Tokenizer = Tokenizer(music_token=get_token_converter(TO_MUSIC))
tokenizer.rev_mode()
args = MORTMArgs("configs/models/bertm/class_file.json")
model = BERTM(progress=_DefaultLearningProgress(), args=args)
model.load_state_dict(torch.load("out/model/class/MORTM.4.0.1_0.07270082146024857.pth"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# ----------------------------------------------------------------

# 2. 評価対象のnpzファイルのパスリストを定義
# NOTE: このリストを実際のファイルパスに置き換えてください
with open("out/model/class/val_paths_4.0.1.json", "r") as f:
    import json
    npz_file_paths = json.load(f)[0]

# 3. モデルを評価モードに設定
model.eval()

# 4. 推論結果を保存するためのリストを初期化
predictions = []

# 5. ループ処理で各ファイルを推論
# torch.no_grad() と autocast をループの外で一度だけ呼び出すのが効率的
with torch.no_grad():
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for file_path in npz_file_paths:
            if not os.path.exists(file_path):
                print(f"Warning: File not found, skipping. -> {file_path}")
                continue
            # npzファイルをロード
            np_notes_list = np.load(file_path)
            for i in range(len(np_notes_list) - 1):
                # テンソルに変換し、バッチ次元を追加
                src = torch.tensor(np_notes_list[f'array{i+1}']).unsqueeze(0).to(device)
                prompt_padding_mask = (src != 0)

                # モデルで推論実行
                out = model(src, prompt_padding_mask)

                # スコア（AIである確率）を計算
                score = F.sigmoid(out).item() # .item()でPythonの数値に変換

                # 結果をリストに追加
                predictions.append({
                    "file_path": file_path,
                    "score": score
                })

# 6. すべての推論結果を表示
print("\n--- Inference Results ---")
for p in predictions:
    # 閾値0.5でAIか人間かを判定
    label = "AI" if p['score'] > 0.5 else "Human"
    print(f"File: {p['file_path']}, Score: {p['score']:.4f}, Prediction: {label}")


# 'predictions' リストがすでにあると仮定

true_labels = [] # 正解ラベル (0: Human, 1: AI)
pred_scores = [] # モデルの予測スコア

for p in predictions:
    # ファイルパスに 'ai' が含まれていれば正解ラベルを1とする
    label = 1 if 'ai' in p['file_path'] else 0
    true_labels.append(label)
    pred_scores.append(p['score'])


pred_labels = [1 if score > 0.5 else 0 for score in pred_scores]
print("\n--- Evaluation Metrics ---")
print(classification_report(true_labels, pred_labels, target_names=['Human', 'AI']))
print(f"AUC Score: {roc_auc_score(true_labels, pred_scores):.4f}")