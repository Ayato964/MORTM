import math

from torch.nn import Embedding
import torch
from torch import Tensor
from pretty_midi import PrettyMIDI
import umap
import plotly.express as px
import pandas as pd
from numpy import ndarray
import numpy as np
import matplotlib.pyplot as plt
from mortm.models.mortm import MORTM
from .tokenizer import Tokenizer
from abc import abstractmethod


class AbstractEval:
    def __init__(self, mortm:MORTM):
        self.mortm = mortm

    @abstractmethod
    def view(self, *args, **kwargs):
        pass


class EvalEmbedding(AbstractEval):
    def __init__(self, mortm: MORTM, tokenizer: Tokenizer):
        super().__init__(mortm)
        self.emb: Embedding = mortm.embedding
        self.emb.eval()
        self.tokenizer = tokenizer

    def view(self, note:ndarray | None = None):
        if note is None:
            emb_list:Tensor = self.get_embedding_list().squeeze(0).to("cpu")
            emb_list: ndarray = emb_list.detach().numpy()
        else:
            tensor_note:Tensor = torch.tensor(note).unsqueeze(0).to(self.mortm.progress.get_device())
            emb = self.emb(tensor_note).squeeze(0).to("cpu")
            emb_list: ndarray = emb.detach().numpy()

        umap_reducer = umap.UMAP(
            n_components=3,         # 3次元に削減
            n_neighbors=15,         # 近傍数（調整可能）
            min_dist=0.1,            # 最小距離（調整可能）
            metric='cosine',         # 距離計算の指標
            random_state=42
        )
        # 次元削減の実行
        embeddings_3d = umap_reducer.fit_transform(emb_list)
        print(f"UMAP Reduced Embeddings Shape: {embeddings_3d.shape}")
        vocab = self.get_vocab_list()
                # データフレームの作成
        df = pd.DataFrame({
            'UMAP1': embeddings_3d[:, 0],
            'UMAP2': embeddings_3d[:, 1],
            'UMAP3': embeddings_3d[:, 2],
            'Token': vocab
        })

        # Plotlyの3D散布図
        fig = px.scatter_3d(
            df,
            x='UMAP1',
            y='UMAP2',
            z='UMAP3',
            text='Token',
            hover_data=['Token'],
            title='UMAP Projection of Transformer Vocabulary Embeddings (3D)',
            width=1200,
            height=800
        )

        fig.update_traces(marker=dict(size=3, opacity=0.7))
        fig.show()

        pass

    def get_vocab_list(self):
        vocab = []
        for i in range(len(self.tokenizer.tokens)):
            vocab.append(self.tokenizer.rev_get(i))
        return vocab

    def get_embedding_list(self):
        tokens = torch.tensor([i for i in range(len(self.tokenizer.tokens))]).unsqueeze(0).to(self.mortm.progress.get_device())
        return self.emb(tokens)

class EvalSoftMaxScale(AbstractEval):

    def __init__(self, mortm: MORTM, tokenizer: Tokenizer):
        super().__init__(mortm)
        self.tokenizer = tokenizer
        self.progress = mortm.progress

    def view(self, input_seq):
        # _get_softmax_matrix により softmax の出力リストを取得する
        sm: list = self._get_softmax_matrix(input_seq)

        num_plots = len(sm)
        top_k = 10  # 各分布で上位10件を表示

        # サブプロットを自動で配置するため、行数と列数を決定します
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        axes = np.array(axes).reshape(-1)  # サブプロットの配列を1次元に変換

        for i, distribution in enumerate(sm):
            # distribution が PyTorch の Tensor なら、NumPy array に変換
            if hasattr(distribution, 'detach'):
                distribution = distribution.detach().cpu().numpy()

            # 降順ソートして上位 top_k 件取得
            sorted_indices = np.argsort(distribution)[::-1]
            sorted_probs = distribution[sorted_indices]

            top_indices = sorted_indices[:top_k]
            top_probs = sorted_probs[:top_k]

            # トークンID を対応するトークン名に変換
            # 例: 135 -> "p_65" （self.tokenizer.rev_get(135) が "p_65" を返す）
            token_names = [self.tokenizer.rev_get(tok_id) for tok_id in top_indices]

            # 棒グラフを描画
            ax = axes[i]
            ax.bar(range(top_k), top_probs, tick_label=token_names)
            ax.set_xlabel("Token")
            ax.set_ylabel("Probability")
            ax.set_title(f"Time step {i}")
            ax.tick_params(axis='x', rotation=45)

        # 余ったサブプロットがあれば削除
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def _get_softmax_matrix(self, input_seq) -> list:
        self.mortm.eval()
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.long, device=self.progress.get_device())
        sm = []
        max_measure = 3
        seg: Tensor = self.split_tensor_at_value(input_seq, 3, include_split=True)
        tgt = torch.tensor([2], dtype=torch.long, device=self.progress.get_device())
        tgt = torch.concatenate((tgt, seg[-1])).to(self.progress.get_device())
        point = 0 if len(seg[:-1]) - 8 <= 0 else len(seg[:-1]) - 8

        src = torch.tensor([], dtype=torch.long, device=self.progress.get_device())

        for i in range(point, len(seg[point:-1])):
            src = torch.concatenate((src, seg[i]))
        generated = src.clone()

        for i in range(max_measure):
            while not (tgt[-1] == 391 or tgt[-1] == 392):
                logit = self.mortm(src=src.unsqueeze(0), tgt=tgt.unsqueeze(0))
                outputs = logit.view(-1, logit.size(-1)).to(self.mortm.progress.get_device())
                if tgt[-1] in range(5, 69):
                    sm.append(self.mortm.softmax(outputs[-1]))
                token = self.mortm.top_p_sampling(outputs[-1], p=0.95, temperature=1.0)
                tgt = torch.concatenate((tgt, torch.tensor([token], dtype=torch.long,
                                                           device=self.progress.get_device())), dim=0)

            if tgt[-1] == 392:
                break
            generated = torch.concatenate((generated, tgt[1: -1]))
            src = torch.concatenate((src, tgt[1:-1]))
            tgt = torch.tensor([2], device=self.progress.get_device())
            seg = self.split_tensor_at_value(src, 3, include_split=True)
            if len(seg) > 8:
                src = torch.tensor([], dtype=torch.long, device=self.progress.get_device())
                for i in seg[1:]:
                    src = torch.concatenate((src, i))
        return sm

    def split_tensor_at_value(self, tensor: Tensor, split_value, include_split=True):
        """
        指定した値を基準にテンソルを分割します。

        Args:
            tensor (torch.Tensor): 1次元のテンソルを想定しています。
            split_value (int or float): 分割の基準となる値。
            include_split (bool, optional): 分割値を各セグメントに含めるかどうか。デフォルトは True。

        Returns:
            List[torch.Tensor]: 分割されたテンソルのリスト。
        """
        if tensor.dim() != 1:
            raise ValueError("この関数は1次元のテンソルに対してのみ動作します。")

        # 分割値が存在するインデックスを取得
        split_indices = (tensor == split_value).nonzero(as_tuple=True)[0]

        if len(split_indices) == 0:
            # 分割値が見つからない場合、元のテンソルをそのまま返す
            return [tensor]

        segments = []
        num_splits = len(split_indices)

        for i in range(num_splits):
            start = split_indices[i]
            if include_split:
                start = start  # 分割値を含める場合
            else:
                start = split_indices[i] + 1  # 分割値を含めない場合

            if i + 1 < num_splits:
                end = split_indices[i + 1]
            else:
                end = len(tensor)

            if include_split:
                end = end  # 次の分割値の位置まで含める
            else:
                end = end  # 次の分割値の位置まで含めない

            segment = tensor[start:end]
            segments.append(segment)

        return segments




class EvalPianoRoll(AbstractEval):
    def __init__(self, mortm: MORTM, midi_data: str):
        super().__init__(mortm)
        self.midi_data = PrettyMIDI(midi_data)

    def view(self):
        # MIDIファイルを読み込む
        midi_data = self.midi_data
        # 音名（ピッチクラス）のカウントを初期化
        pitch_class_count = {pitch: 0 for pitch in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']}

        # 各ノートのピッチクラスをカウント
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                pitch_class = self.midi_note_to_pitch_class(note.pitch)
                pitch_class_count[pitch_class] += 1

        # ヒストグラムの描画
        labels = list(pitch_class_count.keys())
        frequencies = list(pitch_class_count.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels, frequencies, color='skyblue')
        plt.title("Pitch Class Histogram (Including Sharps and Flats)")
        plt.xlabel("Pitch Class")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()


    def midi_note_to_pitch_class(self, note):
        """MIDIノート番号を音名に変換"""
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return pitch_classes[note % 12]