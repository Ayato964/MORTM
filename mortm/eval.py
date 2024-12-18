import torch
from torch.nn import Embedding
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
from pretty_midi import PrettyMIDI, Instrument, Note
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
from tqdm import tqdm
from numpy import ndarray

import mido
import matplotlib.pyplot as plt
from .mortm import MORTM
from .tokenizer import Tokenizer
from abc import abstractmethod


class AbstractEval:
    def __init__(self, mortm:MORTM):
        self.mortm = mortm

    @abstractmethod
    def view(self):
        pass


class EvalEmbedding(AbstractEval):
    def __init__(self, mortm: MORTM, tokenizer: Tokenizer):
        super().__init__(mortm)
        self.emb: Embedding = mortm.embedding
        self.emb.eval()
        self.tokenizer = tokenizer

    def view(self, note:ndarray=None):
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