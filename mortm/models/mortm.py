import numpy
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, List

from .modules.PositionalEncoding import PositionalEncoding
from .modules.progress import LearningProgress
from .modules.config import MORTMArgs
from .modules.layers import MORTMDecoder, MORTMEncoder

class MORTM(nn.Module):
    def __init__(self, args: MORTMArgs, progress: LearningProgress):
        super(MORTM, self).__init__()
        self.progress = progress
        self.e_layer = args.e_layer
        self.d_layer = args.d_layer
        self.num_heads = args.num_heads
        self.d_model = args.d_model
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout

        self.decoder = MORTMDecoder(args,
                               batch_first=True, bias=True,
                               layer_norm_eps=1e-5, progress=progress)

        print(f"Input Vocab Size:{args.vocab_size}")
        self.Wout: nn.Linear = nn.Linear(self.d_model, args.vocab_size).to(self.progress.get_device())
        self.embedding: nn.Embedding = nn.Embedding(args.vocab_size, self.d_model, padding_idx=0).to(self.progress.get_device())
        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, input_padding_mask=None,
                tgt_padding_mask=None, src_is_causal=False, tgt_is_causal=False):
        src_e: Tensor = self.embedding(src)

        out = self.decoder(tgt=src_e, memory=None, tgt_mask=tgt_mask,
                           memory_key_padding_mask=input_padding_mask,
                           tgt_key_padding_mask=input_padding_mask, memory_is_causal=src_is_causal, tgt_is_causal=src_is_causal)

        score: Tensor = self.Wout(out)
        return score.to(self.progress.get_device())

    def top_p_sampling_measure(self, src: Tensor, p=0.9, max_measure=20, temperature=1.0) -> Tuple[Tensor, Tensor]:
        """
        トークンを生成するためのメソッドです。

        Args:
            src (Tensor): 入力テンソル
            p (float): 確率の閾値
            max_measure (int): 最大生成長
            temperature (float): 温度パラメータ

        Returns:
            List[Tensor]: 生成されたトークンのリスト
        """
        if isinstance(src, numpy.ndarray):
            src = torch.tensor(src, device=self.progress.get_device())
        src = src.unsqueeze(0)
        #src_mask = _generate_square_subsequent_mask(src.size(1)).to(self.progress.get_device())
        #src_key_padding_mask = torch.zeros(src.size(0), src.size(1), dtype=torch.bool).to(self.progress.get_device())

        generated_tokens = []
        is_running = True
        while is_running:
            logits: Tensor = self(src, src_is_causal=True)
            logits = logits.squeeze(0)
            sampled_index = self.top_p_sampling(logits[-1], p=p, temperature=temperature)
            generated_tokens.append(sampled_index)
            src = torch.cat([src, torch.tensor([[sampled_index]], device=self.progress.get_device())], dim=1)
            measure_count = (src == 8).sum().item()
            if sampled_index == 585 or sampled_index == 586 or measure_count > max_measure:
                is_running = False

        return torch.tensor(generated_tokens), src.squeeze(0)


    def top_p_sampling(self, logits, p=0.9, temperature=1.0) -> int:

        logits = logits / temperature
        # logitsをソフトマックスで確率分布に変換
        probs = self.softmax(logits)
        # 確率の降順に並べ替え、そのインデックスを取得
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)

        # 累積確率を計算
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 累積確率がpを超えるインデックスを取得
        cutoff_index = torch.where(cumulative_probs > p)[0][0]

        # 上位pに入らないトークンの確率を0にする
        sorted_probs[cutoff_index + 1:] = 0

        # 確率を再正規化
        sorted_probs /= torch.sum(sorted_probs)

        # トークンをサンプリング
        sampled_index = torch.multinomial(sorted_probs, 1)

        # インデックスを元の順序に戻す
        return sorted_indices[sampled_index].item()

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

