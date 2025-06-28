import numpy
import torch
from torch import Tensor
import torch.nn as nn
from typing import Tuple, List
from einops import rearrange
import loralib.layers as lora

from .modules.progress import LearningProgress
from .modules.config import MORTMArgs
from .modules.layers import MORTMDecoder
from flash_attn.bert_padding import pad_input, unpad_input

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
        self.use_lora = args.use_lora

        self.decoder = MORTMDecoder(args, progress=progress)

        print(f"Input Vocab Size:{args.vocab_size}")
        self.embedding: nn.Embedding = nn.Embedding(args.vocab_size, self.d_model, padding_idx=0).to(self.progress.get_device())
        if not self.use_lora:
            self.Wout: nn.Linear = nn.Linear(self.d_model, args.vocab_size).to(self.progress.get_device())
        else:
            self.Wout: lora.Linear = lora.Linear(self.d_model, args.vocab_size, r=args.lora_r, lora_alpha=args.lora_alpha)

        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, x, padding_mask=None, is_causal=False):
        x: Tensor = self.embedding(x).to(dtype=torch.bfloat16)
        if padding_mask is not None:
            batch, tgt_len, embed_dim = x.size()
            x, indices, cu_seqlens, max_s, used_seqlens = unpad_input(x, padding_mask)
        else:
            tgt_len, embed_dim = x.size()
            batch = None
            indices = cu_seqlens = max_s = used_seqlens = None
        out = self.decoder(tgt=x, tgt_is_causal=is_causal, cu_seqlens=cu_seqlens, max_seqlen=max_s)
        if padding_mask is not None:
            out = pad_input(out, indices, batch, tgt_len)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            score: Tensor = self.Wout(out)
        return score

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
        self.eval()
        if isinstance(src, numpy.ndarray):
            src = torch.tensor(src, device=self.progress.get_device())
        #src = src.unsqueeze(0)
        #src_mask = _generate_square_subsequent_mask(src.size(1)).to(self.progress.get_device())
        #src_key_padding_mask = torch.zeros(src.size(0), src.size(1), dtype=torch.bool).to(self.progress.get_device())

        generated_tokens = []
        is_running = True
        with torch.no_grad():
            while is_running:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16) :
                    logits: Tensor = self(src, is_causal=True)
                sampled_index = self.top_p_sampling(logits[-1], p=p, temperature=temperature)
                generated_tokens.append(sampled_index)
                print(sampled_index)

                src = torch.cat([src, torch.tensor([sampled_index], device=self.progress.get_device())], dim=0)
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

