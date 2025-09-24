import numpy
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical
import torch.nn as nn
from typing import Tuple, List
from einops import rearrange
import loralib.layers as lora

from .modules.progress import LearningProgress
from .modules.config import MORTMArgs
from .modules.layers import MORTMDecoder
from flash_attn.bert_padding import pad_input, unpad_input

from ..train.tokenizer import Tokenizer


class MORTM(nn.Module):
    """
    Main class for the MORTM model.

    Attributes:
        args (MORTMArgs): Model configuration arguments.
        progress (LearningProgress): Progress tracker.
        embedding (nn.Embedding): Embedding layer.
        Wout (nn.Linear or lora.Linear): Output layer.
        softmax (nn.Softmax): Softmax layer.
    """
    def __init__(self, args: MORTMArgs, progress: LearningProgress):
        super(MORTM, self).__init__()
        self.args = args
        self.progress = progress
        self.e_layer = args.e_layer
        self.d_layer = args.d_layer
        self.num_heads = args.num_heads
        self.d_model = args.d_model
        self.dim_feedforward = args.dim_feedforward
        self.dropout = args.dropout
        self.use_lora = args.use_lora
        print("Use LoRA:", self.use_lora)

        self.decoder = MORTMDecoder(args, progress=progress)

        print(f"Input Vocab Size:{args.vocab_size}")
        self.embedding: nn.Embedding = nn.Embedding(args.vocab_size, self.d_model, padding_idx=0).to(self.progress.get_device())
        if not self.use_lora:
            self.Wout: nn.Linear = nn.Linear(self.d_model, args.vocab_size).to(self.progress.get_device())
        else:
            self.Wout: lora.Linear = lora.Linear(self.d_model, args.vocab_size, r=args.lora_r, lora_alpha=args.lora_alpha)

        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())

    def forward(self, x, padding_mask=None, is_causal=False, is_save_cache=False):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor.
            padding_mask (Tensor, optional): Padding mask tensor.
            is_causal (bool, optional): Whether to use causal masking.
            is_save_cache (bool, optional): Whether to save KV cache.

        Returns:
            Tensor: Output logits.
        """
        x: Tensor = self.embedding(x).to(dtype=torch.bfloat16)
        if padding_mask is not None:
            batch, tgt_len, embed_dim = x.size()
            x, indices, cu_seqlens, max_s, used_seqlens = unpad_input(x, padding_mask)
        else:
            tgt_len, embed_dim = x.size()
            batch = None
            indices = cu_seqlens = max_s = used_seqlens = None
        out = self.decoder(tgt=x, tgt_is_causal=is_causal, cu_seqlens=cu_seqlens, max_seqlen=max_s,
                           batch_size=batch, indices=indices, is_save_cache=is_save_cache)
        if padding_mask is not None:
            out = pad_input(out, indices, batch, tgt_len)

        with torch.autocast(device_type="cuda", dtype=torch.float32):
            score: Tensor = self.Wout(out)
        return score

    @torch.inference_mode()
    def top_sampling_measure_kv_cache(self, tokenizer: Tokenizer, src: Tensor | numpy.ndarray, p=0.9, temperature=1.0, print_log=True):
        """
        Generates tokens using top-p sampling with KV cache.

        Args:
            src (Tensor | numpy.ndarray): Input sequence.
            p (float, optional): Top-p sampling threshold.
            temperature (float, optional): Sampling temperature.
            print_log (bool, optional): Whether to print logs.

        Returns:
            Tuple[List[numpy.ndarray], Tuple[List[numpy.ndarray], List[numpy.ndarray]]]: Generated tokens and split sequences.
            :param end_tokens:
        """
        self.eval()
        is_running = True
        end_count = 0
        device = self.progress.get_device()

        if isinstance(src, numpy.ndarray):
            src = torch.tensor(src, device=device)
        if src.dim() == 1:
            src = src.unsqueeze(0)


        # --- 1. プロンプト処理 (Pre-fill) ---
        if print_log: print("--- Pre-fill Phase ---")
        prompt_padding_mask = (src != self.embedding.padding_idx)
        print(prompt_padding_mask, src)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.forward(src, padding_mask=prompt_padding_mask, is_causal=True, is_save_cache=True)

        last_token_logits = logits[:, -1, :]
        # next_tokens は (batch_size,) の形状を持つテンソル
        next_tokens = self.top_p_sampling(last_token_logits, p=p, temperature=temperature)

        # 全トークンを保持するテンソル
        all_tokens = torch.cat([src, next_tokens.unsqueeze(1)], dim=1)
        
        # --- 2. トークン生成 (Decoding) ---
        i = len(all_tokens)
        if print_log: print("\n--- Decoding Phase ---")
        while is_running:
            # 入力は直前に生成されたトークン (B, 1)
            input_tokens = next_tokens
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.forward(input_tokens, padding_mask=None, is_causal=True, is_save_cache=True)

            # next_tokens は (batch_size,) の形状を持つテンソル
            next_tokens = self.top_p_sampling(logits.squeeze(1), p=p, temperature=temperature)

            #print(next_tokens.max(), all_tokens.max())
            # 生成されたトークンを連結
            all_tokens = torch.cat([all_tokens, next_tokens.unsqueeze(1)], dim=1)
        
            if print_log: print(f"\r Step {i+1}: Generated tokens {next_tokens.tolist()}", end="")

            if self.is_end_point(all_tokens, (tokenizer.get("<ESEQ>"), tokenizer.get("<TE>"))) or i > self.args.position_length:
                is_running = False

            i += 1

        np_all_tokens = []
        prompt = []
        generated = []
        print(all_tokens.max())
        for seq in all_tokens:
            seq: Tensor
            np_seq = np.array([], dtype=int)
            pad = (seq == 0).nonzero(as_tuple=True)[0]
            eseq = (seq == tokenizer.get("<ESEQ>")).nonzero(as_tuple=True)[0]
            if len(eseq) == 0:
                eseq = len(seq)-1
            elif len(eseq) != 1:
                eseq = eseq[0].item()
            else:
                eseq = eseq.item()

            if len(pad) != 0:
                start = pad[0]
                end = pad[-1]
                prompt.append(seq[:start].cpu().numpy())
                np_seq = np.append(np_seq, seq[:start].cpu().numpy())
                if eseq == len(seq):
                    generated.append(seq[end+1:].cpu().numpy())
                    np_seq = np.append(np_seq, seq[end+1:].cpu().numpy())
                else:
                    generated.append(seq[end+1:eseq+1].cpu().numpy())
                    np_seq = np.append(np_seq, seq[end+1:eseq+1].cpu().numpy())
            else:
                gen_id = ((seq == tokenizer.get("<MGEN>")) | (seq == tokenizer.get("<CGEN>"))).nonzero(as_tuple=True)[0]
                if eseq == len(seq):
                    generated.append(seq[gen_id+1:].cpu().numpy())
                    np_seq = np.append(np_seq, seq.cpu().numpy())
                else:
                    generated.append(seq[gen_id:eseq+1].cpu().numpy())
                    np_seq = np.append(np_seq,seq[:eseq+1].cpu().numpy())
            np_all_tokens.append(np_seq)
            if np_seq.max() > self.args.vocab_size:
                raise ValueError(
                    f"生成されたトークンIDが語彙サイズ({self.args.vocab_size})を超えています: {np_seq.max()}"
                )

        return np_all_tokens, (prompt, generated)

    def is_end_point(self, x: torch.Tensor, end_tokens) -> bool:
        """
        Check if all rows in the tensor contain at least one of the end tokens.

        Args:
            x (torch.Tensor): Tensor of shape [n, 14].

        Returns:
            bool: True if all rows contain at least one end token, False otherwise.
        """
        mask = (x <= -1)
        for e in end_tokens:
            mask = mask | (x == e)
        per_row_has5 = mask.any(dim=1)
        # 3) 全行が True かを判定する
        all_rows_ok = per_row_has5.all()

        # 4) Python の bool 型で返す
        return bool(all_rows_ok)

    def top_p_sampling(self, logits: Tensor, p=0.9, temperature=1.0) -> Tensor:
        """
        Perform top-p sampling for multiple batches.

        Args:
            logits (Tensor): Logits tensor.
            p (float, optional): Top-p sampling threshold.
            temperature (float, optional): Sampling temperature.

        Returns:
            Tensor: Sampled token indices.
        """
        logits = logits / temperature
        probs = self.softmax(logits)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_probs_to_remove = cumulative_probs > p
        sorted_probs_to_remove[..., 1:] = sorted_probs_to_remove[..., :-1].clone()
        sorted_probs_to_remove[..., 0] = 0

        probs_to_keep = sorted_probs.masked_fill(sorted_probs_to_remove, 0)

        # ゼロ除算を避けるため、分母に微小な値を加える
        probs_sum = probs_to_keep.sum(dim=-1, keepdim=True)
        renormalized_probs = probs_to_keep / (probs_sum + 1e-9) #

        sampled_next_indices = torch.multinomial(renormalized_probs, num_samples=1)
        sampled_original_indices = torch.gather(sorted_indices, dim=-1, index=sampled_next_indices)

        r = sampled_original_indices.squeeze(-1)

        # vocab_sizeを直接取得してチェックする
        vocab_size = logits.shape[-1]
        if r.max().item() > vocab_size:
            raise ValueError(
                f"サンプリングされたトークンIDが語彙サイズ({vocab_size})以上です: {r.max().item()}"
            )

        return r


    def split_tensor_at_value(self, tensor: Tensor, split_value, include_split=True):
        """
        Split a 1D tensor at specified value.

        Args:
            tensor (torch.Tensor): 1D tensor to split.
            split_value (int or float): Value to split at.
            include_split (bool, optional): Whether to include the split value in segments.

        Returns:
            List[torch.Tensor]: List of split tensor segments.
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

    def get_log_probs(self, sequence_tensors: torch.Tensor, padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Calculate log probabilities for a given sequence.

        Args:
            sequence_tensors (torch.Tensor): Token ID tensor of shape (batch, seq_len).
            padding_mask (torch.Tensor): Corresponding attention mask.

        Returns:
            torch.Tensor: Log probabilities tensor for each token position.
        """
        # 1. 入力とターゲットを作成（1つずらす）
        input_ids = sequence_tensors[:, :-1]
        target_ids = sequence_tensors[:, 1:]

        # マスクも同様にずらす
        if padding_mask is not None:
            input_mask = padding_mask[:, :-1]
        else:
            input_mask = None

        # 2. モデルに`input_ids`を通して、各ステップのlogitsを取得
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = self.forward(input_ids, padding_mask=input_mask, is_causal=True)

        reshaped_logits = logits.view(-1, self.args.vocab_size)
        reshaped_targets = target_ids.reshape(-1).long()
        # Categorical分布を使って一括で計算
        dist = Categorical(logits=reshaped_logits)
        log_probs = dist.log_prob(reshaped_targets)

        # 元の形状 (batch, seq_len-1) に戻す
        log_probs = log_probs.view(logits.size(0), logits.size(1))

        # パディング部分のlog_probを0にする
        if padding_mask is not None:
            log_probs = log_probs * padding_mask[:, 1:]

        return log_probs
