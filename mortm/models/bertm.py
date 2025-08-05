import numpy as np
import torch
from torch.distributions import Categorical
from .modules.layers import *
from .modules.config import MORTMArgs

from flash_attn.bert_padding import unpad_input, pad_input



class ActorCritic(nn.Module):
    def __init__(self, args: MORTMArgs, progress):
        super(ActorCritic, self).__init__()
        self.args = args
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

        self.critic_hidden = nn.Linear(self.d_model, self.d_model // 2)
        self.critic_out = nn.Linear(self.d_model // 2, 1)  # 出力次元を1に設定

        self.softmax: nn.Softmax = nn.Softmax(dim=-1).to(self.progress.get_device())


    def evaluate_actions(self, sequence_tensors, padding_mask):
        # 1. 入力とターゲットを作成（1つずらす）
        input_ids = sequence_tensors[:, :-1]
        target_ids = sequence_tensors[:, 1:]

        # マスクも同様にずらす
        if padding_mask is not None:
            input_mask = padding_mask[:, :-1]
        else:
            input_mask = None

        # 2. モデルに`input_ids`を通して、各ステップのlogitsを取得
        logits, new_values = self.forward(input_ids, padding_mask=input_mask, is_causal=True)

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
        return log_probs, new_values.reshape(new_values.shape[0], new_values.shape[1])


    def eval_seq(self, src, print_log=True):
        """
        KVキャッシュを利用してトークンを生成するためのメソッドです。
        複数バッチに対応しています。
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
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, values = self.forward(src, padding_mask=prompt_padding_mask, is_causal=True, is_save_cache=True)
        prob_list = Categorical(logits=logits)

        last_token_logits = logits[:, -1, :]
        # next_tokens は (batch_size,) の形状を持つテンソル
        next_tokens = self.top_p_sampling(last_token_logits, p=0.95, temperature=1.0)

        # 全トークンを保持するテンソル
        all_tokens = torch.cat([src, next_tokens.unsqueeze(1)], dim=1)
        all_values = values
        all_probs = prob_list.log_prob(all_tokens[:, 1:])
        # --- 2. トークン生成 (Decoding) ---
        i = len(all_tokens)
        if print_log: print("\n--- Decoding Phase ---")
        while is_running:
            # 入力は直前に生成されたトークン (B, 1)
            input_tokens = next_tokens
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, values = self.forward(input_tokens, padding_mask=None, is_causal=True, is_save_cache=True)
            probs_list = Categorical(logits=logits.squeeze(1))
            # next_tokens は (batch_size,) の形状を持つテンソル
            next_tokens = self.top_p_sampling(logits.squeeze(1), p=0.95, temperature=1.0)

            #print(next_tokens.max(), all_tokens.max())
            # 生成されたトークンを連結
            all_tokens = torch.cat([all_tokens, next_tokens.unsqueeze(1)], dim=1)
            all_values = torch.cat([all_values, values.unsqueeze(1)], dim=1)
            all_probs = torch.cat([all_probs, probs_list.log_prob(next_tokens).unsqueeze(1)], dim=1)

            if print_log: print(f"\r Step {i+1}: Generated tokens {next_tokens.tolist()}", end="")

            if self.is_end_point(all_tokens) or i > self.args.position_length:
                is_running = False

            i += 1
        print(all_tokens.shape, all_values.shape, all_probs.shape)
        np_all_tokens = []
        np_all_values = []
        np_all_probs = []
        np_generated_only_tokens = []
        if print_log: print(all_tokens.max())
        for i, seq in enumerate(all_tokens):
            seq: Tensor
            np_seq = np.array([], dtype=int)
            np_value = np.array([], dtype=float)
            np_probs = np.array([], dtype=float)
            pad = (seq == 0).nonzero(as_tuple=True)[0]
            eseq = (seq == 585).nonzero(as_tuple=True)[0]

            # ... eseq の長さを決定するロジックは変更なし ...
            if len(eseq) == 0:
                eseq = len(seq)-1
            elif len(eseq) != 1:
                eseq = eseq[0].item()
            else:
                eseq = eseq.item()

            if len(pad) != 0:
                start = pad[0]
                end = pad[-1]

                # トークン部分は変更なし
                np_seq = np.append(np_seq, seq[:start].cpu().numpy())
                # --- 修正: スライスの終点を-1する ---
                np_value = all_values[i, :start-1].cpu().numpy()
                np_probs = all_probs[i, :start-1].cpu().numpy()

                if eseq == len(seq):
                    # トークン部分は変更なし
                    np_seq = np.append(np_seq, seq[end+1:].cpu().numpy())
                    # --- 修正: スライスの始点と終点を-1する ---
                    np_value = np.append(np_value, all_values[i, end:].cpu().numpy()) # eseqはlen(seq)-1なので-1不要
                    np_probs = np.append(np_probs, all_probs[i, end:].cpu().numpy())
                    if print_log:print(f"fdkso LEN : {np_seq.shape, np_value.shape, np_probs.shape}")

                else:
                    # トークン部分は変更なし
                    np_seq = np.append(np_seq, seq[end+1:eseq+1].cpu().numpy())
                    # --- 修正: スライスの始点と終点を-1する ---
                    np_value = np.append(np_value, all_values[i, end:eseq].cpu().numpy())
                    np_probs = np.append(np_probs, all_probs[i, end:eseq].cpu().numpy())
                    if print_log:print(f"!WDFG : {np_seq.shape, np_value.shape, np_probs.shape}")
            else: # パディングがない場合
                if eseq == len(seq):
                    # トークン部分は変更なし
                    np_seq = np.append(np_seq, seq.cpu().numpy())
                    # --- 修正: 全体をスライスするが、長さは元々1短いのでそのままでOK ---
                    np_value = np.append(np_value, all_values[i].cpu().numpy())
                    np_probs = np.append(np_probs, all_probs[i].cpu().numpy())
                    if print_log:print(f"ESEQ LEN : {np_seq.shape, np_value.shape, np_probs.shape}")
                else:
                    # トークン部分は変更なし
                    np_seq = np.append(np_seq, seq[:eseq+1].cpu().numpy())
                    # --- 修正: スライスの終点を-1する ---
                    np_value = np.append(np_value, all_values[i, :eseq].cpu().numpy())
                    np_probs = np.append(np_probs, all_probs[i, :eseq].cpu().numpy())
                    if print_log: print(f"Not ESEQ LEN : {np_seq.shape, np_value.shape, np_probs.shape}")

            np_all_tokens.append(np_seq)
            np_all_values.append(np_value)
            np_all_probs.append(np_probs)
            if np_seq.max() > self.args.vocab_size:
                raise ValueError(
                    f"生成されたトークンIDが語彙サイズ({self.args.vocab_size})を超えています: {np_seq.max()}"
                )

        return np_all_tokens, np_all_values, np_all_probs

    def forward(self, x, padding_mask=None, is_causal=False, is_save_cache=False):
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
            hidden = self.critic_hidden(out)
            hidden = F.relu(hidden)
            critic_score = self.critic_out(hidden)
        return score, critic_score

    def is_end_point(self, x: torch.Tensor) -> bool:
        """
        x: Tensor of shape [n, 14]
        戻り値: 全ての行に少なくとも1つ 5 があれば True、そうでなければ False
        """

        mask = (x == 585) | (x == 586)
        per_row_has5 = mask.any(dim=1)
        # 3) 全行が True かを判定する
        all_rows_ok = per_row_has5.all()

        # 4) Python の bool 型で返す
        return bool(all_rows_ok)

    def top_p_sampling(self, logits: Tensor, p=0.9, temperature=1.0) -> Tensor:
        """
        複数バッチに対応したTop-pサンプリング。（修正版）
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


class BERTM(nn.Module):

    def __init__(self, args: MORTMArgs, progress):
        super(BERTM, self).__init__()
        self.args = args # argsを保存しておくと便利
        self.embedding = nn.Embedding(args.vocab_size, args.d_model)
        self.decoder = MORTMDecoder(args=args,
                                    progress=progress)
        self.attn_pool = Pool(args)
        self.hidden = nn.Linear(args.d_model, args.d_model // 2)
        self.Wout = nn.Linear(args.d_model // 2, 1) # linear層の出力次元に合わせる

    def forward(self, x: Tensor, padding_mask=None):
        x: Tensor = self.embedding(x).to(dtype=torch.bfloat16)

        if padding_mask is not None:
            x, indices, cu_seqlens, max_s, used_seqlens = unpad_input(x, padding_mask)
        else:
            indices = cu_seqlens = max_s = used_seqlens = None

        out = self.decoder(tgt=x, tgt_is_causal=False, cu_seqlens=cu_seqlens, max_seqlen=max_s)

        out = self.attn_pool(out, cu_seqlens if cu_seqlens is not None else torch.tensor([0, len(x)], dtype=torch.int32, device=x.device))  # バッチサイズをcu_seqlensに設定

        out = self.hidden(out)
        hid = F.relu(out)
        score = self.Wout(hid)

        return score
