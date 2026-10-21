from torch import nn
from .modules.layers import *
from .modules.config import PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, RVQ_TOKEN_OFFSET
from .modules.attention import pad_input, unpad_input, _get_default_device
from vector_quantize_pytorch import GroupedResidualVQ

import torch.distributed as dist
from typing import Optional


class MidiVision(nn.Module):
    def __init__(self, args: MORTM5Args, sync_codebook: Optional[bool] = None):
        super().__init__()
        self.args = args
        self.encoder = ResNetEncoder(args)

        if sync_codebook is None:
            sync_codebook = dist.is_available() and dist.is_initialized()

        self.vq_groups = getattr(args, "vq_groups", 4)
        self.num_quantizers = getattr(args, "encoder_layer", 2)
        assert args.encoder_wout % self.vq_groups == 0, (
            f"encoder_wout ({args.encoder_wout}) must be divisible by vq_groups ({self.vq_groups})"
        )
        self.codebook_dim = args.encoder_wout // self.vq_groups
        self.codebook_size = getattr(args, "codebook_size", 1024)
        self.token_offset = getattr(args, "token_offset", RVQ_TOKEN_OFFSET)

        # 各グループ(codebook_dim次元)ごとに独立してLayerNorm(アフィンなし)を適用する。
        # elementwise_affine=False により量子化入力の分散を厳密に 1.0 に固定し、スケール膨張とCommit Loss肥大化を根絶する。
        self.pre_vq_norm = nn.LayerNorm(self.codebook_dim, elementwise_affine=False)

        self.rvq = GroupedResidualVQ(
            dim=args.encoder_wout,
            groups=self.vq_groups,
            codebook_dim=self.codebook_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            sync_codebook=sync_codebook,
            # 複数ビッグバッチを跨いでコードブックを安定保持するため長寿命EMA (decay=0.999) を使用
            threshold_ema_dead_code=0.05,
            # rotation_trick: 残差0時の勾配爆発を防ぐため Straight-Through を使用
            rotation_trick=False,
            kmeans_init=True,
            kmeans_iters=10,
            decay=0.999,
        )

        self.use_pre_vq_norm = getattr(args, "use_pre_vq_norm", True)
        self.decoder = ResNetDecoder(args)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if any(k.startswith("vq.") for k in state_dict.keys()):
            self.use_pre_vq_norm = False
            state_dict = {
                (k.replace("vq.", "rvq.", 1) if k.startswith("vq.") else k): v
                for k, v in state_dict.items()
            }
        elif "use_pre_vq_norm" not in self.__dict__:
            self.use_pre_vq_norm = True
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _encode_latent(self, x: Tensor) -> Tensor:
        z = self.encoder(x)
        if getattr(self, "use_pre_vq_norm", True):
            orig_shape = z.shape
            # グループ単位で正規化: [..., encoder_wout] -> [..., vq_groups, codebook_dim] -> [..., encoder_wout]
            return self.pre_vq_norm(
                z.reshape(*orig_shape[:-1], self.vq_groups, self.codebook_dim)
            ).reshape(orig_shape)
        return z

    def _format_rvq_tokens(self, tokens: Tensor, include_offset: bool = True) -> Tensor:
        # GroupedResidualVQ の tokens は [groups, B, (S,), num_quantizers] (各値 [0, codebook_size)) で返るため
        # レイヤー順 (Coarse-to-Fine) の [B, (S,), num_quantizers, groups] に並べ替え、
        # token_offset + g * codebook_size の語彙オフセットを加算して [B, (S,), num_quantizers * groups] に整形する
        offset_base = getattr(self, "token_offset", RVQ_TOKEN_OFFSET) if include_offset else 0
        group_offsets = (
            torch.arange(self.vq_groups, device=tokens.device, dtype=tokens.dtype)
            * self.codebook_size
            + offset_base
        )
        return (tokens.movedim(0, -1) + group_offsets).flatten(-2)

    @torch.inference_mode()
    def encode_tokens(self, x: Tensor, include_offset: bool = True) -> Tensor:
        """ピアノロールチャンク [B, 8, 128, 24] を離散トークン列 [B, L * G] に高速変換する（Decoderを省略）。"""
        z_norm = self._encode_latent(x)
        _, tokens, _ = self.rvq(z_norm)
        return self._format_rvq_tokens(tokens, include_offset=include_offset)

    @torch.inference_mode()
    def decode_tokens(self, tokens: Tensor) -> Tensor:
        """
        離散トークン列からピアノロールロジット [N_chunks, 8, 128, 24] を復元する。
        対応入力形状 (L = num_quantizers, G = vq_groups):
          - 1D: [N_chunks * L * G]
          - 2D: [N_chunks, L * G], [B, N_chunks * L * G], または [N_chunks * L, G]
          - 3D: [N_chunks, L, G] または [B, N_chunks * L, G]
          - 4D: [B, N_chunks, L, G]
        """
        num_q = self.num_quantizers
        tokens_per_chunk = num_q * self.vq_groups

        if tokens.dim() not in (1, 2, 3, 4):
            raise ValueError(f"decode_tokens expects 1D/2D/3D/4D tensor, got {tuple(tokens.shape)}.")
        if tokens.numel() == 0 or tokens.numel() % tokens_per_chunk != 0:
            raise ValueError(
                f"decode_tokens requires non-empty tensor whose total elements ({tokens.numel()}) "
                f"are divisible by num_quantizers * vq_groups ({num_q} * {self.vq_groups} = {tokens_per_chunk}), "
                f"got shape {tuple(tokens.shape)}."
            )
        if tokens.dim() == 2 and (
            tokens.size(-1) not in (self.vq_groups, tokens_per_chunk)
            and tokens.size(-1) % tokens_per_chunk != 0
        ):
            raise ValueError(
                f"2D tokens last dim ({tokens.size(-1)}) must be vq_groups ({self.vq_groups}) "
                f"or a multiple of tokens_per_chunk ({tokens_per_chunk}), got {tuple(tokens.shape)}."
            )
        if tokens.dim() in (3, 4) and tokens.size(-1) != self.vq_groups:
            raise ValueError(
                f"{tokens.dim()}D tokens last dim ({tokens.size(-1)}) must equal vq_groups ({self.vq_groups}), "
                f"got {tuple(tokens.shape)}."
            )

        tokens = tokens.reshape(-1, num_q, self.vq_groups)

        offset_base = getattr(self, "token_offset", RVQ_TOKEN_OFFSET)
        if offset_base > 0:
            shifted = torch.where(tokens >= offset_base, tokens - offset_base, tokens.clamp(min=0))
        else:
            shifted = tokens.clamp(min=0)
        local_indices = (shifted % self.codebook_size).long()
        # [N_chunks, L, G] -> [G, N_chunks, L]
        raw_indices = local_indices.movedim(-1, 0).contiguous()
        x_q = self.rvq.get_output_from_indices(raw_indices)
        return self.decoder(x_q)

    def forward(self, x: Tensor):
        z_norm = self._encode_latent(x)
        x_q, tokens, commit_loss = self.rvq(z_norm)
        tokens = self._format_rvq_tokens(tokens, include_offset=True)
        commit_loss = commit_loss.mean()

        out = self.decoder(x_q)
        return out, tokens, commit_loss


def _compute_model_params(model: nn.Module, args: MORTMArgs) -> Tuple[int, int]:
    """総パラメータ数と推論時アクティブパラメータ数（MoE考慮）を返す。"""
    total_params = sum(p.numel() for p in model.parameters())

    if not getattr(args, "use_moe_decoder", False):
        return total_params, total_params

    decoder = getattr(model, "decoder", None)
    if decoder is None or not hasattr(decoder, "layers") or len(decoder.layers) == 0:
        return total_params, total_params

    first_layer = decoder.layers[0]
    if not hasattr(first_layer, "ffn"):
        return total_params, total_params

    moe_module = first_layer.ffn
    if getattr(moe_module, "stacked", False):
        one_expert_params = sum(
            p.numel() // args.num_experts
            for name, p in moe_module.named_parameters(recurse=False)
            if name in ("w13", "w2", "b13", "b2")
        )
    elif hasattr(moe_module, "experts"):
        valid_expert = next((e for e in moe_module.experts if e is not None), None)
        if valid_expert is None:
            return total_params, total_params
        one_expert_params = sum(p.numel() for p in valid_expert.parameters())
    else:
        return total_params, total_params

    if one_expert_params == 0:
        return total_params, total_params

    num_inactive = max(0, args.num_experts - args.topk_experts)
    active_params = total_params - (num_inactive * one_expert_params * args.d_layer)
    return total_params, active_params


class AblationTransformer1(nn.Module):
    """
    1トークンずつフラットに自己回帰するモデル。
    語彙サイズ args.vocab_size (例: token_offset(3) + G * K) を全RVQレイヤーで共有する。
    特殊トークン: <PAD>=0, <BOS>=1, <EOS>=2, RVQトークン=[3, 3 + G * K)。
    """
    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__()
        self.args = args
        self.progress = progress
        self.vq_groups = getattr(args, "vq_groups", 4)
        self.codebook_size = getattr(args, "codebook_size", args.vocab_size // self.vq_groups)
        self.rvq_vocab_size = self.vq_groups * self.codebook_size
        self.token_offset = getattr(args, "token_offset", args.vocab_size - self.rvq_vocab_size)

        if self.token_offset < 0 or args.vocab_size < self.token_offset + self.rvq_vocab_size:
            raise ValueError(
                f"args.vocab_size ({args.vocab_size}) must be >= token_offset + vq_groups * codebook_size "
                f"({self.token_offset} + {self.vq_groups} * {self.codebook_size} = {self.token_offset + self.rvq_vocab_size})."
            )

        device = (
            progress.get_device()
            if progress is not None
            else _get_default_device()
        )

        pad_idx = PAD_TOKEN_ID if self.token_offset > 0 else None
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=pad_idx).to(device)
        self.decoder = MORTMDecoder(args=args, progress=progress).to(device)
        self.Wout = nn.Linear(args.d_model, args.vocab_size).to(device)
        self.wout = self.Wout

    def get_param(self) -> Tuple[int, int]:
        return _compute_model_params(self, self.args)

    def reset_kv_cache(self):
        """全デコーダ層の自己注意KVキャッシュをクリアする。新しい系列の生成開始前に呼び出す。"""
        for layer in self.decoder.layers:
            attn = getattr(layer, "self_attention", None)
            if attn is not None:
                attn.kv_cache = None
                attn.cache_seqlens = None

    def _has_active_kv_cache(self) -> bool:
        return (
            len(self.decoder.layers) > 0
            and getattr(self.decoder.layers[0].self_attention, "kv_cache", None) is not None
        )

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        is_causal: bool = True,
        is_save_cache: bool = False,
    ) -> Tensor:
        if not is_save_cache and self._has_active_kv_cache():
            self.reset_kv_cache()

        # 1ステップ KV-cache デコーディング判定:
        # padding_mask=None かつ is_save_cache=True かつ KVキャッシュ初期化済みで 1トークン入力 ([B] or [B, 1]) の場合
        is_step_decode = (
            padding_mask is None
            and is_save_cache
            and self._has_active_kv_cache()
            and (x.dim() == 1 or (x.dim() == 2 and x.size(1) == 1))
        )

        if is_step_decode:
            keep_seq_dim = (x.dim() == 2)
            if keep_seq_dim:
                x = x.squeeze(1)
            # FlashSelfAttentionM の Phase 2 は出力を 2D [B, d_model] で返すため、
            # 残差加算での [B, 1, D] + [B, D] -> [B, B, D] ブロードキャスト事故を防ぐべく 2D [B, D] で渡す
            x_idx = x.to(dtype=torch.long).clamp(min=0)
            x_emb = self.embedding(x_idx).to(dtype=torch.bfloat16)
            with torch.autocast(device_type=x_emb.device.type, dtype=torch.bfloat16):
                out = self.decoder(
                    tgt=x_emb,
                    tgt_is_causal=is_causal,
                    cu_seqlens=None,
                    max_seqlen=None,
                    batch_size=None,
                    indices=None,
                    is_save_cache=True,
                )
            if keep_seq_dim:
                out = out.unsqueeze(1)
            with torch.autocast(device_type=out.device.type, enabled=False):
                score: Tensor = self.Wout(out.to(self.Wout.weight.dtype)).float()
            return score

        # Prefill / 学習 / 評価フェーズ (2D [B, S] 入力)
        unbatched = (x.dim() == 1)
        if unbatched:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(f"AblationTransformer1 expects x of shape [B, S] (or [B] in decoding), got {tuple(x.shape)}.")

        if padding_mask is None:
            if self.token_offset > 0:
                padding_mask = (x != PAD_TOKEN_ID)
            else:
                padding_mask = (x >= 0)
        elif padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)

        x_idx = x.to(dtype=torch.long).clamp(min=0)
        x_emb = self.embedding(x_idx).to(dtype=torch.bfloat16)
        batch, tgt_len, _ = x_emb.size()
        x_unpadded, indices, cu_seqlens, max_s, _ = unpad_input(x_emb, padding_mask)

        with torch.autocast(device_type=x_unpadded.device.type, dtype=torch.bfloat16):
            out = self.decoder(
                tgt=x_unpadded,
                tgt_is_causal=is_causal,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_s,
                batch_size=batch,
                indices=indices,
                is_save_cache=is_save_cache,
            )
        out = pad_input(out, indices, batch, tgt_len)

        with torch.autocast(device_type=out.device.type, enabled=False):
            score: Tensor = self.Wout(out.to(self.Wout.weight.dtype)).float()

        if unbatched:
            score = score.squeeze(0)
        return score


class AblationTransformer2(nn.Module):
    """
    同一RVQレイヤー内の G 個のグループトークンを1つのソフトトークンに集約して自己回帰するモデル。
    - 入力 x:
      * 3D `[B, S, G]` (または 2D フラット列 `[B, S * G]` を自動で `[B, S, G]` に変換)
      * 1ステップ KV-cache デコーディング時は `[B, G]` または `[B, 1, G]`
      * 特殊トークン (`<PAD>=0`, `<BOS>=1`, `<EOS>=2`) はそのまま特殊Embedding ID `0..2` にマッピングし、
        RVQトークンは各グループ g 専用の Embedding 領域 `[token_offset + g * K, token_offset + (g + 1) * K)` にマッピングする。
    - 出力 score:
      * `[B, S, G, token_offset + K]` (1ステップで G 個のグループの分布を同時予測)
    """
    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__()
        self.args = args
        self.progress = progress
        self.vq_groups = getattr(args, "vq_groups", 4)
        self.codebook_size = getattr(args, "codebook_size", args.vocab_size // self.vq_groups)
        self.rvq_vocab_size = self.vq_groups * self.codebook_size
        self.token_offset = getattr(args, "token_offset", args.vocab_size - self.rvq_vocab_size)

        if self.token_offset < 0 or args.vocab_size < self.token_offset + self.rvq_vocab_size:
            raise ValueError(
                f"args.vocab_size ({args.vocab_size}) must be >= token_offset + vq_groups * codebook_size "
                f"({self.token_offset} + {self.vq_groups} * {self.codebook_size} = {self.token_offset + self.rvq_vocab_size})."
            )

        self.group_vocab_size = self.token_offset + self.codebook_size
        device = (
            progress.get_device()
            if progress is not None
            else _get_default_device()
        )

        self.register_buffer(
            "group_offsets",
            torch.arange(self.vq_groups, device=device, dtype=torch.long) * self.codebook_size,
            persistent=False,
        )

        pad_idx = PAD_TOKEN_ID if self.token_offset > 0 else None
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=pad_idx).to(device)
        self.decoder = MORTMDecoder(args=args, progress=progress).to(device)
        self.Wout = nn.Linear(args.d_model, self.vq_groups * self.group_vocab_size).to(device)
        self.wout = self.Wout

    def get_param(self) -> Tuple[int, int]:
        return _compute_model_params(self, self.args)

    def reset_kv_cache(self):
        """全デコーダ層の自己注意KVキャッシュをクリアする。新しい系列の生成開始前に呼び出す。"""
        for layer in self.decoder.layers:
            attn = getattr(layer, "self_attention", None)
            if attn is not None:
                attn.kv_cache = None
                attn.cache_seqlens = None

    def _has_active_kv_cache(self) -> bool:
        return (
            len(self.decoder.layers) > 0
            and getattr(self.decoder.layers[0].self_attention, "kv_cache", None) is not None
        )

    def to_global_ids(self, x: Tensor) -> Tensor:
        """
        [..., G] のトークンテンソルを Embedding テーブル上のインデックス (torch.long) に正規化する。
        - 特殊トークン (<PAD>=0, <BOS>=1, <EOS>=2 等: x < token_offset) はそのまま [0, token_offset) を保持。
        - RVQトークン (x >= token_offset) は各グループ専用領域 [token_offset + g*K, token_offset + (g+1)*K) に変換。
        """
        x = x.to(dtype=torch.long)
        offsets = self.group_offsets.to(device=x.device, dtype=torch.long)
        if self.token_offset > 0:
            is_rvq = x >= self.token_offset
            local_k = (x.clamp(min=self.token_offset) - self.token_offset) % self.codebook_size
            global_id = self.token_offset + local_k + offsets
            special_id = x.clamp(min=0)
            return torch.where(is_rvq, global_id, special_id)
        else:
            local_k = x.clamp(min=0) % self.codebook_size
            return local_k + offsets

    def to_local_targets(self, targets: Tensor, ignore_index: int = PAD_TOKEN_ID) -> Tensor:
        """
        教師ラベルを出力ロジット `[..., G, group_vocab_size]` に対応するローカルクラスID `[..., G]` (値域 `[0, token_offset + K)`) に変換する。
        2D `[B, S * G]` が渡された場合は自動的に `[B, S, G]` に変換する。
        - パディング (<= 0): ignore_index
        - 特殊トークン (0 < t < token_offset, i.e. <BOS>=1, <EOS>=2): そのまま 1, 2
        - RVQトークン (t >= token_offset): token_offset + (t - token_offset) % K
        """
        if targets.dim() == 2:
            if targets.size(1) % self.vq_groups != 0:
                raise ValueError(
                    f"to_local_targets received 2D targets of shape {tuple(targets.shape)}, "
                    f"whose length {targets.size(1)} is not divisible by vq_groups ({self.vq_groups})."
                )
            targets = targets.reshape(targets.size(0), targets.size(1) // self.vq_groups, self.vq_groups)
        elif targets.dim() != 3 or targets.size(-1) != self.vq_groups:
            raise ValueError(
                f"to_local_targets expects targets of shape [B, S, {self.vq_groups}] or [B, S * {self.vq_groups}], "
                f"got {tuple(targets.shape)}."
            )

        targets = targets.to(dtype=torch.long)
        if self.token_offset > 0:
            is_pad = targets <= 0
            is_rvq = targets >= self.token_offset
            local_k = self.token_offset + (targets.clamp(min=self.token_offset) - self.token_offset) % self.codebook_size
            out = torch.where(is_rvq, local_k, targets.clamp(min=0))
            return torch.where(is_pad, torch.full_like(targets, ignore_index), out)
        else:
            is_valid = targets >= 0
            local_k = targets.clamp(min=0) % self.codebook_size
            return torch.where(is_valid, local_k, torch.full_like(targets, ignore_index))

    def local_to_global_ids(self, local_ids: Tensor) -> Tensor:
        """
        グループ別の予測クラスID `[..., G]` (値域 `[0, token_offset + K)`) を
        グローバルID `[..., G]` (特殊トークン `0..2` または `token_offset + g*K + k`) に逆変換する。
        """
        local_ids = local_ids.to(dtype=torch.long)
        offsets = self.group_offsets.to(device=local_ids.device, dtype=torch.long)
        if self.token_offset > 0:
            is_rvq = local_ids >= self.token_offset
            local_k = (local_ids.clamp(min=self.token_offset) - self.token_offset) % self.codebook_size
            global_id = self.token_offset + local_k + offsets
            return torch.where(is_rvq, global_id, local_ids.clamp(min=0))
        else:
            return (local_ids.clamp(min=0) % self.codebook_size) + offsets

    def forward(
        self,
        x: Tensor,
        padding_mask: Optional[Tensor] = None,
        is_causal: bool = True,
        is_save_cache: bool = False,
    ) -> Tensor:
        if not is_save_cache and self._has_active_kv_cache():
            self.reset_kv_cache()

        # 1ステップ KV-cache デコーディング判定:
        # padding_mask=None かつ is_save_cache=True かつ KVキャッシュ初期化済みで 1ソフトトークン分 ([B, G] or [B, 1, G]) の場合
        is_step_decode = (
            padding_mask is None
            and is_save_cache
            and self._has_active_kv_cache()
            and (
                (x.dim() == 2 and x.size(1) == self.vq_groups)
                or (x.dim() == 3 and x.size(1) == 1 and x.size(2) == self.vq_groups)
            )
        )

        if is_step_decode:
            keep_seq_dim = (x.dim() == 3)
            if keep_seq_dim:
                x = x.squeeze(1)  # [B, G]
            x_global = self.to_global_ids(x)
            # [B, G, d_model] -> [B, d_model] (2D で decoder に渡し残差ブロードキャスト事故を防止)
            x_emb = self.embedding(x_global).sum(dim=-2).to(dtype=torch.bfloat16)
            with torch.autocast(device_type=x_emb.device.type, dtype=torch.bfloat16):
                out = self.decoder(
                    tgt=x_emb,
                    tgt_is_causal=is_causal,
                    cu_seqlens=None,
                    max_seqlen=None,
                    batch_size=None,
                    indices=None,
                    is_save_cache=True,
                )
            if keep_seq_dim:
                out = out.unsqueeze(1)
            with torch.autocast(device_type=out.device.type, enabled=False):
                score: Tensor = self.Wout(out.to(self.Wout.weight.dtype)).float()
                score = score.reshape(*score.shape[:-1], self.vq_groups, self.group_vocab_size)
            return score

        # Prefill / 学習 / 評価フェーズ
        if x.dim() == 2:
            if x.size(1) % self.vq_groups != 0:
                raise ValueError(
                    f"AblationTransformer2 received 2D input of shape {tuple(x.shape)}, whose length {x.size(1)} "
                    f"is not divisible by vq_groups ({self.vq_groups}). Note: for causal training, shift by 1 layer "
                    f"({self.vq_groups} tokens: seq[:, :-{self.vq_groups}] or 3D seq[:, :-1, :]), not by 1 flat token."
                )
            x = x.reshape(x.size(0), x.size(1) // self.vq_groups, self.vq_groups)
        elif x.dim() != 3 or x.size(-1) != self.vq_groups:
            raise ValueError(
                f"AblationTransformer2 expects x of shape [B, S, {self.vq_groups}] or [B, S * {self.vq_groups}], "
                f"got {tuple(x.shape)}."
            )

        batch, tgt_len, _ = x.size()

        # padding_mask をソフトトークン単位の 2D [B, S] に正規化
        if padding_mask is None:
            if self.token_offset > 0:
                padding_mask = (x != PAD_TOKEN_ID).any(dim=-1)
            else:
                padding_mask = (x >= 0).any(dim=-1)
        elif padding_mask.dim() == 3:
            if padding_mask.shape != x.shape:
                raise ValueError(
                    f"3D padding_mask shape {tuple(padding_mask.shape)} does not match x shape {tuple(x.shape)}."
                )
            padding_mask = padding_mask.any(dim=-1)
        elif padding_mask.dim() == 2:
            if padding_mask.size(1) != tgt_len:
                if padding_mask.size(1) == tgt_len * self.vq_groups:
                    padding_mask = padding_mask.reshape(batch, tgt_len, self.vq_groups).any(dim=-1)
                else:
                    raise ValueError(
                        f"2D padding_mask length ({padding_mask.size(1)}) must match soft-token length ({tgt_len}) "
                        f"or flat token length ({tgt_len * self.vq_groups})."
                    )
        else:
            raise ValueError(f"Unsupported padding_mask shape: {tuple(padding_mask.shape)}.")

        x_global = self.to_global_ids(x)
        # [B, S, G] -> [B, S, G, d_model] -> グループ次元(-2)で合算して1つのソフトトークン [B, S, d_model] に集約
        x_emb = self.embedding(x_global).sum(dim=-2).to(dtype=torch.bfloat16)

        x_unpadded, indices, cu_seqlens, max_s, _ = unpad_input(x_emb, padding_mask)

        with torch.autocast(device_type=x_unpadded.device.type, dtype=torch.bfloat16):
            out = self.decoder(
                tgt=x_unpadded,
                tgt_is_causal=is_causal,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_s,
                batch_size=batch,
                indices=indices,
                is_save_cache=is_save_cache,
            )
        out = pad_input(out, indices, batch, tgt_len)

        with torch.autocast(device_type=out.device.type, enabled=False):
            score: Tensor = self.Wout(out.to(self.Wout.weight.dtype)).float()
            # [B, S, G * group_vocab_size] -> [B, S, G, group_vocab_size] に分割し、Gグループ分のロジットを並列出力
            score = score.reshape(batch, tgt_len, self.vq_groups, self.group_vocab_size)
        return score
