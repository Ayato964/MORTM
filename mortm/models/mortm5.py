from torch import nn
from .modules.layers import *
from .modules.config import (
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    RVQ_TOKEN_OFFSET,
    INST_PROGRAM_OFFSET,
    INST_DRUMS_TOKEN_ID,
    INST_EMPTY_TOKEN_ID,
    INST_RVQ_TOKEN_OFFSET,
    NUM_SLOT_TOKENS,
    SLOT_CODE_DRUMS,
    SLOT_CODE_EMPTY,
    slot_code_to_token_id,
    token_id_to_slot_code,
)
from .modules.attention import pad_input, unpad_input, _get_default_device
from vector_quantize_pytorch import GroupedResidualVQ
import torch.distributed as dist
from typing import Optional, Sequence, Tuple, Union
import numpy as np


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
    def decode_tokens(self, tokens: Tensor, token_offset: Optional[int] = None) -> Tensor:
        """
        離散トークン列からピアノロールロジット [N_chunks, 8, 128, 24] を復元する。
        separate_rvq_layers=False (G*K) と True (L*G*K) の両方のトークン列をそのままデコード可能。
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

        offset_base = int(token_offset) if token_offset is not None else getattr(self, "token_offset", RVQ_TOKEN_OFFSET)
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


class _BaseMORTM5Transformer(nn.Module):
    """
    MORTM5 下流自己回帰 Transformer (AblationTransformer1 / AblationTransformer2) の共通基底クラス。
    8通りのアブレーション設定:
      - Tokenizer: M3 (G=8, L=2) / M4 (G=4, L=4)
      - Sequence Mode: Flat (AblationTransformer1) / Grouped Soft-Token (AblationTransformer2)
      - RVQ Layer Vocabulary: Shared (G * K) / Separated (L * G * K)
    および 4スロット楽器条件付け (use_instrument_tokens=True/False) を統一的に管理する。
    """

    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__()
        self.args = args
        self.progress = progress
        self.vq_groups = int(getattr(args, "vq_groups", 4))
        self.vq_layers = int(getattr(args, "vq_layers", getattr(args, "encoder_layer", 2)))
        if self.vq_groups <= 0 or self.vq_layers <= 0:
            raise ValueError(f"vq_groups ({self.vq_groups}) and vq_layers ({self.vq_layers}) must be positive.")
        self.tokens_per_chunk = self.vq_groups * self.vq_layers

        self.separate_rvq_layers = bool(getattr(args, "separate_rvq_layers", False))
        self.use_instrument_tokens = bool(getattr(args, "use_instrument_tokens", False))
        self.effective_layers = self.vq_layers if self.separate_rvq_layers else 1
        self.num_rvq_buckets = self.effective_layers * self.vq_groups

        default_codebook = max(1, (args.vocab_size - getattr(args, "token_offset", RVQ_TOKEN_OFFSET)) // self.num_rvq_buckets)
        self.codebook_size = int(getattr(args, "codebook_size", default_codebook))
        self.rvq_vocab_size = self.num_rvq_buckets * self.codebook_size

        default_offset = (
            INST_RVQ_TOKEN_OFFSET
            if self.use_instrument_tokens
            else max(0, args.vocab_size - self.rvq_vocab_size)
        )
        self.token_offset = int(getattr(args, "token_offset", default_offset))

        if self.use_instrument_tokens and self.token_offset < INST_RVQ_TOKEN_OFFSET:
            raise ValueError(
                f"use_instrument_tokens=True requires token_offset >= {INST_RVQ_TOKEN_OFFSET}, "
                f"got {self.token_offset}."
            )
        if self.token_offset < 0 or args.vocab_size < self.token_offset + self.rvq_vocab_size:
            raise ValueError(
                f"args.vocab_size ({args.vocab_size}) must be >= token_offset + effective_layers * vq_groups * codebook_size "
                f"({self.token_offset} + {self.effective_layers} * {self.vq_groups} * {self.codebook_size} = "
                f"{self.token_offset + self.rvq_vocab_size})."
            )

    def _resolve_init_device(self) -> torch.device:
        return (
            self.progress.get_device()
            if self.progress is not None
            else _get_default_device()
        )

    def get_param(self) -> Tuple[int, int]:
        return _compute_model_params(self, self.args)

    def reset_kv_cache(self) -> None:
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

    def _slot_codes_to_token_list(
        self, slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None
    ) -> list[int]:
        if slot_codes is None:
            default_codes = [SLOT_CODE_DRUMS, 0, SLOT_CODE_EMPTY, SLOT_CODE_EMPTY]
            return [slot_code_to_token_id(c) for c in default_codes]
        if isinstance(slot_codes, Tensor):
            codes_list = [int(x) for x in slot_codes.detach().cpu().reshape(-1).tolist()]
        elif isinstance(slot_codes, np.ndarray):
            codes_list = [int(x) for x in slot_codes.reshape(-1).tolist()]
        else:
            codes_list = [int(x) for x in slot_codes]
        if len(codes_list) != NUM_SLOT_TOKENS:
            raise ValueError(f"Expected {NUM_SLOT_TOKENS} slot_codes, got {len(codes_list)}: {codes_list}")
        return [slot_code_to_token_id(c) for c in codes_list]

    def extract_slot_codes(self, seq: Tensor) -> Optional[np.ndarray]:
        """
        生成系列またはデータセット系列から 4 スロットの楽器コード [4] (int16: 128=Drums, 0..127=Program, -1=Empty) を抽出する。
        楽器トークンが含まれない場合は None を返す。
        """
        flat = seq.detach().cpu().reshape(-1).tolist()
        inst_tokens = [
            int(t) for t in flat
            if INST_PROGRAM_OFFSET <= int(t) <= INST_EMPTY_TOKEN_ID and int(t) < self.token_offset
        ]
        if not inst_tokens:
            return None
        # Grouped モードでは各スロットトークンが G 個ずつ連続するため縮約する
        if isinstance(self, AblationTransformer2) and len(inst_tokens) >= NUM_SLOT_TOKENS * self.vq_groups:
            slot_toks = [inst_tokens[i * self.vq_groups] for i in range(NUM_SLOT_TOKENS)]
        elif len(inst_tokens) >= NUM_SLOT_TOKENS:
            slot_toks = inst_tokens[:NUM_SLOT_TOKENS]
        else:
            return None
        return np.asarray([token_id_to_slot_code(t) for t in slot_toks], dtype=np.int16)

    def extract_rvq_tokens(self, seq: Tensor, target_token_offset: int = RVQ_TOKEN_OFFSET) -> Tensor:
        """
        系列テンソルから特殊トークン (<PAD>, <BOS>, <EOS>, <INST_...>) を除去し、
        MidiVision.decode_tokens にそのまま渡せる標準 RVQ トークン列 (target_token_offset + g * K + k) を抽出する。
        separate_rvq_layers=True / False および use_instrument_tokens=True / False の全モードに対応。
        """
        flat = seq.reshape(-1).to(dtype=torch.long)
        rvq_only = flat[flat >= self.token_offset]
        if rvq_only.numel() == 0 or rvq_only.numel() % self.tokens_per_chunk != 0:
            raise ValueError(
                f"Extracted RVQ token count ({rvq_only.numel()}) must be a positive multiple of "
                f"tokens_per_chunk ({self.tokens_per_chunk})."
            )
        u = rvq_only - self.token_offset
        local_k = u % self.codebook_size
        # [N_chunks, L, G]
        local_k_3d = local_k.reshape(-1, self.vq_layers, self.vq_groups)
        group_offsets = (
            torch.arange(self.vq_groups, device=seq.device, dtype=torch.long) * self.codebook_size
            + int(target_token_offset)
        )
        return (local_k_3d + group_offsets).reshape(-1)

    @staticmethod
    def _sample_from_logits(
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tensor:
        """[..., V] のロジットからサンプリングまたは greedy 選択を行い [...] のインデックスを返す。"""
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1)
        scaled = logits / float(temperature)
        if top_k > 0 and top_k < scaled.size(-1):
            topk_vals, _ = torch.topk(scaled, k=top_k, dim=-1)
            min_topk = topk_vals[..., -1, None]
            scaled = torch.where(scaled < min_topk, torch.full_like(scaled, float("-inf")), scaled)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = torch.zeros_like(scaled, dtype=torch.bool).scatter(-1, sorted_indices, sorted_remove)
            scaled = scaled.masked_fill(remove_mask, float("-inf"))
        probs = torch.softmax(scaled, dim=-1)
        flat_probs = probs.reshape(-1, probs.size(-1))
        sampled = torch.multinomial(flat_probs, num_samples=1).reshape(probs.shape[:-1])
        return sampled


class AblationTransformer1(_BaseMORTM5Transformer):
    """
    1トークンずつフラットに自己回帰するモデル (Flat Autoregressive Transformer)。
    - separate_rvq_layers=False の場合: 語彙サイズ = token_offset + G * K (全RVQレイヤー共有)
    - separate_rvq_layers=True  の場合: 語彙サイズ = token_offset + L * G * K (各RVQレイヤー独立)
    - use_instrument_tokens=True の場合: token_offset = 133 (<BOS> 直後に4スロットの楽器トークンを配置)
    """

    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__(args=args, progress=progress)
        device = self._resolve_init_device()

        pad_idx = PAD_TOKEN_ID if self.token_offset > 0 else None
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=pad_idx).to(device)
        self.decoder = MORTMDecoder(args=args, progress=progress).to(device)
        self.Wout = nn.Linear(args.d_model, args.vocab_size).to(device)
        self.wout = self.Wout

    def build_prefix_tokens(
        self,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """<BOS> (+ use_instrument_tokens=True の場合は 4 スロット楽器トークン) の初期プレフィックス [B, prefix_len] を構築する。"""
        if device is None:
            device = self.embedding.weight.device
        prefix = [BOS_TOKEN_ID]
        if self.use_instrument_tokens:
            prefix.extend(self._slot_codes_to_token_list(slot_codes))
        t = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        return t.expand(batch_size, -1).contiguous()

    def _apply_flat_grammar_mask(self, logits: Tensor, pos: int) -> Tensor:
        """
        フラット自己回帰ステップ pos (0-indexed, 予測対象トークンの系列内位置) における文法マスクを適用する。
        """
        masked = torch.full_like(logits, float("-inf"))
        prefix_len = 1 + (NUM_SLOT_TOKENS if self.use_instrument_tokens else 0)

        if self.use_instrument_tokens and 1 <= pos < prefix_len:
            slot_idx = pos - 1
            if slot_idx == 0:
                masked[..., INST_DRUMS_TOKEN_ID] = logits[..., INST_DRUMS_TOKEN_ID]
                masked[..., INST_EMPTY_TOKEN_ID] = logits[..., INST_EMPTY_TOKEN_ID]
            else:
                masked[..., INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID] = logits[..., INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID]
                masked[..., INST_EMPTY_TOKEN_ID] = logits[..., INST_EMPTY_TOKEN_ID]
            return masked

        rvq_step = max(0, pos - prefix_len)
        intra_chunk = rvq_step % self.tokens_per_chunk
        layer_idx = intra_chunk // self.vq_groups
        group_idx = intra_chunk % self.vq_groups
        eff_layer = layer_idx if self.separate_rvq_layers else 0

        low = self.token_offset + (eff_layer * self.vq_groups + group_idx) * self.codebook_size
        high = low + self.codebook_size
        masked[..., low:high] = logits[..., low:high]
        return masked

    @torch.inference_mode()
    def generate(
        self,
        num_chunks: int = 8,
        prompt: Optional[Tensor] = None,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        append_eos: bool = True,
        repetition_penalty: float = 1.0,
        repetition_window: int = 24,
        fine_temperature: Optional[float] = None,
        fine_top_k: Optional[int] = None,
        fine_top_p: Optional[float] = None,
    ) -> Tensor:
        """
        KVキャッシュと文法制約マスクを用いて num_chunks チャンク分のトークン系列 [B, S_total] を自己回帰生成する。
        """
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        device = self.embedding.weight.device
        if prompt is None:
            seq = self.build_prefix_tokens(slot_codes=slot_codes, batch_size=batch_size, device=device)
        else:
            seq = prompt.to(device=device, dtype=torch.long)
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)

        prefix_len = 1 + (NUM_SLOT_TOKENS if self.use_instrument_tokens else 0)
        target_total_len = prefix_len + num_chunks * self.tokens_per_chunk
        if seq.size(1) >= target_total_len:
            return seq

        recent_l0_by_group: list[list[Tensor]] = [[] for _ in range(self.vq_groups)]
        if repetition_penalty > 1.0 and seq.size(1) > prefix_len:
            for p_idx in range(prefix_len, seq.size(1)):
                intra = (p_idx - prefix_len) % self.tokens_per_chunk
                if intra < self.vq_groups:
                    recent_l0_by_group[intra].append(seq[:, p_idx])

        was_training = self.training
        self.eval()
        self.reset_kv_cache()
        try:
            # Prefill
            logits = self.forward(seq, is_causal=True, is_save_cache=True)
            next_logits = logits[:, -1, :]  # [B, V]

            cur_pos = seq.size(1)
            generated = [seq]
            while cur_pos < target_total_len:
                constrained = self._apply_flat_grammar_mask(next_logits, pos=cur_pos)
                if cur_pos >= prefix_len:
                    rvq_step = cur_pos - prefix_len
                    intra_chunk = rvq_step % self.tokens_per_chunk
                    layer_idx = intra_chunk // self.vq_groups
                    group_idx = intra_chunk % self.vq_groups
                    if layer_idx == 0:
                        if repetition_penalty > 1.0 and recent_l0_by_group[group_idx]:
                            win = recent_l0_by_group[group_idx][-max(1, int(repetition_window)) :]
                            for b_i in range(constrained.size(0)):
                                counts: dict[int, int] = {}
                                for past_t in win:
                                    tid = int(past_t[b_i].item())
                                    counts[tid] = counts.get(tid, 0) + 1
                                for tid, cnt in counts.items():
                                    pen = float(repetition_penalty) ** min(cnt, 4)
                                    val = constrained[b_i, tid]
                                    constrained[b_i, tid] = val / pen if val > 0 else val * pen
                        step_temp = temperature
                        step_top_k = top_k
                        step_top_p = top_p
                    else:
                        step_temp = fine_temperature if fine_temperature is not None else temperature
                        step_top_k = fine_top_k if fine_top_k is not None else top_k
                        step_top_p = fine_top_p if fine_top_p is not None else top_p
                else:
                    layer_idx = -1
                    group_idx = -1
                    step_temp = temperature
                    step_top_k = top_k
                    step_top_p = top_p

                next_tok = self._sample_from_logits(
                    constrained, temperature=step_temp, top_k=step_top_k, top_p=step_top_p
                ).unsqueeze(1)  # [B, 1]
                if layer_idx == 0 and repetition_penalty > 1.0 and 0 <= group_idx < self.vq_groups:
                    recent_l0_by_group[group_idx].append(next_tok.squeeze(1))
                generated.append(next_tok)
                cur_pos += 1
                if cur_pos < target_total_len:
                    step_logits = self.forward(next_tok, is_causal=True, is_save_cache=True)
                    next_logits = step_logits[:, -1, :]

            if append_eos and self.token_offset > EOS_TOKEN_ID:
                eos_col = torch.full((seq.size(0), 1), EOS_TOKEN_ID, dtype=torch.long, device=device)
                generated.append(eos_col)
            return torch.cat(generated, dim=1)
        finally:
            self.reset_kv_cache()
            if was_training:
                self.train()

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


class AblationTransformer2(_BaseMORTM5Transformer):
    """
    同一RVQレイヤー内の G 個のグループトークンを1つのソフトトークンに集約して自己回帰するモデル (Grouped Soft-Token Transformer)。
    - 入力 x:
      * 3D `[B, S, G]` (または 2D フラット列 `[B, S * G]` を自動で `[B, S, G]` に変換)
      * 1ステップ KV-cache デコーディング時は `[B, G]` または `[B, 1, G]`
      * 特殊トークン (`<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<INST_...>=3..132`) は特殊Embedding ID `[0, token_offset)` にマッピングし、
        同一トークン G 個の反復和を `1 / G` スケーリング（平均）して分散 1.0 を維持する。
      * RVQトークン (`>= token_offset`) は `(eff_layer * G + g)` 専用の領域にマッピングし、
        独立な G 個のグループ埋め込みの和を `1 / sqrt(G)` スケーリングして分散 1.0 を維持する。
    - 出力 score:
      * `[B, S, G, group_vocab_size]` (ここで `group_vocab_size = token_offset + K`)
    """

    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__(args=args, progress=progress)
        self.group_vocab_size = self.token_offset + self.codebook_size
        self.rvq_emb_scale = float(self.vq_groups) ** -0.5
        self.special_emb_scale = 1.0 / float(self.vq_groups)

        device = self._resolve_init_device()

        self.register_buffer(
            "group_offsets",
            torch.arange(self.vq_groups, device=device, dtype=torch.long) * self.codebook_size,
            persistent=False,
        )

        pad_idx = PAD_TOKEN_ID if self.token_offset > 0 else None
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=pad_idx).to(device)
        self.decoder = MORTMDecoder(args=args, progress=progress).to(device)
        if self.separate_rvq_layers:
            self.Wout = DecodeCodebookHead(args).to(device)
        else:
            self.Wout = nn.Linear(args.d_model, self.vq_groups * self.group_vocab_size).to(device)
        self.wout = self.Wout

    def to_global_ids(self, x: Tensor) -> Tensor:
        """
        [..., G] のトークンテンソルを Embedding テーブル上のグローバルインデックス (torch.long) に正規化する。
        - 特殊トークン (x < token_offset) はそのまま [0, token_offset) を保持。
        - RVQトークン (x >= token_offset) は `token_offset + (eff_layer * G + g) * K + k` に正規化。
        """
        x = x.to(dtype=torch.long)
        offsets = self.group_offsets.to(device=x.device, dtype=torch.long)
        layer_stride = self.vq_groups * self.codebook_size

        if self.token_offset > 0:
            is_rvq = x >= self.token_offset
            u = x.clamp(min=self.token_offset) - self.token_offset
            local_k = u % self.codebook_size
            if self.separate_rvq_layers:
                eff_layer = (u // layer_stride) % self.effective_layers
                global_id = self.token_offset + eff_layer * layer_stride + offsets + local_k
            else:
                global_id = self.token_offset + offsets + local_k
            special_id = x.clamp(min=0)
            return torch.where(is_rvq, global_id, special_id)
        else:
            u = x.clamp(min=0)
            local_k = u % self.codebook_size
            if self.separate_rvq_layers:
                eff_layer = (u // layer_stride) % self.effective_layers
                return eff_layer * layer_stride + offsets + local_k
            return local_k + offsets

    def _aggregate_group_embeddings(self, x: Tensor) -> Tensor:
        """
        [..., G] のトークン列を Embedding [..., G, d_model] に変換し、分散を 1.0 に保ちつつ [..., d_model] に集約する。
        - RVQトークン (x >= token_offset): 独立な G 個のグループ埋め込みの和を 1 / sqrt(G) でスケーリング
        - 特殊トークン (0 < x < token_offset): 同一トークン G 個の反復和を 1 / G でスケーリング (= 平均)
        """
        x_global = self.to_global_ids(x)
        emb = self.embedding(x_global)  # [..., G, d_model]
        if self.token_offset > 0:
            is_rvq = (x >= self.token_offset).unsqueeze(-1)
            scale = torch.where(
                is_rvq,
                emb.new_full((), self.rvq_emb_scale),
                emb.new_full((), self.special_emb_scale),
            )
            emb = emb * scale
        else:
            emb = emb * self.rvq_emb_scale
        return emb.sum(dim=-2)

    def to_local_targets(self, targets: Tensor, ignore_index: int = PAD_TOKEN_ID) -> Tensor:
        """
        教師ラベルを出力ロジット `[..., G, group_vocab_size]` に対応するローカルクラスID `[..., G]`
        (値域 `[0, token_offset + K)`) に変換する。
        2D `[B, S * G]` が渡された場合は自動的に `[B, S, G]` に変換する。
        - パディング (<= 0): ignore_index
        - 特殊トークン (0 < t < token_offset): そのまま [1, token_offset)
        - RVQトークン (t >= token_offset): token_offset + k
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
            u = targets.clamp(min=self.token_offset) - self.token_offset
            local_k = u % self.codebook_size
            local_cls = self.token_offset + local_k
            out = torch.where(is_rvq, local_cls, targets.clamp(min=0))
            return torch.where(is_pad, torch.full_like(targets, ignore_index), out)
        else:
            is_valid = targets >= 0
            u = targets.clamp(min=0)
            local_k = u % self.codebook_size
            return torch.where(is_valid, local_k, torch.full_like(targets, ignore_index))

    def local_to_global_ids(
        self,
        local_ids: Tensor,
        layer_idx: Optional[int | Tensor] = None,
    ) -> Tensor:
        """
        グループ別の予測ローカルクラスID `[..., G]` (値域 `[0, token_offset + K)`) を
        グローバルID `[..., G]` (特殊トークン `[0, token_offset)` または `token_offset + (eff_layer * G + g) * K + k`) に逆変換する。
        """
        local_ids = local_ids.to(dtype=torch.long)
        offsets = self.group_offsets.to(device=local_ids.device, dtype=torch.long)
        layer_stride = self.vq_groups * self.codebook_size

        if self.token_offset > 0:
            is_rvq = local_ids >= self.token_offset
            v = local_ids.clamp(min=self.token_offset) - self.token_offset
            local_k = v % self.codebook_size
            if self.separate_rvq_layers:
                if layer_idx is not None:
                    eff_layer = layer_idx % self.effective_layers
                    if isinstance(eff_layer, Tensor):
                        eff_layer = eff_layer.to(device=local_ids.device, dtype=torch.long)
                elif local_ids.dim() == 3:
                    rvq_pos = (is_rvq.any(dim=-1).cumsum(dim=1) - 1).clamp(min=0)
                    eff_layer = (rvq_pos % self.effective_layers).unsqueeze(-1)
                else:
                    eff_layer = (v // self.codebook_size) % self.effective_layers
                global_id = self.token_offset + eff_layer * layer_stride + offsets + local_k
            else:
                global_id = self.token_offset + offsets + local_k
            return torch.where(is_rvq, global_id, local_ids.clamp(min=0))
        else:
            v = local_ids.clamp(min=0)
            local_k = v % self.codebook_size
            if self.separate_rvq_layers:
                if layer_idx is not None:
                    eff_layer = layer_idx % self.effective_layers
                    if isinstance(eff_layer, Tensor):
                        eff_layer = eff_layer.to(device=local_ids.device, dtype=torch.long)
                elif local_ids.dim() == 3:
                    seq_idx = torch.arange(local_ids.size(1), device=local_ids.device, dtype=torch.long)
                    eff_layer = (seq_idx % self.effective_layers).view(1, -1, 1)
                else:
                    eff_layer = (v // self.codebook_size) % self.effective_layers
                return eff_layer * layer_stride + offsets + local_k
            return local_k + offsets

    def build_prefix_tokens(
        self,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        <BOS>*G (+ use_instrument_tokens=True の場合は [Slot0]*G .. [Slot3]*G) の初期プレフィックス
        [B, prefix_soft_len, G] を構築する。
        """
        if device is None:
            device = self.embedding.weight.device
        prefix_tokens = [BOS_TOKEN_ID]
        if self.use_instrument_tokens:
            prefix_tokens.extend(self._slot_codes_to_token_list(slot_codes))
        # [prefix_soft_len, G]
        t = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(-1).expand(-1, self.vq_groups)
        return t.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _apply_grouped_grammar_mask(self, logits: Tensor, soft_pos: int) -> Tensor:
        """
        ソフトトークン自己回帰ステップ soft_pos (0-indexed, 予測対象ソフトトークンの位置) における文法マスクを適用する。
        logits: [B, G, group_vocab_size]
        """
        masked = torch.full_like(logits, float("-inf"))
        prefix_soft_len = 1 + (NUM_SLOT_TOKENS if self.use_instrument_tokens else 0)

        if self.use_instrument_tokens and 1 <= soft_pos < prefix_soft_len:
            slot_idx = soft_pos - 1
            if slot_idx == 0:
                masked[..., INST_DRUMS_TOKEN_ID] = logits[..., INST_DRUMS_TOKEN_ID]
                masked[..., INST_EMPTY_TOKEN_ID] = logits[..., INST_EMPTY_TOKEN_ID]
            else:
                masked[..., INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID] = logits[..., INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID]
                masked[..., INST_EMPTY_TOKEN_ID] = logits[..., INST_EMPTY_TOKEN_ID]
            return masked

        low = self.token_offset
        high = low + self.codebook_size
        masked[..., low:high] = logits[..., low:high]
        return masked

    @torch.inference_mode()
    def generate(
        self,
        num_chunks: int = 8,
        prompt: Optional[Tensor] = None,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        append_eos: bool = True,
        return_flat: bool = False,
        repetition_penalty: float = 1.0,
        repetition_window: int = 24,
        fine_temperature: Optional[float] = None,
        fine_top_k: Optional[int] = None,
        fine_top_p: Optional[float] = None,
    ) -> Tensor:
        """
        KVキャッシュと文法制約マスクを用いて num_chunks チャンク分のソフトトークン系列
        [B, S_soft, G] (return_flat=True の場合は [B, S_soft * G]) を自己回帰生成する。
        """
        if num_chunks <= 0:
            raise ValueError(f"num_chunks must be positive, got {num_chunks}")
        device = self.embedding.weight.device
        if prompt is None:
            seq = self.build_prefix_tokens(slot_codes=slot_codes, batch_size=batch_size, device=device)
        else:
            seq = prompt.to(device=device, dtype=torch.long)
            if seq.dim() == 2:
                if seq.size(1) % self.vq_groups != 0:
                    raise ValueError(
                        f"2D prompt length ({seq.size(1)}) must be divisible by vq_groups ({self.vq_groups})."
                    )
                seq = seq.reshape(seq.size(0), seq.size(1) // self.vq_groups, self.vq_groups)
            elif seq.dim() != 3 or seq.size(-1) != self.vq_groups:
                raise ValueError(f"Expected prompt of shape [B, S, {self.vq_groups}], got {tuple(seq.shape)}.")

        prefix_soft_len = 1 + (NUM_SLOT_TOKENS if self.use_instrument_tokens else 0)
        target_soft_len = prefix_soft_len + num_chunks * self.vq_layers
        if seq.size(1) >= target_soft_len:
            return seq.reshape(seq.size(0), -1) if return_flat else seq

        recent_l0: list[Tensor] = []
        if repetition_penalty > 1.0 and seq.size(1) > prefix_soft_len:
            prompt_rvq_local = self.to_local_targets(seq[:, prefix_soft_len:, :])
            for l0_pos in range(0, prompt_rvq_local.size(1), self.vq_layers):
                recent_l0.append(prompt_rvq_local[:, l0_pos, :])

        was_training = self.training
        self.eval()
        self.reset_kv_cache()
        try:
            # Prefill
            logits = self.forward(seq, is_causal=True, is_save_cache=True)
            next_logits = logits[:, -1, :, :]  # [B, G, group_vocab_size]

            cur_soft_pos = seq.size(1)
            generated = [seq]
            while cur_soft_pos < target_soft_len:
                constrained = self._apply_grouped_grammar_mask(next_logits, soft_pos=cur_soft_pos)
                if self.use_instrument_tokens and 1 <= cur_soft_pos < prefix_soft_len:
                    # 楽器スロットステップでは全 G グループで同一の楽器トークンを共有する
                    inst_tok = self._sample_from_logits(
                        constrained[:, 0, :], temperature=temperature, top_k=top_k, top_p=top_p
                    )  # [B]
                    next_global = inst_tok[:, None, None].expand(-1, 1, self.vq_groups).contiguous()
                else:
                    cur_layer_idx = max(0, cur_soft_pos - prefix_soft_len) % self.vq_layers
                    if cur_layer_idx == 0:
                        if repetition_penalty > 1.0 and recent_l0:
                            win = recent_l0[-max(1, int(repetition_window)) :]
                            for b_i in range(constrained.size(0)):
                                for g_i in range(self.vq_groups):
                                    counts: dict[int, int] = {}
                                    for past_t in win:
                                        tid = int(past_t[b_i, g_i].item())
                                        if tid >= self.token_offset:
                                            counts[tid] = counts.get(tid, 0) + 1
                                    for tid, cnt in counts.items():
                                        pen = float(repetition_penalty) ** min(cnt, 4)
                                        val = constrained[b_i, g_i, tid]
                                        constrained[b_i, g_i, tid] = val / pen if val > 0 else val * pen
                        step_temp = temperature
                        step_top_k = top_k
                        step_top_p = top_p
                    else:
                        step_temp = fine_temperature if fine_temperature is not None else temperature
                        step_top_k = fine_top_k if fine_top_k is not None else top_k
                        step_top_p = fine_top_p if fine_top_p is not None else top_p

                    local_sampled = self._sample_from_logits(
                        constrained, temperature=step_temp, top_k=step_top_k, top_p=step_top_p
                    )  # [B, G]
                    if cur_layer_idx == 0 and repetition_penalty > 1.0:
                        recent_l0.append(local_sampled)
                    next_global = self.local_to_global_ids(
                        local_sampled, layer_idx=cur_layer_idx
                    ).unsqueeze(1)  # [B, 1, G]

                generated.append(next_global)
                cur_soft_pos += 1
                if cur_soft_pos < target_soft_len:
                    step_logits = self.forward(next_global, is_causal=True, is_save_cache=True)
                    next_logits = step_logits[:, -1, :, :]

            if append_eos and self.token_offset > EOS_TOKEN_ID:
                eos_soft = torch.full(
                    (seq.size(0), 1, self.vq_groups), EOS_TOKEN_ID, dtype=torch.long, device=device
                )
                generated.append(eos_soft)
            full_seq = torch.cat(generated, dim=1)
            return full_seq.reshape(full_seq.size(0), -1) if return_flat else full_seq
        finally:
            self.reset_kv_cache()
            if was_training:
                self.train()

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
            attn0 = self.decoder.layers[0].self_attention
            step_pos = int(attn0.cache_seqlens[0].item()) if attn0.cache_seqlens is not None else 0

            # [B, G] -> [B, d_model] (分散1.0正規化付き集約、2D で decoder に渡し残差ブロードキャスト事故を防止)
            x_emb = self._aggregate_group_embeddings(x).to(dtype=torch.bfloat16)
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
                if isinstance(self.Wout, DecodeCodebookHead):
                    score: Tensor = self.Wout(out.to(self.Wout.weight.dtype), start_pos=step_pos).float()
                else:
                    score = self.Wout(out.to(self.Wout.weight.dtype)).float()
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

        # [B, S, G] -> [B, S, d_model] (分散1.0正規化付きソフトトークン集約)
        x_emb = self._aggregate_group_embeddings(x).to(dtype=torch.bfloat16)

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
            if isinstance(self.Wout, DecodeCodebookHead):
                score = self.Wout(out.to(self.Wout.weight.dtype), start_pos=0).float()
            else:
                score = self.Wout(out.to(self.Wout.weight.dtype)).float()
                # [B, S, G * group_vocab_size] -> [B, S, G, group_vocab_size] に分割し、Gグループ分のロジットを並列出力
                score = score.reshape(batch, tgt_len, self.vq_groups, self.group_vocab_size)
        return score


ABLATION_MODEL_CONFIGS = {
    "m3_flat_shared": "configs/models/mortm5/ablation/m3_flat_shared.json",
    "m3_flat_separated": "configs/models/mortm5/ablation/m3_flat_separated.json",
    "m3_grouped_shared": "configs/models/mortm5/ablation/m3_grouped_shared.json",
    "m3_grouped_separated": "configs/models/mortm5/ablation/m3_grouped_separated.json",
    "m4_flat_shared": "configs/models/mortm5/ablation/m4_flat_shared.json",
    "m4_flat_separated": "configs/models/mortm5/ablation/m4_flat_separated.json",
    "m4_grouped_shared": "configs/models/mortm5/ablation/m4_grouped_shared.json",
    "m4_grouped_separated": "configs/models/mortm5/ablation/m4_grouped_separated.json",
}


def build_mortm5_ablation_model(
    args: MORTM5Args | str | dict = "m4_grouped_shared",
    progress=None,
) -> _BaseMORTM5Transformer:
    """
    MORTM5Args (またはプリセット名 'm4_grouped_shared' / '#7'、設定辞書、JSONファイルパス) の
    model_mode ("flat" or "grouped") に応じて AblationTransformer1 または AblationTransformer2 を構築するファクトリ関数。
    """
    if not isinstance(args, MORTM5Args):
        args = MORTM5Args(args)
    mode = getattr(args, "model_mode", "flat")
    if mode == "flat":
        return AblationTransformer1(args=args, progress=progress)
    elif mode == "grouped":
        return AblationTransformer2(args=args, progress=progress)
    else:
        raise ValueError(f"Unsupported args.model_mode: {mode}. Expected 'flat' or 'grouped'.")

