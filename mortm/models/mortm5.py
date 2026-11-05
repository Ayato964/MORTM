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
    OCTUPLE_DEFAULT_ATTR_SIZES,
    REMI_DEFAULT_ATTR_SIZES,
    slot_code_to_token_id,
    token_id_to_slot_code,
)
from .modules.attention import pad_input, unpad_input, _get_default_device
from vector_quantize_pytorch import GroupedResidualVQ
import torch.distributed as dist
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
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
        self.is_symbolic = bool(
            getattr(args, "symbolic_vocab", False)
            or getattr(args, "model_mode", "flat") in ("octuple", "remi")
        )
        self.effective_layers = self.vq_layers if self.separate_rvq_layers else 1
        self.num_rvq_buckets = self.effective_layers * self.vq_groups

        default_codebook = max(1, (args.vocab_size - getattr(args, "token_offset", RVQ_TOKEN_OFFSET)) // self.num_rvq_buckets)
        self.codebook_size = int(getattr(args, "codebook_size", default_codebook))
        if self.is_symbolic:
            raw_offset = int(
                getattr(
                    args,
                    "token_offset",
                    INST_RVQ_TOKEN_OFFSET if self.use_instrument_tokens else RVQ_TOKEN_OFFSET,
                )
            )
            self.token_offset = raw_offset
            self.rvq_vocab_size = max(0, int(args.vocab_size) - self.token_offset)
        else:
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

    @property
    def step_unit_width(self) -> int:
        return 1

    def format_training_targets(self, targets: Tensor, ignore_index: int = PAD_TOKEN_ID) -> Tensor:
        return targets.reshape(-1).long()

    def _resolve_init_device(self) -> torch.device:
        return (
            self.progress.get_device()
            if self.progress is not None
            else _get_default_device()
        )

    def get_param(self) -> Tuple[int, int]:
        return _compute_model_params(self, self.args)

    def get_non_embedding_param_count(self) -> int:
        return sum(p.numel() for p in self.decoder.parameters())

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
        w = self.step_unit_width
        if w > 1 and len(inst_tokens) >= NUM_SLOT_TOKENS * w:
            slot_toks = [inst_tokens[i * w] for i in range(NUM_SLOT_TOKENS)]
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
        """[..., V] のロジットから `-inf` 文法マスクを厳守しつつサンプリングまたは greedy 選択を行い [...] のインデックスを返す。"""
        if logits.numel() == 0:
            return torch.empty(logits.shape[:-1], dtype=torch.long, device=logits.device)

        logits_f32 = logits.to(dtype=torch.float32)
        was_nan = torch.isnan(logits_f32)
        clean_logits = torch.nan_to_num(
            logits_f32, nan=float("-inf"), posinf=1.0e30, neginf=float("-inf")
        )
        row_has_finite = torch.isfinite(clean_logits).any(dim=-1, keepdim=True)
        row_has_nan = was_nan.any(dim=-1, keepdim=True)
        nan_uniform = torch.where(
            was_nan,
            torch.zeros_like(clean_logits),
            torch.full_like(clean_logits, float("-inf")),
        )
        base_logits = torch.where(
            row_has_finite,
            clean_logits,
            torch.where(row_has_nan, nan_uniform, torch.zeros_like(clean_logits)),
        )

        if float(temperature) <= 0.0:
            return torch.argmax(base_logits, dim=-1)

        row_max = base_logits.amax(dim=-1, keepdim=True)
        shifted = base_logits - row_max
        scaled = torch.nan_to_num(
            shifted / float(temperature), nan=float("-inf"), posinf=0.0, neginf=float("-inf")
        )

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

    @property
    def step_unit_width(self) -> int:
        return self.vq_groups

    def format_training_targets(self, targets: Tensor, ignore_index: int = PAD_TOKEN_ID) -> Tensor:
        return self.to_local_targets(targets, ignore_index=ignore_index).reshape(-1).long()

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


class OctupleTransformer(_BaseMORTM5Transformer):
    """
    Paradigm 2 (Note-Bundled Symbolic): 1ノート6属性 `(bar, pos, slot, pitch, dur, vel)` を
    1自己回帰ステップのソフトトークンに集約して予測する `Octuple-MMT` モデル。
    - 入力 x:
      * 3D `[B, S, 6]` (または 2D フラット列 `[B, S * 6]` を自動で `[B, S, 6]` に変換)
      * 1ステップ KV-cache デコーディング時は `[B, 6]` または `[B, 1, 6]`
      * 特殊・楽器トークン (`<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<INST_...>=3..132`) は `[0, token_offset)` にマッピングし、
        同一トークン 6 個の反復和を `1 / 6` スケーリング（平均）して分散 1.0 を維持する。
      * ノート属性トークン (`>= token_offset`) は各属性専用スライス `token_offset + attr_offsets[m] + v_m` を参照し、
        独立な 6 属性埋め込みの和を `1 / sqrt(6)` スケーリングして分散 1.0 を維持する。
    - 出力 score:
      * `[B, S, 6, token_offset + max(attr_sizes)]` (標準設定で `[B, S, 6, 445]`、各属性の無効クラスは `-inf` マスク済み)
    """

    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__(args=args, progress=progress)
        self.vocab_size: int = int(args.vocab_size)
        self.attr_sizes: Tuple[int, ...] = tuple(
            int(s) for s in getattr(args, "octuple_attr_sizes", OCTUPLE_DEFAULT_ATTR_SIZES)
        )
        self.tuple_size: int = len(self.attr_sizes)
        self.max_attr_size: int = int(max(self.attr_sizes))
        self.local_vocab_size: int = self.token_offset + self.max_attr_size
        self.note_emb_scale: float = float(self.tuple_size) ** -0.5
        self.special_emb_scale: float = 1.0 / float(self.tuple_size)

        offsets_list: list[int] = []
        cur = 0
        for sz in self.attr_sizes:
            if sz <= 0:
                raise ValueError(f"All octuple_attr_sizes must be positive, got {self.attr_sizes}")
            offsets_list.append(cur)
            cur += sz
        expected_vocab = self.token_offset + cur
        if self.vocab_size < expected_vocab:
            raise ValueError(
                f"OctupleTransformer vocab_size ({self.vocab_size}) is smaller than required "
                f"token_offset + sum(attr_sizes) = {self.token_offset} + {cur} = {expected_vocab}."
            )

        device = self._resolve_init_device()
        self.register_buffer(
            "attr_offsets",
            torch.tensor(offsets_list, device=device, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "attr_limits",
            torch.tensor(list(self.attr_sizes), device=device, dtype=torch.long),
            persistent=False,
        )

        pad_idx = PAD_TOKEN_ID if self.token_offset > 0 else None
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=pad_idx).to(device)
        self.decoder = MORTMDecoder(args=args, progress=progress).to(device)
        self.Wout = DecodeOctupleHead(args).to(device)
        self.wout = self.Wout
        self._symbolic_tokenizer = None

    @property
    def step_unit_width(self) -> int:
        return self.tuple_size

    def format_training_targets(self, targets: Tensor, ignore_index: int = PAD_TOKEN_ID) -> Tensor:
        return self.to_local_targets(targets, ignore_index=ignore_index).reshape(-1).long()

    def to_global_ids(self, x: Tensor) -> Tensor:
        """
        [..., 6] のトークンテンソルを Embedding テーブル上の有効なグローバルインデックス `[0, vocab_size)` に正規化する。
        """
        x = x.to(dtype=torch.long)
        offsets = self.attr_offsets.to(device=x.device, dtype=torch.long)
        limits = self.attr_limits.to(device=x.device, dtype=torch.long)

        if self.token_offset > 0:
            is_note = x >= self.token_offset
            u = x.clamp(min=self.token_offset) - self.token_offset
            local_v = torch.minimum((u - offsets).clamp(min=0), limits - 1)
            global_id = self.token_offset + offsets + local_v
            special_id = x.clamp(min=0, max=self.token_offset - 1)
            return torch.where(is_note, global_id, special_id)
        u = x.clamp(min=0)
        local_v = torch.minimum((u - offsets).clamp(min=0), limits - 1)
        return offsets + local_v

    def _aggregate_tuple_embeddings(self, x: Tensor) -> Tensor:
        """
        [..., 6] のトークン列を Embedding [..., 6, d_model] に変換し、分散 1.0 を保ちつつ [..., d_model] に集約する。
        - ノート属性トークン (x >= token_offset): 独立な 6 属性埋め込みの和を `1 / sqrt(6)` スケーリング
        - 特殊・楽器トークン (0 < x < token_offset): 同一トークン 6 個の反復和を `1 / 6` スケーリング (= 平均)
        """
        x_global = self.to_global_ids(x)
        emb = self.embedding(x_global)
        if self.token_offset > 0:
            is_note = (x >= self.token_offset).unsqueeze(-1)
            scale = torch.where(
                is_note,
                emb.new_full((), self.note_emb_scale),
                emb.new_full((), self.special_emb_scale),
            )
            emb = emb * scale
        else:
            emb = emb * self.note_emb_scale
        return emb.sum(dim=-2)

    def to_local_targets(self, targets: Tensor, ignore_index: int = PAD_TOKEN_ID) -> Tensor:
        """
        グローバル教師ラベル `[B, S, 6]` (または 2D `[B, S * 6]`) を `DecodeOctupleHead` の出力ロジット
        `[B, S, 6, token_offset + max_attr_size]` に対応するローカルクラスID `[B, S, 6]` に変換する。
        - パディング (<= 0): `ignore_index`
        - 特殊・楽器トークン (`0 < t < token_offset`): そのまま `[1, token_offset)` (`<EOS>=2`, `<INST_...>=3..132`)
        - ノート属性トークン (`t >= token_offset`): `token_offset + v_m` (`0 <= v_m < attr_sizes[m]`)
        """
        if targets.dim() == 2:
            if targets.size(1) % self.tuple_size != 0:
                raise ValueError(
                    f"OctupleTransformer.to_local_targets received 2D targets of shape {tuple(targets.shape)}, "
                    f"whose length {targets.size(1)} is not divisible by tuple_size ({self.tuple_size})."
                )
            targets = targets.reshape(targets.size(0), targets.size(1) // self.tuple_size, self.tuple_size)
        elif targets.dim() != 3 or targets.size(-1) != self.tuple_size:
            raise ValueError(
                f"OctupleTransformer.to_local_targets expects shape [B, S, {self.tuple_size}] or "
                f"[B, S * {self.tuple_size}], got {tuple(targets.shape)}."
            )

        targets = targets.to(dtype=torch.long)
        offsets = self.attr_offsets.to(device=targets.device, dtype=torch.long)
        limits = self.attr_limits.to(device=targets.device, dtype=torch.long)

        if self.token_offset > 0:
            is_pad = targets <= 0
            is_note = targets >= self.token_offset
            u = targets.clamp(min=self.token_offset) - self.token_offset
            local_v = torch.minimum((u - offsets).clamp(min=0), limits - 1)
            local_cls = self.token_offset + local_v
            out = torch.where(is_note, local_cls, targets.clamp(min=0, max=self.token_offset - 1))
            return torch.where(is_pad, torch.full_like(targets, ignore_index), out)

        is_valid = targets >= 0
        u = targets.clamp(min=0)
        local_v = torch.minimum((u - offsets).clamp(min=0), limits - 1)
        return torch.where(is_valid, local_v, torch.full_like(targets, ignore_index))

    def local_to_global_ids(self, local_ids: Tensor) -> Tensor:
        """
        属性別の予測ローカルクラスID `[..., 6]` (`[0, token_offset + attr_sizes[m])`) を
        グローバルID `[..., 6]` (`[0, token_offset)` または `token_offset + attr_offsets[m] + v_m`) に逆変換する。
        """
        local_ids = local_ids.to(dtype=torch.long)
        offsets = self.attr_offsets.to(device=local_ids.device, dtype=torch.long)
        limits = self.attr_limits.to(device=local_ids.device, dtype=torch.long)

        if self.token_offset > 0:
            is_note = local_ids >= self.token_offset
            local_v = torch.minimum((local_ids.clamp(min=self.token_offset) - self.token_offset), limits - 1)
            global_id = self.token_offset + offsets + local_v
            return torch.where(is_note, global_id, local_ids.clamp(min=0, max=self.token_offset - 1))

        local_v = torch.minimum(local_ids.clamp(min=0), limits - 1)
        return offsets + local_v

    def build_prefix_tokens(
        self,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """
        `<BOS>*6` (+ `use_instrument_tokens=True` の場合は `[Slot0]*6 .. [Slot3]*6`) の初期プレフィックス
        `[B, prefix_soft_len, 6]` を構築する。
        """
        if device is None:
            device = self.embedding.weight.device
        prefix_tokens = [BOS_TOKEN_ID]
        if self.use_instrument_tokens:
            prefix_tokens.extend(self._slot_codes_to_token_list(slot_codes))
        t = torch.tensor(prefix_tokens, dtype=torch.long, device=device).unsqueeze(-1).expand(-1, self.tuple_size)
        return t.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def _apply_instrument_prefix_mask(self, logits_m0: Tensor, soft_pos: int) -> Tensor:
        masked = torch.full_like(logits_m0, float("-inf"))
        slot_idx = soft_pos - 1
        if slot_idx == 0:
            masked[..., INST_DRUMS_TOKEN_ID] = logits_m0[..., INST_DRUMS_TOKEN_ID]
            masked[..., INST_EMPTY_TOKEN_ID] = logits_m0[..., INST_EMPTY_TOKEN_ID]
        else:
            masked[..., INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID] = logits_m0[
                ..., INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID
            ]
            masked[..., INST_EMPTY_TOKEN_ID] = logits_m0[..., INST_EMPTY_TOKEN_ID]
        return masked

    @torch.inference_mode()
    def generate(
        self,
        max_notes: int = 1024,
        max_bars: Optional[int] = 16,
        min_notes: int = 1,
        min_bar: int = 0,
        min_pos: int = 0,
        max_steps: Optional[int] = None,
        prompt: Optional[Tensor] = None,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        monotonic_time: bool = True,
        allow_eos: bool = True,
        append_eos: bool = True,
        return_flat: bool = False,
    ) -> Tensor:
        """
        KVキャッシュ・属性文法マスク・非空スロット制約・辞書順時間単調制約 `(bar, pos) >= (prev_bar, prev_pos)` を用いて
        6属性タプル系列 `[B, S_soft, 6]` (`return_flat=True` の場合は `[B, S_soft * 6]`) を自己回帰生成する。
        - `min_bar`, `min_pos`: 継続生成時の最小 `(bar, pos)` 下限（半拍チャンク境界プロンプトにも対応）。
        - `max_steps`: ウィンドウ内の排他的ステップ上限（`bar * 192 + pos < max_steps`）。
        - `min_notes`: 新規に生成すべき最小音符ステップ数（プロンプト内の既存音符数は含めない）。
        """
        if max_notes <= 0:
            raise ValueError(f"max_notes must be positive, got {max_notes}")
        device = self.embedding.weight.device
        if prompt is None:
            seq = self.build_prefix_tokens(slot_codes=slot_codes, batch_size=batch_size, device=device)
        else:
            seq = prompt.to(device=device, dtype=torch.long)
            if seq.dim() == 2:
                if seq.size(1) % self.tuple_size != 0:
                    raise ValueError(
                        f"2D prompt length ({seq.size(1)}) must be divisible by tuple_size ({self.tuple_size})."
                    )
                seq = seq.reshape(seq.size(0), seq.size(1) // self.tuple_size, self.tuple_size)
            elif seq.dim() != 3 or seq.size(-1) != self.tuple_size:
                raise ValueError(f"Expected prompt of shape [B, S, {self.tuple_size}], got {tuple(seq.shape)}.")

        bsz = seq.size(0)
        valid_mask = (seq != PAD_TOKEN_ID).any(dim=-1) if self.token_offset > 0 else (seq >= 0).any(dim=-1)
        prompt_list = [seq[b_i][valid_mask[b_i]] for b_i in range(bsz)]
        if any(p.size(0) == 0 for p in prompt_list):
            raise ValueError("Every sequence in prompt must contain at least one valid non-pad tuple.")
        seq = torch.nn.utils.rnn.pad_sequence(prompt_list, batch_first=True, padding_value=PAD_TOKEN_ID)
        valid_mask = (seq != PAD_TOKEN_ID).any(dim=-1) if self.token_offset > 0 else (seq >= 0).any(dim=-1)
        prompt_lens = valid_mask.sum(dim=1)  # [B]

        prefix_soft_len = 1 + (NUM_SLOT_TOKENS if self.use_instrument_tokens else 0)
        pos_limit = self.attr_sizes[1]
        slot_limit = self.attr_sizes[2]
        raw_bar_limit = self.attr_sizes[0] if max_bars is None else max(1, min(self.attr_sizes[0], int(max_bars)))
        horizon_steps = raw_bar_limit * pos_limit
        if max_steps is not None:
            horizon_steps = max(1, min(horizon_steps, int(max_steps)))
        max_valid_step = horizon_steps - 1
        bar_limit = (max_valid_step // pos_limit) + 1
        last_bar_pos_limit = (max_valid_step % pos_limit) + 1

        if int(min_bar) < 0:
            req_min_step = 0
        else:
            req_min_step = int(min_bar) * pos_limit + max(0, min(int(min_pos), pos_limit - 1))
        clamped_min_step = min(req_min_step, max_valid_step)
        floor_bar = clamped_min_step // pos_limit
        floor_pos = clamped_min_step % pos_limit

        prev_bar = torch.full((bsz,), floor_bar, dtype=torch.long, device=device)
        prev_pos = torch.full((bsz,), floor_pos, dtype=torch.long, device=device)
        finished = torch.zeros((bsz,), dtype=torch.bool, device=device)
        active_slots = torch.ones((bsz, slot_limit), dtype=torch.bool, device=device)

        if self.use_instrument_tokens and slot_codes is not None and slot_limit == NUM_SLOT_TOKENS:
            init_tok_ids = self._slot_codes_to_token_list(slot_codes)
            for s_idx, tid in enumerate(init_tok_ids[:slot_limit]):
                active_slots[:, s_idx] = (int(tid) != INST_EMPTY_TOKEN_ID)

        for b_i in range(bsz):
            p_seq = prompt_list[b_i]
            p_len = int(p_seq.size(0))
            if self.use_instrument_tokens and slot_limit == NUM_SLOT_TOKENS:
                inst_end = min(p_len, prefix_soft_len)
                for s_pos in range(1, inst_end):
                    tid = int(p_seq[s_pos, 0].item())
                    if INST_PROGRAM_OFFSET <= tid <= INST_EMPTY_TOKEN_ID:
                        active_slots[b_i, s_pos - 1] = (tid != INST_EMPTY_TOKEN_ID)

            if p_len > prefix_soft_len:
                p_local = self.to_local_targets(p_seq[prefix_soft_len:].unsqueeze(0), ignore_index=PAD_TOKEN_ID)[0]
                cur_b = int(prev_bar[b_i].item())
                cur_p = int(prev_pos[b_i].item())
                for t_i in range(p_local.size(0)):
                    tok0 = int(p_local[t_i, 0].item())
                    if tok0 == EOS_TOKEN_ID:
                        finished[b_i] = True
                        break
                    if tok0 >= self.token_offset:
                        raw_nb = max(0, tok0 - self.token_offset)
                        tok1 = int(p_local[t_i, 1].item())
                        raw_np = max(0, min(tok1 - self.token_offset, pos_limit - 1)) if tok1 >= self.token_offset else 0
                        note_step = min(raw_nb * pos_limit + raw_np, max_valid_step)
                        nb = note_step // pos_limit
                        np_pos = note_step % pos_limit
                        if nb > cur_b or (nb == cur_b and np_pos > cur_p):
                            cur_b = nb
                            cur_p = np_pos
                prev_bar[b_i] = cur_b
                prev_pos[b_i] = cur_p

        if bool(finished.all().item()):
            return seq.reshape(bsz, -1) if return_flat else seq

        was_training = self.training
        self.eval()
        self.reset_kv_cache()
        try:
            logits = self.forward(seq, padding_mask=valid_mask, is_causal=True, is_save_cache=True)
            batch_arange = torch.arange(bsz, device=device, dtype=torch.long)
            next_logits = logits[batch_arange, prompt_lens - 1, :, :]  # [B, 6, local_vocab_size]

            cur_pos_vec = prompt_lens.clone()
            notes_gen_vec = torch.zeros((bsz,), dtype=torch.long, device=device)
            gen_lists: list[list[Tensor]] = [[prompt_list[b_i]] for b_i in range(bsz)]
            bar_idx_grid = torch.arange(bar_limit, device=device, dtype=torch.long).unsqueeze(0)
            pos_idx_grid = torch.arange(pos_limit, device=device, dtype=torch.long).unsqueeze(0)
            max_notes_int = int(max_notes)
            min_notes_int = max(0, int(min_notes))

            while True:
                reached_note_cap = notes_gen_vec >= max_notes_int
                active_batch = (~finished) & (~reached_note_cap)
                if not bool(active_batch.any().item()):
                    break

                need_inst = (
                    active_batch
                    & bool(self.use_instrument_tokens)
                    & (cur_pos_vec >= 1)
                    & (cur_pos_vec < prefix_soft_len)
                )

                # A. Note tuple sampling across batch
                has_any_slot = active_slots.any(dim=-1)  # [B]
                bar_logits = torch.full(
                    (bsz, self.local_vocab_size),
                    float("-inf"),
                    device=device,
                    dtype=next_logits.dtype,
                )
                bar_slice = next_logits[:, 0, self.token_offset : self.token_offset + bar_limit]
                if monotonic_time:
                    cur_min_bar = prev_bar.clamp(min=0, max=bar_limit - 1).unsqueeze(1)
                    bar_slice = bar_slice.masked_fill(bar_idx_grid < cur_min_bar, float("-inf"))
                if allow_eos and self.token_offset > EOS_TOKEN_ID:
                    bar_slice = torch.where(
                        has_any_slot.unsqueeze(1),
                        bar_slice,
                        torch.full_like(bar_slice, float("-inf")),
                    )
                bar_logits[:, self.token_offset : self.token_offset + bar_limit] = bar_slice

                if allow_eos and self.token_offset > EOS_TOKEN_ID:
                    can_eos = (notes_gen_vec >= min_notes_int) | (~has_any_slot)
                    if bool(can_eos.any().item()):
                        bar_logits[:, EOS_TOKEN_ID] = torch.where(
                            can_eos,
                            next_logits[:, 0, EOS_TOKEN_ID],
                            bar_logits[:, EOS_TOKEN_ID],
                        )

                bar_local = self._sample_from_logits(
                    bar_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )  # [B]
                step_is_eos = (bar_local == EOS_TOKEN_ID) | (~active_batch)
                sampled_bar = (bar_local - self.token_offset).clamp(min=0, max=bar_limit - 1)

                rest_logits = torch.full(
                    (bsz, self.tuple_size - 1, self.local_vocab_size),
                    float("-inf"),
                    device=device,
                    dtype=next_logits.dtype,
                )
                for m in range(1, self.tuple_size):
                    sz = self.attr_sizes[m]
                    rest_logits[:, m - 1, self.token_offset : self.token_offset + sz] = next_logits[
                        :, m, self.token_offset : self.token_offset + sz
                    ]

                # Mask inactive SLOT_CODE_EMPTY slots when at least 1 slot is active
                slot_slice = rest_logits[:, 1, self.token_offset : self.token_offset + slot_limit]
                masked_slot_slice = slot_slice.masked_fill(~active_slots, float("-inf"))
                rest_logits[:, 1, self.token_offset : self.token_offset + slot_limit] = torch.where(
                    has_any_slot.unsqueeze(1),
                    masked_slot_slice,
                    slot_slice,
                )

                pos_slice = rest_logits[:, 0, self.token_offset : self.token_offset + pos_limit]
                if last_bar_pos_limit < pos_limit:
                    in_last_bar = (~step_is_eos) & (~need_inst) & (sampled_bar == (bar_limit - 1))
                    if bool(in_last_bar.any().item()):
                        cur_max_pos = torch.where(
                            in_last_bar,
                            torch.full_like(prev_pos, last_bar_pos_limit),
                            torch.full_like(prev_pos, pos_limit),
                        ).unsqueeze(1)
                        pos_slice = pos_slice.masked_fill(pos_idx_grid >= cur_max_pos, float("-inf"))

                if monotonic_time:
                    same_bar = (~step_is_eos) & (~need_inst) & (sampled_bar == prev_bar)
                    if bool(same_bar.any().item()):
                        cur_min_pos = torch.where(
                            same_bar,
                            prev_pos.clamp(min=0, max=pos_limit - 1),
                            torch.zeros_like(prev_pos),
                        ).unsqueeze(1)
                        pos_slice = pos_slice.masked_fill(pos_idx_grid < cur_min_pos, float("-inf"))
                rest_logits[:, 0, self.token_offset : self.token_offset + pos_limit] = pos_slice

                rest_local = self._sample_from_logits(
                    rest_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )  # [B, 5]
                local_tuple = torch.cat([bar_local.unsqueeze(1), rest_local], dim=1)
                tuple_global = self.local_to_global_ids(local_tuple)

                newly_eos = active_batch & (~need_inst) & (bar_local == EOS_TOKEN_ID)
                if bool(newly_eos.any().item()):
                    tuple_global = torch.where(
                        newly_eos.unsqueeze(1),
                        torch.full_like(tuple_global, EOS_TOKEN_ID),
                        tuple_global,
                    )

                # B. Instrument prefix token sampling if any item has prompt_len < prefix_soft_len
                if bool(need_inst.any().item()):
                    inst_rows = next_logits[:, 0, :].clone()
                    for b_i in range(bsz):
                        if bool(need_inst[b_i].item()):
                            s_pos = int(cur_pos_vec[b_i].item())
                            inst_rows[b_i] = self._apply_instrument_prefix_mask(inst_rows[b_i], soft_pos=s_pos)
                    inst_tok = self._sample_from_logits(
                        inst_rows, temperature=temperature, top_k=top_k, top_p=top_p
                    )
                    inst_tuple = inst_tok.unsqueeze(1).expand(-1, self.tuple_size)
                    tuple_global = torch.where(need_inst.unsqueeze(1), inst_tuple, tuple_global)
                    for b_i in range(bsz):
                        if bool(need_inst[b_i].item()) and slot_limit == NUM_SLOT_TOKENS:
                            s_idx = int(cur_pos_vec[b_i].item()) - 1
                            if 0 <= s_idx < slot_limit:
                                active_slots[b_i, s_idx] = (int(inst_tok[b_i].item()) != INST_EMPTY_TOKEN_ID)

                if bool((~active_batch).any().item()):
                    tuple_global = torch.where(
                        (~active_batch).unsqueeze(1),
                        torch.full_like(tuple_global, PAD_TOKEN_ID),
                        tuple_global,
                    )

                for b_i in range(bsz):
                    if bool(active_batch[b_i].item()):
                        gen_lists[b_i].append(tuple_global[b_i : b_i + 1])

                sampled_note_mask = active_batch & (~need_inst) & (~newly_eos)
                sampled_pos = (rest_local[:, 0] - self.token_offset).clamp(min=0, max=pos_limit - 1)
                prev_bar = torch.where(sampled_note_mask, sampled_bar, prev_bar)
                prev_pos = torch.where(sampled_note_mask, sampled_pos, prev_pos)
                notes_gen_vec = notes_gen_vec + sampled_note_mask.to(dtype=torch.long)
                cur_pos_vec = cur_pos_vec + active_batch.to(dtype=torch.long)
                finished = finished | newly_eos

                next_active = (~finished) & (notes_gen_vec < max_notes_int)
                if not bool(next_active.any().item()):
                    break

                next_global = tuple_global.unsqueeze(1)
                step_logits = self.forward(next_global, is_causal=True, is_save_cache=True)
                next_logits = step_logits[:, -1, :, :]

            if append_eos and self.token_offset > EOS_TOKEN_ID:
                eos_row = torch.full((1, self.tuple_size), EOS_TOKEN_ID, dtype=torch.long, device=device)
                for b_i in range(bsz):
                    if not bool(finished[b_i].item()):
                        gen_lists[b_i].append(eos_row)

            per_seq = [torch.cat(parts, dim=0) for parts in gen_lists]
            full_seq = torch.nn.utils.rnn.pad_sequence(
                per_seq, batch_first=True, padding_value=PAD_TOKEN_ID
            )
            return full_seq.reshape(bsz, -1) if return_flat else full_seq
        finally:
            self.reset_kv_cache()
            if was_training:
                self.train()

    def get_symbolic_tokenizer(self):
        if self._symbolic_tokenizer is None:
            from mortm.utils.symbolic_tokenizers import OctupleMMTTokenizer, OctupleVocabSpec

            vocab_spec = OctupleVocabSpec(
                token_offset=self.token_offset,
                attr_sizes=self.attr_sizes,
            )
            self._symbolic_tokenizer = OctupleMMTTokenizer(vocab_spec=vocab_spec)
        return self._symbolic_tokenizer

    def decode_to_notes(
        self,
        seq: Tensor,
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_symbolic_tokenizer().decode_tokens_to_notes(
            seq, slot_codes=slot_codes, num_chunks=num_chunks
        )

    def decode_to_chunks_u8(
        self,
        seq: Tensor,
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.get_symbolic_tokenizer().decode_tokens_to_chunks_u8(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )

    def decode_to_chunks_float(
        self,
        seq: Tensor,
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.get_symbolic_tokenizer().decode_tokens_to_chunks_float(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )

    def decode_to_midi(
        self,
        seq: Tensor,
        out_path: str,
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> np.ndarray:
        return self.get_symbolic_tokenizer().decode_tokens_to_midi(
            seq, output_path=out_path, slot_codes=slot_codes, num_chunks=num_chunks
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

        is_step_decode = (
            padding_mask is None
            and is_save_cache
            and self._has_active_kv_cache()
            and (
                (x.dim() == 2 and x.size(1) == self.tuple_size)
                or (x.dim() == 3 and x.size(1) == 1 and x.size(2) == self.tuple_size)
            )
        )

        if is_step_decode:
            keep_seq_dim = x.dim() == 3
            if keep_seq_dim:
                x = x.squeeze(1)

            x_emb = self._aggregate_tuple_embeddings(x).to(dtype=torch.bfloat16)
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

        if x.dim() == 2:
            if x.size(1) % self.tuple_size != 0:
                raise ValueError(
                    f"OctupleTransformer received 2D input of shape {tuple(x.shape)}, whose length {x.size(1)} "
                    f"is not divisible by tuple_size ({self.tuple_size}). Note: for causal training, shift by 1 tuple "
                    f"({self.tuple_size} flat tokens: seq[:, :-{self.tuple_size}] or 3D seq[:, :-1, :])."
                )
            x = x.reshape(x.size(0), x.size(1) // self.tuple_size, self.tuple_size)
        elif x.dim() != 3 or x.size(-1) != self.tuple_size:
            raise ValueError(
                f"OctupleTransformer expects x of shape [B, S, {self.tuple_size}] or [B, S * {self.tuple_size}], "
                f"got {tuple(x.shape)}."
            )

        batch, tgt_len, _ = x.size()
        if tgt_len == 0:
            return x.new_empty((batch, 0, self.tuple_size, self.local_vocab_size), dtype=torch.float32)

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
                if padding_mask.size(1) == tgt_len * self.tuple_size:
                    padding_mask = padding_mask.reshape(batch, tgt_len, self.tuple_size).any(dim=-1)
                else:
                    raise ValueError(
                        f"2D padding_mask length ({padding_mask.size(1)}) must match tuple length ({tgt_len}) "
                        f"or flat token length ({tgt_len * self.tuple_size})."
                    )
        else:
            raise ValueError(f"Unsupported padding_mask shape: {tuple(padding_mask.shape)}.")

        x_emb = self._aggregate_tuple_embeddings(x).to(dtype=torch.bfloat16)
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
            score = self.Wout(out.to(self.Wout.weight.dtype)).float()
        return score


class REMITransformer(_BaseMORTM5Transformer):
    """
    `Multitrack-REMI` フラットイベントベースライン用 1次元自己回帰モデル (`vocab_size = 802`):
      - 共有プレフィックス `0..132`: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<INST_0..3>=3..132` (`token_offset = 133`)
      - 本体イベント語彙 `133..801`: `<BAR>=133`, `<POS_0..191>=134..325`, `<SLOT_0..3>=326..329`,
        `<PITCH_0..127>=330..457`, `<DUR_0..311>=458..769`, `<VEL_0..31>=770..801`
      - `SyMIc-G4L4` および `Octuple-MMT` と 1パラメータの狂いもない共通バックボーン (`MORTMDecoder`: `163,603,456` params)
      - 有限状態オートマトン (FSM) 文法マスク・非空スロット制約・時間単調性・半拍チャンク境界継続に対応した `generate()`
    """

    _STATE_BEFORE_BAR: int = 0
    _STATE_AFTER_BAR: int = 1
    _STATE_AFTER_POS: int = 2
    _STATE_AFTER_SLOT: int = 3
    _STATE_AFTER_PITCH: int = 4
    _STATE_AFTER_DUR: int = 5
    _STATE_AFTER_VEL: int = 6

    def __init__(self, args: MORTM5Args, progress=None):
        super().__init__(args=args, progress=progress)
        device = self._resolve_init_device()

        self.attr_sizes: Tuple[int, ...] = tuple(
            int(x) for x in getattr(args, "remi_attr_sizes", REMI_DEFAULT_ATTR_SIZES)
        )
        if len(self.attr_sizes) != 6 or int(self.attr_sizes[0]) != 1 or any(s <= 0 for s in self.attr_sizes):
            raise ValueError(
                f"REMITransformer expects 6 positive remi_attr_sizes starting with bar=1, got {self.attr_sizes}"
            )
        min_vocab = self.token_offset + sum(self.attr_sizes)
        if int(args.vocab_size) < min_vocab:
            raise ValueError(
                f"REMITransformer vocab_size ({args.vocab_size}) must be >= {min_vocab} "
                f"(token_offset={self.token_offset} + sum(remi_attr_sizes)={sum(self.attr_sizes)})."
            )

        self.bar_token_id: int = int(self.token_offset)
        self.pos_offset: int = self.bar_token_id + int(self.attr_sizes[0])
        self.pos_size: int = int(self.attr_sizes[1])
        self.slot_offset: int = self.pos_offset + self.pos_size
        self.slot_size: int = int(self.attr_sizes[2])
        self.pitch_offset: int = self.slot_offset + self.slot_size
        self.pitch_size: int = int(self.attr_sizes[3])
        self.dur_offset: int = self.pitch_offset + self.pitch_size
        self.dur_size: int = int(self.attr_sizes[4])
        self.vel_offset: int = self.dur_offset + self.dur_size
        self.vel_size: int = int(self.attr_sizes[5])
        self.max_segment_bars: int = 16

        pad_idx = PAD_TOKEN_ID if self.token_offset > 0 else None
        self.embedding = nn.Embedding(args.vocab_size, args.d_model, padding_idx=pad_idx).to(device)
        self.decoder = MORTMDecoder(args=args, progress=progress).to(device)
        self.Wout = nn.Linear(args.d_model, args.vocab_size, bias=bool(args.use_bias)).to(device)
        self.wout = self.Wout
        self._symbolic_tokenizer = None

    def build_prefix_tokens(
        self,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        """`<BOS>` (+ `use_instrument_tokens=True` の場合は 4 スロット楽器トークン) の初期プレフィックス `[B, prefix_len]` を構築する。"""
        if device is None:
            device = self.embedding.weight.device
        prefix = [BOS_TOKEN_ID]
        if self.use_instrument_tokens:
            prefix.extend(self._slot_codes_to_token_list(slot_codes))
        t = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        return t.expand(batch_size, -1).contiguous()

    def _parse_prompt_fsm_state(
        self,
        p_seq: Tensor,
        prefix_len: int,
        initial_active_slots: Sequence[bool],
    ) -> Tuple[int, int, int, list[bool], bool]:
        """1系列分のプロンプト `p_seq [S_p]` を走査し `(state, cur_bar, last_pos, active_slots, is_finished)` を復元する。"""
        p_toks = [int(t) for t in p_seq.detach().cpu().tolist()]
        p_len = len(p_toks)
        act_slots = [bool(x) for x in initial_active_slots]

        if self.use_instrument_tokens and self.slot_size == NUM_SLOT_TOKENS:
            inst_end = min(p_len, prefix_len)
            for s_pos in range(1, inst_end):
                tid = p_toks[s_pos]
                if INST_PROGRAM_OFFSET <= tid <= INST_EMPTY_TOKEN_ID:
                    act_slots[s_pos - 1] = (tid != INST_EMPTY_TOKEN_ID)

        state = self._STATE_BEFORE_BAR
        cur_bar = -1
        last_pos = -1
        is_finished = False

        if p_len > prefix_len:
            vel_hi = self.vel_offset + self.vel_size
            for tok in p_toks[prefix_len:]:
                if tok == EOS_TOKEN_ID:
                    is_finished = True
                    break
                if tok == self.bar_token_id:
                    cur_bar += 1
                    last_pos = -1
                    state = self._STATE_AFTER_BAR
                elif self.pos_offset <= tok < self.slot_offset:
                    if cur_bar < 0:
                        cur_bar = 0
                    last_pos = tok - self.pos_offset
                    state = self._STATE_AFTER_POS
                elif self.slot_offset <= tok < self.pitch_offset:
                    state = self._STATE_AFTER_SLOT
                elif self.pitch_offset <= tok < self.dur_offset:
                    state = self._STATE_AFTER_PITCH
                elif self.dur_offset <= tok < self.vel_offset:
                    state = self._STATE_AFTER_DUR
                elif self.vel_offset <= tok < vel_hi:
                    state = self._STATE_AFTER_VEL

        return state, cur_bar, last_pos, act_slots, is_finished

    def _unmask_active_slots_row(
        self,
        constrained_row: Tensor,
        raw_row: Tensor,
        active_slots: Sequence[bool],
        has_any_slot: bool,
    ) -> None:
        if has_any_slot:
            for s_i in range(self.slot_size):
                if active_slots[s_i]:
                    idx = self.slot_offset + s_i
                    constrained_row[idx] = raw_row[idx]
        else:
            constrained_row[self.slot_offset : self.slot_offset + self.slot_size] = raw_row[
                self.slot_offset : self.slot_offset + self.slot_size
            ]

    def _forward_decode_step(self, next_tok: Tensor) -> Tensor:
        """1ステップ KVキャッシュ推論を実行し `[B, vocab_size]` ロジットを返す。"""
        if next_tok.dim() == 2 and next_tok.size(1) == 1:
            next_tok = next_tok.squeeze(1)
        return self.forward(next_tok, padding_mask=None, is_causal=True, is_save_cache=True)

    @torch.inference_mode()
    def generate(
        self,
        max_notes: int = 1024,
        max_bars: Optional[int] = 16,
        min_notes: int = 1,
        min_bar: int = 0,
        min_pos: int = 0,
        max_steps: Optional[int] = None,
        min_bars: int = 0,
        prompt: Optional[Tensor] = None,
        slot_codes: Optional[Union[Sequence[int], np.ndarray, Tensor]] = None,
        batch_size: int = 1,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        monotonic_time: bool = True,
        allow_eos: bool = True,
        append_eos: bool = True,
    ) -> Tensor:
        """
        KVキャッシュ・有限状態オートマトン (FSM) 文法マスク・非空スロット制約・時間単調性を用いて
        `Multitrack-REMI` 系列 `[B, S_total]` を自己回帰生成する。
        - `min_bar`, `min_pos`: 継続生成時の最小 `(bar, pos)` 下限（半拍チャンク境界プロンプトにも対応）。
        - `max_steps`: ウィンドウ内の排他的ステップ上限（`bar * 192 + pos < max_steps`）。
        - `min_notes`: 新規に生成すべき最小音符数（プロンプト内の既存音符数は含めない）。
        """
        if max_notes <= 0:
            raise ValueError(f"max_notes must be positive, got {max_notes}")
        device = self.embedding.weight.device
        if prompt is None:
            seq = self.build_prefix_tokens(slot_codes=slot_codes, batch_size=batch_size, device=device)
        else:
            seq = prompt.to(device=device, dtype=torch.long)
            if seq.dim() == 1:
                seq = seq.unsqueeze(0)
            elif seq.dim() != 2:
                raise ValueError(f"REMITransformer.generate expects prompt of shape [B, S] or [S], got {tuple(seq.shape)}.")

        bsz = int(seq.size(0))
        valid_mask = (seq != PAD_TOKEN_ID) if self.token_offset > 0 else (seq >= 0)
        prompt_list = [seq[b_i][valid_mask[b_i]] for b_i in range(bsz)]
        if any(p.numel() == 0 for p in prompt_list):
            raise ValueError("Every sequence in prompt must contain at least one valid non-pad token.")
        seq = torch.nn.utils.rnn.pad_sequence(prompt_list, batch_first=True, padding_value=PAD_TOKEN_ID)
        valid_mask = (seq != PAD_TOKEN_ID) if self.token_offset > 0 else (seq >= 0)
        prompt_lens = valid_mask.sum(dim=1)  # [B]

        prefix_len = 1 + (NUM_SLOT_TOKENS if self.use_instrument_tokens else 0)
        pos_limit = self.pos_size
        raw_bar_limit = (
            self.max_segment_bars
            if max_bars is None
            else max(1, min(self.max_segment_bars, int(max_bars)))
        )
        horizon_steps = raw_bar_limit * pos_limit
        if max_steps is not None:
            horizon_steps = max(1, min(horizon_steps, int(max_steps)))
        max_valid_step = horizon_steps - 1
        bar_limit = (max_valid_step // pos_limit) + 1
        last_bar_pos_limit = (max_valid_step % pos_limit) + 1

        if int(min_bar) < 0:
            req_min_step = 0
        else:
            req_min_step = int(min_bar) * pos_limit + max(0, min(int(min_pos), pos_limit - 1))
        clamped_min_step = min(req_min_step, max_valid_step)
        floor_bar = clamped_min_step // pos_limit
        floor_pos = clamped_min_step % pos_limit
        min_bars_idx = max(-1, min(bar_limit - 1, int(min_bars) - 1))

        default_active = [True for _ in range(self.slot_size)]
        if self.use_instrument_tokens and slot_codes is not None and self.slot_size == NUM_SLOT_TOKENS:
            init_tok_ids = self._slot_codes_to_token_list(slot_codes)
            for s_idx, tid in enumerate(init_tok_ids[: self.slot_size]):
                default_active[s_idx] = (int(tid) != INST_EMPTY_TOKEN_ID)

        state_list: list[int] = []
        cur_bar_list: list[int] = []
        last_pos_list: list[int] = []
        active_slots_list: list[list[bool]] = []
        finished_list: list[bool] = []
        halted_list: list[bool] = [False for _ in range(bsz)]
        cur_pos_list: list[int] = [int(p.numel()) for p in prompt_list]
        notes_gen_list: list[int] = [0 for _ in range(bsz)]

        for b_i in range(bsz):
            st, cb, lp, act_s, is_fin = self._parse_prompt_fsm_state(
                prompt_list[b_i], prefix_len=prefix_len, initial_active_slots=default_active
            )
            state_list.append(st)
            cur_bar_list.append(cb)
            last_pos_list.append(lp)
            active_slots_list.append(act_s)
            finished_list.append(is_fin)

        if all(finished_list):
            return seq

        max_notes_int = int(max_notes)
        min_notes_int = max(0, int(min_notes))
        vel_hi = self.vel_offset + self.vel_size

        was_training = self.training
        self.eval()
        self.reset_kv_cache()
        try:
            logits = self.forward(seq, padding_mask=valid_mask, is_causal=True, is_save_cache=True)
            batch_arange = torch.arange(bsz, device=device, dtype=torch.long)
            next_logits = logits[batch_arange, prompt_lens - 1, :]  # [B, V]
            gen_lists: list[list[Tensor]] = [[prompt_list[b_i]] for b_i in range(bsz)]

            while True:
                active_b = [
                    (not finished_list[b_i])
                    and (not halted_list[b_i])
                    and not (
                        notes_gen_list[b_i] >= max_notes_int
                        and state_list[b_i] in (self._STATE_BEFORE_BAR, self._STATE_AFTER_BAR, self._STATE_AFTER_VEL)
                    )
                    for b_i in range(bsz)
                ]
                if not any(active_b):
                    break

                constrained = torch.full_like(next_logits, float("-inf"))
                for b_i in range(bsz):
                    if not active_b[b_i]:
                        constrained[b_i, PAD_TOKEN_ID] = 0.0
                        continue

                    c_pos = cur_pos_list[b_i]
                    raw_row = next_logits[b_i]
                    out_row = constrained[b_i]

                    # A. Instrument prefix tokens
                    if self.use_instrument_tokens and 1 <= c_pos < prefix_len:
                        slot_idx = c_pos - 1
                        if slot_idx == 0:
                            out_row[INST_DRUMS_TOKEN_ID] = raw_row[INST_DRUMS_TOKEN_ID]
                            out_row[INST_EMPTY_TOKEN_ID] = raw_row[INST_EMPTY_TOKEN_ID]
                        else:
                            out_row[INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID] = raw_row[
                                INST_PROGRAM_OFFSET:INST_DRUMS_TOKEN_ID
                            ]
                            out_row[INST_EMPTY_TOKEN_ID] = raw_row[INST_EMPTY_TOKEN_ID]
                        continue

                    # B. Body event FSM transitions
                    st = state_list[b_i]
                    cb = cur_bar_list[b_i]
                    lp = last_pos_list[b_i]
                    ng = notes_gen_list[b_i]
                    act_s = active_slots_list[b_i]
                    has_slot = any(act_s)
                    can_emit_notes = has_slot
                    has_valid = False

                    if st == self._STATE_BEFORE_BAR:
                        if cb + 1 < bar_limit:
                            out_row[self.bar_token_id] = raw_row[self.bar_token_id]
                            has_valid = True

                    elif st == self._STATE_AFTER_BAR:
                        if cb + 1 < bar_limit:
                            out_row[self.bar_token_id] = raw_row[self.bar_token_id]
                            has_valid = True
                        if can_emit_notes and floor_bar <= cb < bar_limit:
                            p_end = last_bar_pos_limit if cb == (bar_limit - 1) else pos_limit
                            p_start = floor_pos if cb == floor_bar else 0
                            if p_start < p_end:
                                out_row[self.pos_offset + p_start : self.pos_offset + p_end] = raw_row[
                                    self.pos_offset + p_start : self.pos_offset + p_end
                                ]
                                has_valid = True
                        if (
                            allow_eos
                            and self.token_offset > EOS_TOKEN_ID
                            and cb >= floor_bar
                            and cb >= min_bars_idx
                            and (ng >= min_notes_int or not has_slot)
                        ):
                            out_row[EOS_TOKEN_ID] = raw_row[EOS_TOKEN_ID]
                            has_valid = True

                    elif st == self._STATE_AFTER_POS:
                        self._unmask_active_slots_row(out_row, raw_row, act_s, has_slot)
                        has_valid = True

                    elif st == self._STATE_AFTER_SLOT:
                        out_row[self.pitch_offset : self.pitch_offset + self.pitch_size] = raw_row[
                            self.pitch_offset : self.pitch_offset + self.pitch_size
                        ]
                        has_valid = True

                    elif st == self._STATE_AFTER_PITCH:
                        out_row[self.dur_offset : self.dur_offset + self.dur_size] = raw_row[
                            self.dur_offset : self.dur_offset + self.dur_size
                        ]
                        has_valid = True

                    elif st == self._STATE_AFTER_DUR:
                        out_row[self.vel_offset : vel_hi] = raw_row[self.vel_offset : vel_hi]
                        has_valid = True

                    elif st == self._STATE_AFTER_VEL:
                        p_end = last_bar_pos_limit if cb == (bar_limit - 1) else pos_limit
                        if (
                            can_emit_notes
                            and 0 <= cb < bar_limit
                            and 0 <= lp < p_end
                            and (cb > floor_bar or (cb == floor_bar and lp >= floor_pos))
                        ):
                            self._unmask_active_slots_row(out_row, raw_row, act_s, has_slot)
                            has_valid = True
                        if can_emit_notes and floor_bar <= cb < bar_limit:
                            min_p_bar = floor_pos if cb == floor_bar else 0
                            p_start = max(lp + 1, min_p_bar) if monotonic_time else min_p_bar
                            if p_start < p_end:
                                out_row[self.pos_offset + p_start : self.pos_offset + p_end] = raw_row[
                                    self.pos_offset + p_start : self.pos_offset + p_end
                                ]
                                has_valid = True
                        if cb + 1 < bar_limit:
                            out_row[self.bar_token_id] = raw_row[self.bar_token_id]
                            has_valid = True
                        if (
                            allow_eos
                            and self.token_offset > EOS_TOKEN_ID
                            and cb >= floor_bar
                            and cb >= min_bars_idx
                            and (ng >= min_notes_int or not has_slot)
                        ):
                            out_row[EOS_TOKEN_ID] = raw_row[EOS_TOKEN_ID]
                            has_valid = True

                    if not has_valid:
                        if allow_eos and self.token_offset > EOS_TOKEN_ID:
                            out_row[EOS_TOKEN_ID] = raw_row[EOS_TOKEN_ID]
                        else:
                            halted_list[b_i] = True
                            out_row[PAD_TOKEN_ID] = 0.0

                next_tok = self._sample_from_logits(
                    constrained, temperature=temperature, top_k=top_k, top_p=top_p
                )  # [B]
                next_tok_cpu = next_tok.detach().cpu().tolist()

                for b_i in range(bsz):
                    if not active_b[b_i] or halted_list[b_i]:
                        continue
                    sampled_tok = int(next_tok_cpu[b_i])
                    gen_lists[b_i].append(next_tok[b_i : b_i + 1])
                    c_pos = cur_pos_list[b_i]

                    if self.use_instrument_tokens and 1 <= c_pos < prefix_len:
                        s_idx = c_pos - 1
                        if 0 <= s_idx < self.slot_size:
                            active_slots_list[b_i][s_idx] = (sampled_tok != INST_EMPTY_TOKEN_ID)
                    else:
                        if sampled_tok == EOS_TOKEN_ID:
                            finished_list[b_i] = True
                        elif sampled_tok == self.bar_token_id:
                            cur_bar_list[b_i] += 1
                            last_pos_list[b_i] = -1
                            state_list[b_i] = self._STATE_AFTER_BAR
                        elif self.pos_offset <= sampled_tok < self.slot_offset:
                            if cur_bar_list[b_i] < 0:
                                cur_bar_list[b_i] = 0
                            last_pos_list[b_i] = sampled_tok - self.pos_offset
                            state_list[b_i] = self._STATE_AFTER_POS
                        elif self.slot_offset <= sampled_tok < self.pitch_offset:
                            state_list[b_i] = self._STATE_AFTER_SLOT
                        elif self.pitch_offset <= sampled_tok < self.pitch_offset + self.pitch_size:
                            state_list[b_i] = self._STATE_AFTER_PITCH
                        elif self.dur_offset <= sampled_tok < self.vel_offset:
                            state_list[b_i] = self._STATE_AFTER_DUR
                        elif self.vel_offset <= sampled_tok < vel_hi:
                            state_list[b_i] = self._STATE_AFTER_VEL
                            notes_gen_list[b_i] += 1

                    cur_pos_list[b_i] = c_pos + 1

                still_active = any(
                    (not finished_list[b_i])
                    and (not halted_list[b_i])
                    and not (
                        notes_gen_list[b_i] >= max_notes_int
                        and state_list[b_i] in (self._STATE_BEFORE_BAR, self._STATE_AFTER_BAR, self._STATE_AFTER_VEL)
                    )
                    for b_i in range(bsz)
                )
                if not still_active:
                    break

                next_logits = self._forward_decode_step(next_tok)

            for b_i in range(bsz):
                if not finished_list[b_i] and cur_bar_list[b_i] < min_bars_idx:
                    rem_bars = min_bars_idx - cur_bar_list[b_i]
                    gen_lists[b_i].append(
                        torch.full((rem_bars,), self.bar_token_id, dtype=torch.long, device=device)
                    )
                    cur_bar_list[b_i] = min_bars_idx

            if append_eos and self.token_offset > EOS_TOKEN_ID:
                eos_tok = torch.tensor([EOS_TOKEN_ID], dtype=torch.long, device=device)
                for b_i in range(bsz):
                    if not finished_list[b_i]:
                        gen_lists[b_i].append(eos_tok)

            per_seq = [torch.cat(parts, dim=0) for parts in gen_lists]
            return torch.nn.utils.rnn.pad_sequence(
                per_seq, batch_first=True, padding_value=PAD_TOKEN_ID
            )
        finally:
            self.reset_kv_cache()
            if was_training:
                self.train()

    def get_symbolic_tokenizer(self):
        if self._symbolic_tokenizer is None:
            from mortm.utils.symbolic_tokenizers import MultitrackREMITokenizer, REMIVocabSpec

            vocab_spec = REMIVocabSpec(
                token_offset=self.token_offset,
                attr_sizes=self.attr_sizes,
                max_bars=self.max_segment_bars,
            )
            self._symbolic_tokenizer = MultitrackREMITokenizer(vocab_spec=vocab_spec)
        return self._symbolic_tokenizer

    def decode_to_notes(
        self,
        seq: Tensor,
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.get_symbolic_tokenizer().decode_tokens_to_notes(
            seq, slot_codes=slot_codes, num_chunks=num_chunks
        )

    def decode_to_chunks_u8(
        self,
        seq: Tensor,
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.get_symbolic_tokenizer().decode_tokens_to_chunks_u8(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )

    def decode_to_chunks_float(
        self,
        seq: Tensor,
        num_chunks: int = 128,
        slot_codes: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return self.get_symbolic_tokenizer().decode_tokens_to_chunks_float(
            seq, num_chunks=num_chunks, slot_codes=slot_codes
        )

    def decode_to_midi(
        self,
        seq: Tensor,
        out_path: str,
        slot_codes: Optional[Sequence[int]] = None,
        num_chunks: Optional[int] = None,
    ) -> np.ndarray:
        return self.get_symbolic_tokenizer().decode_tokens_to_midi(
            seq, output_path=out_path, slot_codes=slot_codes, num_chunks=num_chunks
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
            x_idx = x.to(dtype=torch.long).clamp(min=0, max=self.args.vocab_size - 1)
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

        unbatched = (x.dim() == 1)
        if unbatched:
            x = x.unsqueeze(0)
        if x.dim() != 2:
            raise ValueError(
                f"REMITransformer expects x of shape [B, S] (or [B] in decoding), got {tuple(x.shape)}."
            )

        batch, tgt_len = x.size()
        if tgt_len == 0:
            empty_out = x.new_empty((batch, 0, self.args.vocab_size), dtype=torch.float32)
            return empty_out.squeeze(0) if unbatched else empty_out

        if padding_mask is None:
            if self.token_offset > 0:
                padding_mask = (x != PAD_TOKEN_ID)
            else:
                padding_mask = (x >= 0)
        elif padding_mask.dim() == 1:
            padding_mask = padding_mask.unsqueeze(0)

        if padding_mask.shape != (batch, tgt_len):
            raise ValueError(
                f"padding_mask shape {tuple(padding_mask.shape)} does not match x shape {(batch, tgt_len)}."
            )

        x_idx = x.to(dtype=torch.long).clamp(min=0, max=self.args.vocab_size - 1)
        x_emb = self.embedding(x_idx).to(dtype=torch.bfloat16)
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
            score = self.Wout(out.to(self.Wout.weight.dtype)).float()

        if unbatched:
            score = score.squeeze(0)
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

SYMIC_MAIN_MODEL_CONFIGS = {
    "symic_g4l4_160m": "configs/models/mortm5/symic_main/symic_g4l4_160m.json",
    "octuple_160m": "configs/models/mortm5/symic_main/octuple_160m.json",
    "remi_160m": "configs/models/mortm5/symic_main/remi_160m.json",
}

MODEL_MODE_REGISTRY: Dict[str, Callable[..., _BaseMORTM5Transformer]] = {
    "flat": AblationTransformer1,
    "grouped": AblationTransformer2,
    "octuple": OctupleTransformer,
    "remi": REMITransformer,
}


def register_mortm5_model_mode(
    mode: str,
    builder: Callable[..., _BaseMORTM5Transformer],
) -> None:
    MODEL_MODE_REGISTRY[str(mode).strip().lower()] = builder


def build_mortm5_ablation_model(
    args: MORTM5Args | str | dict = "m4_grouped_shared",
    progress=None,
) -> _BaseMORTM5Transformer:
    """
    MORTM5Args (またはプリセット名、設定辞書、JSONファイルパス) の `model_mode` に応じて
    `MODEL_MODE_REGISTRY` から対応する Transformer モデルを構築するファクトリ関数。
    """
    if not isinstance(args, MORTM5Args):
        args = MORTM5Args(args)
    mode = str(getattr(args, "model_mode", "flat")).strip().lower()
    builder = MODEL_MODE_REGISTRY.get(mode)
    if builder is None:
        supported = ", ".join(sorted(MODEL_MODE_REGISTRY.keys()))
        raise ValueError(f"Unsupported args.model_mode: {mode!r}. Supported modes: [{supported}].")
    return builder(args=args, progress=progress)

