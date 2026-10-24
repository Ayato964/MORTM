import copy
import json
import os
from typing import Any, Dict, List, Optional, Union


def _make_ablation_preset(
    name: str,
    tokenizer_key: str,
    model_mode: str,
    separate_rvq_layers: bool,
    vq_groups: int,
    vq_layers: int,
    encoder_wout: int,
    vocab_size: int,
    position_length: int,
) -> Dict[str, Any]:
    return {
        "name": name,
        "tokenizer_key": tokenizer_key,
        "model_mode": model_mode,
        "separate_rvq_layers": separate_rvq_layers,
        "use_instrument_tokens": True,
        "token_offset": 133,
        "vocab_size": vocab_size,
        "vq_groups": vq_groups,
        "vq_layers": vq_layers,
        "encoder_layer": vq_layers,
        "encoder_wout": encoder_wout,
        "codebook_size": 2048,
        "d_model": 768,
        "d_layer": 12,
        "num_heads": 12,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "position_length": position_length,
        "norm_type": "rmsnorm",
        "use_rope": True,
        "use_silu": True,
        "use_bias": False,
        "use_moe_decoder": False,
    }


MORTM5_ABLATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "m3_flat_shared": _make_ablation_preset(
        "MORTM5_M3_Flat_Shared", "m3_g8l2", "flat", False, 8, 2, 256, 16517, 4096
    ),
    "m3_flat_separated": _make_ablation_preset(
        "MORTM5_M3_Flat_Separated", "m3_g8l2", "flat", True, 8, 2, 256, 32901, 4096
    ),
    "m3_grouped_shared": _make_ablation_preset(
        "MORTM5_M3_Grouped_Shared", "m3_g8l2", "grouped", False, 8, 2, 256, 16517, 1024
    ),
    "m3_grouped_separated": _make_ablation_preset(
        "MORTM5_M3_Grouped_Separated", "m3_g8l2", "grouped", True, 8, 2, 256, 32901, 1024
    ),
    "m4_flat_shared": _make_ablation_preset(
        "MORTM5_M4_Flat_Shared", "m4_g4l4", "flat", False, 4, 4, 128, 8325, 4096
    ),
    "m4_flat_separated": _make_ablation_preset(
        "MORTM5_M4_Flat_Separated", "m4_g4l4", "flat", True, 4, 4, 128, 32901, 4096
    ),
    "m4_grouped_shared": _make_ablation_preset(
        "MORTM5_M4_Grouped_Shared", "m4_g4l4", "grouped", False, 4, 4, 128, 8325, 2048
    ),
    "m4_grouped_separated": _make_ablation_preset(
        "MORTM5_M4_Grouped_Separated", "m4_g4l4", "grouped", True, 4, 4, 128, 32901, 2048
    ),
}

_MORTM5_PRESET_ALIASES: Dict[str, str] = {
    "#1": "m3_flat_shared",
    "1": "m3_flat_shared",
    "#2": "m3_flat_separated",
    "2": "m3_flat_separated",
    "#3": "m3_grouped_shared",
    "3": "m3_grouped_shared",
    "#4": "m3_grouped_separated",
    "4": "m3_grouped_separated",
    "#5": "m4_flat_shared",
    "5": "m4_flat_shared",
    "#6": "m4_flat_separated",
    "6": "m4_flat_separated",
    "#7": "m4_grouped_shared",
    "7": "m4_grouped_shared",
    "#8": "m4_grouped_separated",
    "8": "m4_grouped_separated",
}

_MORTM5_ABLATION_SUBDIR_MAP: Dict[str, str] = {
    "#1": "1", "1": "1", "m3_flat_shared": "1", "mortm5_m3_flat_shared": "1",
    "#2": "2", "2": "2", "m3_flat_separated": "2", "mortm5_m3_flat_separated": "2",
    "#3": "3", "3": "3", "m3_grouped_shared": "3", "mortm5_m3_grouped_shared": "3",
    "#4": "4", "4": "4", "m3_grouped_separated": "4", "mortm5_m3_grouped_separated": "4",
    "#5": "5", "5": "5", "m4_flat_shared": "5", "mortm5_m4_flat_shared": "5",
    "#6": "6", "6": "6", "m4_flat_separated": "6", "mortm5_m4_flat_separated": "6",
    "#7": "7", "7": "7", "m4_grouped_shared": "7", "mortm5_m4_grouped_shared": "7",
    "#8": "8", "8": "8", "m4_grouped_separated": "8", "mortm5_m4_grouped_separated": "8",
}
ABLATION_SUBDIR_MAP = _MORTM5_ABLATION_SUBDIR_MAP


def resolve_ablation_subdir(model_name: str = "", ablation_id: str = "") -> str:
    """アブレーション識別子 ('#1'〜'#8', '1'〜'8') またはモデル名から保存サブディレクトリ名 ('1'〜'8') を解決する。"""
    for raw in (ablation_id, model_name):
        key = str(raw or "").strip().lower()
        if not key:
            continue
        if key in _MORTM5_ABLATION_SUBDIR_MAP:
            return _MORTM5_ABLATION_SUBDIR_MAP[key]
        stripped = key.lstrip("#")
        if stripped in {"1", "2", "3", "4", "5", "6", "7", "8"}:
            return stripped
    return "default"


def resolve_mortm_config_dict(source: Union[str, os.PathLike, Dict[str, Any]]) -> Dict[str, Any]:
    """辞書、内蔵アブレーションプリセット名 ('m4_grouped_shared' / '#7' 等)、または JSON パスから設定辞書を解決する。"""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    key = str(source).strip()
    lower_key = key.lower()
    if lower_key in _MORTM5_PRESET_ALIASES:
        lower_key = _MORTM5_PRESET_ALIASES[lower_key]
    if lower_key in MORTM5_ABLATION_PRESETS and not os.path.isfile(key):
        return copy.deepcopy(MORTM5_ABLATION_PRESETS[lower_key])
    with open(source, "r") as f:
        return json.load(f)


class MORTMArgs:
    def __init__(self, json_directory: Union[str, os.PathLike, Dict[str, Any]], log_scale=False):
        data: dict = resolve_mortm_config_dict(json_directory)
        self._raw_config: dict = copy.deepcopy(data)

        self.name = "MORTM"
        self.vocab_size = data['vocab_size'] if data.get('vocab_size') else 128
        self.d_layer = data['d_layer'] if data.get('d_layer') else 12
        self.e_layer = data['e_layer'] if data.get('e_layer') else 12
        self.num_heads = data['num_heads'] if data.get('num_heads') else 12
        self.d_model: int = data['d_model'] if data.get('d_model') else 768
        self.dim_feedforward = data['dim_feedforward'] if data.get('dim_feedforward') else 2048
        self.dropout = 0.1 if data.get('dropout') is None else float(data['dropout'])
        self.position_length = data['position_length'] if data.get('position_length') else 512
        self.min_length = data['min_length'] if data.get("min_length") else 90
        self.num_experts = data['num_experts'] if data.get('num_experts') else 12
        self.topk_experts = data['topk_experts'] if data.get('topk_experts') else 2
        self.num_groups = data['num_groups'] if data.get('num_groups') else 1
        self.topk_groups = data['topk_groups'] if data.get('topk_groups') else 1
        self.route_scale = data['route_scale'] if data.get('route_scale') else 1
        self.score_type = data['score_type'] if data.get('score_type') else "softmax"
        self.use_moe_encoder = False if data.get('use_moe_encoder') is None else data['use_moe_encoder']
        self.use_moe_decoder = True if data.get('use_moe_decoder') is None else data['use_moe_decoder']
        self.is_not_flash = data.get("is_not_flash")
        self.use_silu = False if data.get('use_silu') is None else data['use_silu']
        self.use_rope = False if data.get('use_rope') is None else data['use_rope']
        self.use_cross_attention = False if data.get('use_cross_attention') is None else data['use_cross_attention']
        self.use_bias = True if data.get('use_bias') is None else data['use_bias']
        self.use_gate_bias = True if data.get('use_gate_bias') is None else data['use_gate_bias']

        self.normalize_type = "tanh" if data.get('norm_type') is None else data['norm_type']

        self.use_attn_lora: bool = False if data.get('use_attn_lora') is None else data['use_attn_lora']
        self.use_gate_lora: bool = False if data.get('use_gate_lora') is None else data['use_gate_lora']
        self.use_ffn_lora: bool = False if data.get('use_ffn_lora') is None else data['use_ffn_lora']
        _shared_lora = data.get('use_shared_expert_lora')
        if _shared_lora is None:
            _shared_lora = data.get('use_shared_ffn_lora', False)
        self.use_shared_expert_lora: bool = bool(_shared_lora)
        self.lora_r = data['lora_r'] if data.get('lora_r') else 8
        self.lora_alpha = data['lora_alpha'] if data.get('lora_alpha') else 16

        # デバッグ用: Trueにすると観察用Attentionに切り替わりweightを保存する
        self.debug_attention: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """設定を JSON シリアライズ可能な辞書として返す。"""
        out = copy.deepcopy(getattr(self, "_raw_config", {}))
        out.update({
            "name": self.name,
            "vocab_size": int(self.vocab_size),
            "d_layer": int(self.d_layer),
            "e_layer": int(self.e_layer),
            "num_heads": int(self.num_heads),
            "d_model": int(self.d_model),
            "dim_feedforward": int(self.dim_feedforward),
            "dropout": float(self.dropout),
            "position_length": int(self.position_length),
            "min_length": int(self.min_length),
            "norm_type": str(self.normalize_type),
            "use_rope": bool(self.use_rope),
            "use_silu": bool(self.use_silu),
            "use_bias": bool(self.use_bias),
            "use_moe_decoder": bool(self.use_moe_decoder),
        })
        return out


PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
RVQ_TOKEN_OFFSET = 3

# 楽器条件付け（4スロット: Slot 0=Drums, Slot 1..3=Melodic Tracks）用の特殊トークン定数
INST_PROGRAM_OFFSET = 3          # MIDI Program 0..127 -> Token ID 3..130 (<INST_0> .. <INST_127>)
INST_DRUMS_TOKEN_ID = 131        # Drums Slot -> Token ID 131 (<INST_DRUMS>)
INST_EMPTY_TOKEN_ID = 132        # Empty Slot -> Token ID 132 (<INST_EMPTY>)
NUM_SLOT_TOKENS = 4              # Slot 0, Slot 1, Slot 2, Slot 3
INST_RVQ_TOKEN_OFFSET = 133      # 楽器トークン有効時の RVQ 開始オフセット (3 + 130)

SLOT_CODE_DRUMS = 128            # slots.npy 内での Drums 識別コード
SLOT_CODE_EMPTY = -1             # slots.npy 内での空スロット識別コード


def slot_code_to_token_id(code: int) -> int:
    """slots.npy のスロットコード (-1: 空, 0..127: Program, 128: Drums) を特殊トークンID (3..132) に変換する。"""
    c = int(code)
    if c == SLOT_CODE_DRUMS:
        return INST_DRUMS_TOKEN_ID
    if 0 <= c <= 127:
        return INST_PROGRAM_OFFSET + c
    if c == SLOT_CODE_EMPTY:
        return INST_EMPTY_TOKEN_ID
    raise ValueError(f"Invalid slot code {code}: expected -1 (empty), 0..127 (MIDI program), or 128 (drums).")


def token_id_to_slot_code(token_id: int) -> int:
    """楽器特殊トークンID (3..132) をスロットコード (-1, 0..127, 128) に逆変換する。"""
    tid = int(token_id)
    if tid == INST_DRUMS_TOKEN_ID:
        return SLOT_CODE_DRUMS
    if tid == INST_EMPTY_TOKEN_ID:
        return SLOT_CODE_EMPTY
    if INST_PROGRAM_OFFSET <= tid < INST_PROGRAM_OFFSET + 128:
        return tid - INST_PROGRAM_OFFSET
    raise ValueError(
        f"Token ID {token_id} is not a valid instrument token in [{INST_PROGRAM_OFFSET}, {INST_EMPTY_TOKEN_ID}]."
    )


class MORTM5Args(MORTMArgs):
    def __init__(self, json_directory: Union[str, os.PathLike, Dict[str, Any]] = "m4_grouped_shared"):
        super().__init__(json_directory)

        data: dict = resolve_mortm_config_dict(json_directory)
        self.name = str(data.get("name", "MORTM5"))
        self.tokenizer_key: Optional[str] = data.get("tokenizer_key")

        # MORTM5 の Transformer デコーダ標準設定（JSON未指定時は RMSNorm + RoPE + SwiGLU Dense を既定とする）
        if data.get("use_moe_decoder") is None:
            self.use_moe_decoder = False
        if data.get("norm_type") is None:
            self.normalize_type = "rmsnorm"
        if data.get("use_rope") is None:
            self.use_rope = True
        if data.get("use_silu") is None:
            self.use_silu = True
        if data.get("use_bias") is None:
            self.use_bias = False

        self.roll_w = 24 if data.get("roll_w") is None else int(data["roll_w"])
        self.roll_h = 128 if data.get("roll_h") is None else int(data["roll_h"])
        self.track_size = 4 if data.get("track_size") is None else int(data["track_size"])
        self.first_channel = 64 if data.get("first_channel") is None else int(data["first_channel"])
        self.max_channel = 256 if data.get("max_channel") is None else int(data["max_channel"])
        self.bottle_channel = 128 if data.get("bottle_channel") is None else int(data["bottle_channel"])
        self.encoder_wout = 512 if data.get("encoder_wout") is None else int(data["encoder_wout"])
        self.vq_groups = 4 if data.get("vq_groups") is None else int(data["vq_groups"])
        self.encoder_layer = 2 if data.get("encoder_layer") is None else int(data["encoder_layer"])
        self.vq_layers = int(data.get("vq_layers", self.encoder_layer))
        self.codebook_size = 1024 if data.get("codebook_size") is None else int(data["codebook_size"])
        self.use_pre_vq_norm = True if data.get("use_pre_vq_norm") is None else bool(data["use_pre_vq_norm"])

        self.pad_token_id = PAD_TOKEN_ID
        self.bos_token_id = BOS_TOKEN_ID
        self.eos_token_id = EOS_TOKEN_ID
        if data.get("position_length") is None:
            self.position_length = 4096

        self.model_mode: str = str(data.get("model_mode", "flat"))
        if self.model_mode not in ("flat", "grouped"):
            raise ValueError(f"model_mode must be 'flat' or 'grouped', got {self.model_mode}")

        self.separate_rvq_layers: bool = bool(data.get("separate_rvq_layers", False))
        self.use_instrument_tokens: bool = bool(
            data.get("use_instrument_tokens", data.get("token_offset") == INST_RVQ_TOKEN_OFFSET)
        )
        self._explicit_token_offset: Optional[int] = (
            int(data["token_offset"]) if data.get("token_offset") is not None else None
        )
        self._explicit_vocab_size: Optional[int] = (
            int(data["vocab_size"]) if data.get("vocab_size") is not None else None
        )
        self.recompute_vocab()

    def recompute_vocab(self) -> int:
        """separate_rvq_layers / use_instrument_tokens の変更に応じて token_offset と vocab_size を再計算して返す。"""
        layer_factor = self.vq_layers if getattr(self, "separate_rvq_layers", False) else 1
        self.rvq_vocab_size = int(layer_factor * self.vq_groups * self.codebook_size)

        if getattr(self, "use_instrument_tokens", False):
            self.token_offset = (
                max(INST_RVQ_TOKEN_OFFSET, int(self._explicit_token_offset))
                if getattr(self, "_explicit_token_offset", None) is not None
                else INST_RVQ_TOKEN_OFFSET
            )
        elif getattr(self, "_explicit_token_offset", None) is not None and int(self._explicit_token_offset) < INST_RVQ_TOKEN_OFFSET:
            self.token_offset = int(self._explicit_token_offset)
        elif getattr(self, "_explicit_vocab_size", None) is not None and not getattr(self, "separate_rvq_layers", False):
            self.token_offset = max(0, int(self._explicit_vocab_size) - self.rvq_vocab_size)
        else:
            self.token_offset = RVQ_TOKEN_OFFSET

        if (
            getattr(self, "_explicit_vocab_size", None) is not None
            and not getattr(self, "separate_rvq_layers", False)
            and not getattr(self, "use_instrument_tokens", False)
        ):
            self.vocab_size = int(self._explicit_vocab_size)
        else:
            self.vocab_size = int(self.token_offset + self.rvq_vocab_size)
        return self.vocab_size

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "name": self.name,
            "tokenizer_key": self.tokenizer_key,
            "model_mode": self.model_mode,
            "separate_rvq_layers": bool(self.separate_rvq_layers),
            "use_instrument_tokens": bool(self.use_instrument_tokens),
            "token_offset": int(self.token_offset),
            "vocab_size": int(self.vocab_size),
            "vq_groups": int(self.vq_groups),
            "vq_layers": int(self.vq_layers),
            "encoder_layer": int(self.encoder_layer),
            "encoder_wout": int(self.encoder_wout),
            "codebook_size": int(self.codebook_size),
            "roll_w": int(self.roll_w),
            "roll_h": int(self.roll_h),
            "track_size": int(self.track_size),
            "first_channel": int(self.first_channel),
            "max_channel": int(self.max_channel),
            "bottle_channel": int(self.bottle_channel),
            "use_pre_vq_norm": bool(self.use_pre_vq_norm),
        })
        return out