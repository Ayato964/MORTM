import json
from typing import List, Optional


class MORTMArgs:
    def __init__(self, json_directory: str, log_scale=False):
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)

            self.name = "MORTM"
            self.vocab_size = data['vocab_size'] if data.get('vocab_size') else 128
            self.d_layer = data['d_layer'] if data.get('d_layer') else 12
            self.e_layer = data['e_layer'] if data.get('e_layer') else 12
            self.num_heads = data['num_heads']
            self.d_model: int = data['d_model']
            self.dim_feedforward = data['dim_feedforward']
            self.dropout = data['dropout']
            self.position_length = data['position_length'] if data.get('position_length') else 512
            self.min_length = data['min_length'] if data.get("min_length") else 90
            self.num_experts = data['num_experts'] if data.get('num_experts') else 12
            self.topk_experts = data['topk_experts'] if data.get('topk_experts') else 2
            self.num_groups = data['num_groups'] if data.get('num_groups') else 1
            self.topk_groups = data['topk_groups'] if data.get('topk_groups') else 1
            self.route_scale = data['route_scale'] if data.get('route_scale') else 1
            self.score_type = data['score_type'] if data.get('score_type') else "softmax"
            self.use_moe_encoder = False if data.get('use_moe_encoder') is None else data['use_moe_encoder'],
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
            self.lora_r = data['lora_r'] if data.get('lora_r') else 8
            self.lora_alpha = data['lora_alpha'] if data.get('lora_alpha') else 16

            # デバッグ用: Trueにすると観察用Attentionに切り替わりweightを保存する
            self.debug_attention: bool = False


class MORTM5Args(MORTMArgs):
    def __init__(self, json_directory: str):
        super().__init__(json_directory)

        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.name = "MORTM5"

            self.roll_w = 24 if data.get("roll_w") is None else data["roll_w"]
            self.roll_h = 128 if data.get("roll_h") is None else data["roll_h"]
            self.track_size = 4 if data.get("track_size") is None else data["track_size"]
            self.first_channel = 64 if data.get("first_channel") is None else data["first_channel"]
            self.max_channel = 128 if data.get("max_channel") is None else data["max_channel"]
            self.encoder_wout = 2048 if data.get("encoder_wout") is None else data["encoder_wout"]
            self.encoder_layer = 8 if data.get("encoder_layer") is None else data["encoder_layer"]