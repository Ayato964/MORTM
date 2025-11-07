import json
from typing import List


class MORTMArgs:
    def __init__(self, json_directory: str):
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.name = "MORTM"
            self.vocab_size = data['vocab_size'] if data.get('vocab_size') else 128
            self.d_layer = data['d_layer'] if data.get('d_layer') else 12
            self.e_layer = data['e_layer'] if data.get('e_layer') else 12
            self.num_heads = data['num_heads']
            self.d_model = data['d_model']
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

            self.normalize_type = "tanh" if data.get('norm_type') is None else data['norm_type']

            self.use_lora: bool = False if data.get('use_lora') is None else data['use_lora']
            self.lora_r = data['lora_r'] if data.get('lora_r') else 8
            self.lora_alpha = data['lora_alpha'] if data.get('lora_alpha') else 16


class V_MORTMArgs(MORTMArgs):
    def __init__(self, json_directory: str):
        super().__init__(json_directory)

        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.name = "V_MORTM"
            self.vocab_size = data['vocab_size'] if data.get('vocab_size') else 128
            self.d_layer = data['d_layer'] if data.get('d_layer') else 12
            self.num_heads = data['num_heads']
            self.d_model = data['d_model']
            self.d_spect = data['d_spect'] if data.get('d_spect') else 128
            self.patch_size = data['patch_size'] if data.get('patch_size') else 4
            self.dim_feedforward = data['dim_feedforward']
            self.dropout = data['dropout']

            self.num_experts = data['num_experts'] if data.get('num_experts') else 12
            self.topk_experts = data['topk_experts'] if data.get('topk_experts') else 2
            self.num_groups = data['num_groups'] if data.get('num_groups') else 1
            self.topk_groups = data['topk_groups'] if data.get('topk_groups') else 1
            self.route_scale = data['route_scale'] if data.get('route_scale') else 1
            self.score_type = data['score_type'] if data.get('score_type') else "softmax"

            self.use_moe_decoder = True if data.get('use_moe_decoder') is None else data['use_moe_decoder']

class MORTM_LIVE_Args(MORTMArgs):
    def __init__(self, json_directory: str):
        super().__init__(json_directory)

        self.name = "MORTM_LIVE"
        self.device = "cuda"
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.name = "MORTM_LIVE"
            self.inst_list: List[int] = data['inst_list'] if data.get('inst_list') else [1, 5, 19, 27, 34, 65, 10, 74]
            self.instrument_num = data['instrument_num'] if data.get('instrument_num') else 8
            self.ticks_per_measure = data['ticks_per_measure'] if data.get('ticks_per_measure') else 96
            self.pianoroll_time_step = data['pianoroll_time_step'] if data.get('pianoroll_time_step') else 16
            self.chunk = data['chunk'] if data.get('chunk') else 16
