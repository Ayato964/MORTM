import json


class MORTMArgs:
    def __init__(self, json_directory: str):
        with open(json_directory, 'r') as f:
            data: dict = json.load(f)
            self.vocab_size = data['vocab_size']
            self.d_layer = data['d_layer'] if data.get('d_layer') else 12
            self.e_layer = data['e_layer'] if data.get('e_layer') else 12
            self.num_heads = data['num_heads']
            self.d_model = data['d_model']
            self.dim_feedforward = data['dim_feedforward']
            self.dropout = data['dropout']
            self.position_length = data['position_length']
            self.num_experts = data['num_experts'] if data.get('num_experts') else 12
            self.topk_experts = data['topk_experts'] if data.get('topk_experts') else 2
            self.num_groups = data['num_groups'] if data.get('num_groups') else 1
            self.topk_groups = data['topk_groups'] if data.get('topk_groups') else 1
            self.route_scale = data['route_scale'] if data.get('route_scale') else 1
            self.score_type = data['score_type'] if data.get('score_type') else "softmax"
