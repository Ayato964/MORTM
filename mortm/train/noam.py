import math


def noam_lr(d_model: int, warmup_steps=4000):
    def lr_lambda(step):
        step = max(1, step)
        lr = ((d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5)))
        return lr
    return lr_lambda


def get_cosine_schedule_with_warmup(
        total_steps: int,
        warmup_steps: int = 1000,
        min_lr_ratio: float = 0.05,
        stable_steps: int = 0,
):
    """
    Linear Warmup + (任意: Stable Plateau) + Cosine Annealing Decay スケジューラ。
    - step < warmup_steps: 0 -> 1.0 へ線形ウォームアップ
    - warmup_steps <= step < warmup_steps + stable_steps: 1.0 を維持 (WSD モード; 既定 stable_steps=0 では即減衰開始)
    - warmup_steps + stable_steps <= step < total_steps: 1.0 -> min_lr_ratio へコサイン減衰
    - step >= total_steps: min_lr_ratio を維持
    """
    def lr_lambda(step):
        step = max(1, step)

        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        decay_start = warmup_steps + max(0, int(stable_steps))
        if step < decay_start:
            return 1.0

        if step >= total_steps:
            return min_lr_ratio

        progress = float(step - decay_start) / float(max(1, total_steps - decay_start))
        cosine_component = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_component

    return lr_lambda