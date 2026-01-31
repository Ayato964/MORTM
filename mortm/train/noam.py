import math


def noam_lr(d_model: int, warmup_steps=4000):
    def lr_lambda(step):
        step = max(1, step)
        lr = ((d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5)))
        return lr
    return lr_lambda

def get_cosine_schedule_with_warmup(
        total_steps: int,
        warmup_ratio: float = 0.01, # 比率で指定
        min_lr_ratio: float = 0.05
):
    # total_stepsに基づいてwarmupステップ数を動的に計算
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step):
        step = max(1, step)

        # Warmup phase
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        # Post-training phase
        if step >= total_steps:
            return min_lr_ratio

        # Cosine Decay phase
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_component = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_component

    return lr_lambda