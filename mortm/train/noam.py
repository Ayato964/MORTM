
def noam_lr(d_model: int, warmup_steps=4000):
    def lr_lambda(step):
        step = max(1, step)
        lr = ((d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5)))
        return lr
    return lr_lambda
