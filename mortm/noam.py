import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR


def noam_lr(d_model: int, warmup_steps=4000):
    def lr_lambda(step):
        step = max(1, step)
        lr = ((d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))) * 100
        print(f"steps:{step}  lr:{lr}")
        #(d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))
        return lr
    return lr_lambda
