import torch
import torch.nn.functional as F

def compute_perplexity(logits, targets, ignore_index=0):
    """
    perplexityを計算する
    logits: Tensor of shape [B, T, VocabSize]
    targets: Tensor of shape [B, T] (ground truth token IDs)
    ignore_index: index to ignore in loss (e.g., padding)
    """
    B, T, V = logits.shape
    logits = logits.view(-1, V)
    targets = targets.view(-1)

    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='mean')
    perplexity = torch.exp(loss)
    return loss.item(), perplexity.item()
