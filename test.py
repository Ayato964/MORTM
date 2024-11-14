import torch

def remove_subsequence_tensor(a, b):
    # a と b は 1次元のPyTorch Tensorやとするで
    len_b = b.shape[0]
    for i in range(a.shape[0] - len_b + 1):
        if torch.equal(a[i:i+len_b], b):
            return torch.cat((a[:i], a[i+len_b:]))
    return a

a = torch.tensor([4,5,1, 2, 3, 4,2, 1])
b = torch.tensor([1,2,3])
result = remove_subsequence_tensor(a, b)
print(result)  # 出力: tensor([4,5,1,4,2])