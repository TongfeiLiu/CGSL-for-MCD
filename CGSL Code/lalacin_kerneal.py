import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def laplacian_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X
    if gamma is None:
        gamma = 1.0 / X.size(1)
    K = -gamma * torch.cdist(X, Y, p=1)
    torch.exp(K, out=K)  # exponentiate K in-place
    return K
