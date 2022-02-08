import torch
from torchviz import make_dot

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KLD:
    __version__ = 3.0
    
    def __init__(self, latent_dims, reduction, device=_device):
        self.latent_dims = latent_dims
        self.reduction = reduction
        self.device = device
        
    def var_calc(self, T2, lambda2, t2):
        if lambda2 is not None:
            return -torch.reciprocal(2 * (t2 + lambda2 + T2))
        else:
            return -torch.reciprocal(2 * (t2 + T2))
    
    def mean_calc(self, T1, var, lambda1, t1):
        if lambda1 is not None:
            return var * (T1 + lambda1 + t1)
        else:
            return var * (T1 + t1)
        
    def value_calc(self, z, G, lambda1, lambda2, t1, t2):
        T = torch.cat((z, torch.square(z)), 1)
        Tp = torch.matmul(T, G)
        mid = G.shape[1] // 2
        
        var = self.var_calc(Tp[:, mid:], lambda2, t2)
        mean = self.mean_calc(Tp[:, :mid], var, lambda1, t1)
        return torch.square(mean) + var - torch.log(var) - 1
        
    def calc(self, G, z, z_priors, lambda1s, lambda2s):
        part0 = self.value_calc(z[1], G.t(), lambda1s[0], lambda2s[0], 0, -1).sum(dim=1)
        part1 = self.value_calc(z[0], G, lambda1s[1], lambda2s[1], 0, -1).sum(dim=1)
        res = (part0 + part1) * 0.5
        if self.reduction == 'sum':
            return res.sum()
        elif self.reduction == 'mean':
            return res.mean()
        else:
            return res
    
    