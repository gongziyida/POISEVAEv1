import torch
import torch.nn as nn

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class gibbs_sampler():
    __version__ = 1.0
    
    def __init__(self, latent_dims, batch_size, device=_device):
        self.latent_dims = latent_dims
        self.batch_size = batch_size
        self.device = device

    def var_calc(self,z, g22, lambda_2):
        val = 1 - torch.matmul(torch.square(z), g22)
        if lambda_2 is not None:
            val -= lambda_2
        return torch.reciprocal(2 * val)

    def mean_calc(self, z, var, g11, lambda_1):
        beta = torch.matmul(z, g11)
        if lambda_1 is not None:
            beta += lambda_1
        return var * beta

    def value_calc(self,z, g11, g22, lambda_1, lambda_2):
        var1 = self.var_calc(z, g22, lambda_2)
        mean1 = self.mean_calc(z, var1, g11, lambda_1)
        out = mean1 + torch.sqrt(var1.float()) * torch.randn_like(var1)
        return out

    def sample(self, g11, g22, z=None, lambda1s=None, lambda2s=None, n_iterations=1):
        """
        g11, g22: 
            Diagonal blocks of the metric tensor
        z: 
            If not provided, randomly initialize
        lambda1s: optional
            Natural parameter 1 of the latent distributions
            If not provided, treat as zeros
        lambda1s: optional
            Natural parameter 2 of the latent distributions
            If not provided, treat as zeros
        n_iterations: int, optional; default 1
        """
            # TODO: function signature of gibbs_sample: optional parameters
            # flag_init. not necessary; if z not provided, init. z rand.ly
            # Not really an optimization but make the code clear
            # in case people want to look carefully in the future
            # I made an attempt in the local file `gibbs_sampler_poise.py`; debugging needed
        if z is None:
            z = [torch.randn(self.batch_size, ld).squeeze().to(self.device) 
                 for ld in self.latent_dims]
        if lambda1s is None:
            lambda1s = [None for _ in range(len(self.latent_dims))]
        if lambda2s is None:
            lambda2s = [None for _ in range(len(self.latent_dims))]

        # TODO: make it generic for > 2 latent spaces 
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1], torch.transpose(g11,0,1), torch.transpose(g22,0,1),
                                   lambda1s[0], lambda2s[0]) 
            z[1] = self.value_calc(z[0], g11, g22, lambda1s[1], lambda2s[1])

        return z