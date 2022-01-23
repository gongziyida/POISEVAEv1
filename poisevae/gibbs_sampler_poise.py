import torch
import torch.nn as nn

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GibbsSampler_A:
    def __init__(self,latent_dim1, latent_dim2, batch_size):
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.batch_size = batch_size
        
    def var_calc(self,z,g22,g21,lambda_2):
        val   = 2*(1-torch.matmul(torch.square(z),g22)-torch.matmul(z,g21)-lambda_2)
        return torch.reciprocal(val)
    
    def mean_calc(self,z,var,g11,g12,lambda_1):
        beta = torch.matmul(z,g11)+torch.matmul(torch.square(z),g12)+lambda_1
        return var*beta
    
    def value_calc(self,z,g11,g22,g12,g21,lambda_1,lambda_2):
        var1          = self.var_calc(z,g22,g21,lambda_2)
        mean1         = self.mean_calc(z,var1,g11,g12,lambda_1)
        out           = mean1+torch.sqrt(var1)*torch.randn_like(var1)
        return out
    
    def sample(self,flag,z1,z2,g11,g22,g12,g21,lambda_1,lambda_2,lambdap_1,lambdap_2,n_iterations):
        if flag == 1:
            z1 = torch.randn(self.batch_size,self.latent_dim1).to(_device) 
            z2 = torch.randn(self.batch_size,self.latent_dim2).to(_device) 
            
        for i in range(n_iterations):
            z1  = self.value_calc(z2,g11.t(),g22.t(),g12.t(),g21.t(),lambda_1,lambda_2)
            z2  = self.value_calc(z1,g11,g22,g21,g12,lambdap_1,lambdap_2) 

        return z1,z2
    
class GibbsSampler_Z:
    __version__ = 4.0 # t1 / t2
    def __init__(self, latent_dims, device=_device):
        self.latent_dims = latent_dims
        self.device = device
        self.NONES = [None] * len(self.latent_dims)
    
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
        return mean + torch.sqrt(var) * torch.randn_like(var)
        
    def sample(self, G, z=None, lambda1s=None, lambda2s=None, t1s=None, t2s=None, 
               n_iterations=1, batch_size=None):

        if z is None:
            if batch_size is None:
                raise RuntimeError('batch_size must be specified if z is not given.')
            z = [torch.randn(batch_size, ld).to(self.device, G.dtype).detach() for ld in self.latent_dims]

        if lambda1s is None:
            lambda1s = self.NONES
        if lambda2s is None:
            lambda2s = self.NONES

        # TODO: generalize to M > 2
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1], G.t(), lambda1s[0], lambda2s[0], t1s[0], t2s[0])
            z[1] = self.value_calc(z[0], G, lambda1s[1], lambda2s[1], t1s[1], t2s[1])
            
        return z