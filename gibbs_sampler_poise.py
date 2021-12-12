import torch
import torch.nn as nn
_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class gibbs_sampler():
    def __init__(self, latent_dims, batch_size):
        self.latent_dims = latent_dims
        self.batch_size = batch_size

    def var_calc(self,z,g22,lambda_2):
        val   = 2*(1-torch.matmul(torch.square(z),g22)-lambda_2)
        return torch.reciprocal(val)

    def mean_calc(self,z,var,g11,lambda_1):
        beta = torch.matmul(z,g11)+lambda_1
        return var*beta

    def value_calc(self,z,g11,g22,lambda_1,lambda_2):
        var1          = self.var_calc(z,g22,lambda_2)
        mean1         = self.mean_calc(z,var1,g11,lambda_1)
        out           = mean1+torch.sqrt(var1.float())*torch.randn_like(var1)
        return out

    def gibbs_sample(self, g11,g22, z=None, lambda1s=None, lambda2s=None, n_iterations=1):
        # TODO: ld is n-tuple-- if generate z in the unsqueezed way: can the matmul still be valid?
        if z is None:
            z = [torch.randn(self.batch_size, ld)squeeze().to(_device) for ld in self.latent_dims]
        if lambda1s is None:
            lambda1s = [0 for _ in range(len(self.latent_dims))]
        if lambda2s is None:
            lambda2s = [0 for _ in range(len(self.latent_dims))] 

        # TODO: generic 
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1],torch.transpose(g11,0,1),torch.transpose(g22,0,1),
                                   lambda1s[0],lambda2s[0]) 
            z[1] = self.value_calc(z[0],g11,g22,lambda1s[0],lambda2s[0])

        return z
    
"""
Size of z1:        [batch_size, latent_dim1]
Size of z2:        [batch_size, latent_dim1]
Size of g11:       [latent_dim1, latent_dim2]
Size of g22:       [latent_dim1, latent_dim2]
Size of lambda_1:  [batch_size, latent_dim1]
Size of lambda_2:  [batch_size, latent_dim1]
Size of lambdap_1: [batch_size, latent_dim2]
Size of lambdap_2: [batch_size, latent_dim2]

"""
