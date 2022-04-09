import torch
import torch.nn as nn

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GibbsSampler:
    __version__ = 1.0 # generator
    def __init__(self, latent_dims, device=_device, dtype=torch.float32):
        self.latent_dims = latent_dims
        self.device = device
        self.dtype = dtype
        self.NONES = [None] * len(self.latent_dims)
    
    def var_calc(self, T2, nu2, t2):
        if nu2 is not None:
            return -torch.reciprocal(2 * (t2 + nu2 + T2))
        else:
            return -torch.reciprocal(2 * (t2 + T2))
    
    def mean_calc(self, T1, var, nu1, t1):
        if nu1 is not None:
            return var * (T1 + nu1 + t1)
        else:
            return var * (T1 + t1)
        
    def value_calc(self, z, G, nu1, nu2, t1, t2):
        T = torch.cat((z, torch.square(z)), 1)
        Tp = torch.matmul(T, G)
        mid = G.shape[1] // 2
        
        var = self.var_calc(Tp[:, mid:], nu2, t2)
        mean = self.mean_calc(Tp[:, :mid], var, nu1, t1)
        # assert (var >= 0).all()
        return mean + torch.sqrt(var) * torch.randn_like(var)
        
    def init_z(self, nu1, nu2, mu, var, batch_size):
        z = []
        for i in (0, 1):
            if (nu1[i] is None) and (nu2[i] is None) and \
               (mu[i] is None) and (var[i] is None): # all none
                if batch_size is None:
                    raise RuntimeError('batch_size must be specified if z is not given.')
                z.append(torch.randn(batch_size, self.latent_dims[i]))
            
            else:
                if (mu[i] is None) or (var[i] is None):
                    var_i = -1/(2 * nu2[i])
                    mu_i = nu1[i] / var_i
                assert (var_i > 0).all()
                z.append(mu_i + torch.sqrt(var_i) * torch.randn_like(var_i))
            z[i] = z[i].to(self.device, self.dtype).detach()
        return z
    
#     def init_z(self, nu1=None, nu2=None, mu=None, var=None, batch_size=None):
#         if (nu1 is None) and (nu2 is None) and (mu is None) and (var is None): # all none
#             if batch_size is None:
#                 raise RuntimeError('batch_size must be specified if z is not given.')
#             z = [torch.randn(batch_size, ld).to(self.device, self.dtype).detach() \
#                  for ld in self.latent_dims]
#             return z
        
#         # else
#         if (mu is None) or (var is None):
#             var = [-1/(2 * nu2[0]), -1/(2 * nu2[1])]
#             mu = [nu1[0] / var[0], nu1[1] / var[1]]
            
#         z = [(mu[0] + torch.sqrt(var[0]) * torch.randn_like(var[0])).detach(),
#              (mu[1] + torch.sqrt(var[1]) * torch.randn_like(var[1])).detach()]
        
#         return z
    
    def sample(self, G, z=None, nu1=None, nu2=None, mu=None, var=None, 
               t1s=None, t2s=None, n_iterations=1, batch_size=None):    
        nu1 = self.NONES if nu1 is None else nu1
        nu2 = self.NONES if nu2 is None else nu2
        mu = self.NONES if mu is None else mu
        var = self.NONES if var is None else var
        
        if z is None:
            z = self.init_z(nu1=nu1, nu2=nu2, mu=mu, var=var, batch_size=batch_size)

        # TODO: generalize to M > 2
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1], G.t(), nu1[0], nu2[0], t1s[0], t2s[0])
            z[1] = self.value_calc(z[0], G, nu1[1], nu2[1], t1s[1], t2s[1])
            
        return z
    
    
    def sample_generator(self, G, nu1=None, nu2=None, mu=None, var=None, 
                         t1s=None, t2s=None, n_iterations=1, batch_size=None):
        nu1 = self.NONES if nu1 is None else nu1
        nu2 = self.NONES if nu2 is None else nu2
        mu = self.NONES if mu is None else mu
        var = self.NONES if var is None else var
        
        if z is None:
            z = self.init_z(nu1=nu1, nu2=nu2, mu=mu, var=var, batch_size=batch_size)

        # TODO: generalize to M > 2
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1], G.t(), nu1[0], nu2[0], t1s[0], t2s[0])
            z[1] = self.value_calc(z[0], G, nu1[1], nu2[1], t1s[1], t2s[1])
            yield z
