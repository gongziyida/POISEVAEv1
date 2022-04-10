import torch

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_posterior(nu1, nu2, mu, var, enc_config):
    if enc_config == 'mu/var':
        if mu is not None and var is not None:
            nu1, nu2 = [None, None], [None, None]
            
            for i in (0, 1):
                if mu[i] is not None and var[i] is not None:
                    nu1[i] = mu[i] / var[i]
                    nu2[i] = -1 / (2 * var[i])
            return nu1, nu2, mu, var

    elif enc_config == 'nu':
        if nu1 is not None and nu2 is not None:
            mu, var = [None, None], [None, None]
            
            for i in (0, 1):
                if nu1[i] is not None and nu2[i] is not None:
                    var[i] = -1 / (2 * nu2[i])
                    mu[i] = nu1[i] * var[i]
            return nu1, nu2, mu, var
    
    # Prior
    return [None, None], [None, None], [None, None], [None, None]

class GibbsSampler:
    __version__ = 1.0 # generator
    def __init__(self, latent_dims, enc_config, device=_device, dtype=torch.float32):
        self.latent_dims = latent_dims
        self.enc_config = enc_config
        self.device = device
        self.dtype = dtype
    
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
        
            
    def init_z(self, mu, var, batch_size):
        z = []
        for ld, mu_i, var_i in zip(self.latent_dims, mu, var):
            if (mu_i is None) and (var_i is None): # all none
                if batch_size is None:
                    raise RuntimeError('batch_size must be specified for prior.')
                z.append(torch.randn(batch_size, ld).to(self.device, self.dtype).detach())
            
            else:
                if not (var_i > 0).all():
                    raise ValueError('Invalid variance')
                z.append(mu_i + torch.sqrt(var_i) * torch.randn_like(var_i).detach())
        return z
    
    
    def sample(self, G, nu1=None, nu2=None, mu=None, var=None, 
               t1s=None, t2s=None, n_iterations=1, batch_size=None):
        nu1, nu2, mu, var = init_posterior(nu1, nu2, mu, var, self.enc_config)
            
        z = self.init_z(mu=mu, var=var, batch_size=batch_size)

        # TODO: generalize to M > 2
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1], G.t(), nu1[0], nu2[0], t1s[0], t2s[0])
            z[1] = self.value_calc(z[0], G, nu1[1], nu2[1], t1s[1], t2s[1])
            
        return z
    
    
    def sample_generator(self, G, nu1=None, nu2=None, mu=None, var=None, 
                         t1s=None, t2s=None, n_iterations=1, batch_size=None):
        nu1, nu2, mu, var = init_posterior(nu1, nu2, mu, var, self.enc_config)
        
        z = self.init_z(mu=mu, var=var, batch_size=batch_size)

        # TODO: generalize to M > 2
        for i in range(n_iterations):
            z[0] = self.value_calc(z[1], G.t(), nu1[0], nu2[0], t1s[0], t2s[0])
            z[1] = self.value_calc(z[0], G, nu1[1], nu2[1], t1s[1], t2s[1])
            yield z
