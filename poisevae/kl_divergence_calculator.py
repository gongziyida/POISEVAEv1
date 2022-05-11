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
            return nu1, nu2

    elif enc_config == 'nu':
        if nu1 is not None and nu2 is not None:
            return nu1, nu2
        
    elif enc_config == 'mu/nu2':
        if mu is not None and nu2 is not None:
            nu1 = [None, None]
            
            for i in (0, 1):
                if mu[i] is not None and nu2[i] is not None:
                    nu1[i] = -2 * mu[i] * nu2[i]
            return nu1, nu2
    # Prior
    return [None, None], [None, None]

def new_nu(nu, nu_tilde):
    nu_ = [None, None]
    for i in (0, 1):
        if (nu[i] is None) and (nu_tilde[1-i] is not None):
            nu_[i] = nu_tilde[1-i]
        elif (nu[i] is not None) and (nu_tilde[1-i] is None):
            nu_[i] = nu[i]
        elif (nu[i] is not None) and (nu_tilde[1-i] is not None):
            nu_[i] = nu[i] + nu_tilde[1-i]
    return nu_

class KLDN01:
    __version__ = 3.1 # lambda -> nu
    
    def __init__(self, latent_dims, enc_config, reduction, device=_device):
        self.latent_dims = latent_dims
        self.enc_config = enc_config
        self.reduction = reduction
        self.device = device
        
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
        return torch.square(mean) + var - torch.log(var) - 1
        
    def calc(self, G, z, z_priors, nu1=None, nu2=None, mu=None, var=None, 
             nu1_=None, nu2_=None, mu_=None, var_=None):
        nu1, nu2 = init_posterior(nu1, nu2, mu, var, self.enc_config)
        nu1_, nu2_ = init_posterior(nu1_, nu2_, mu_, var_, self.enc_config)
        nu1, nu2 = new_nu(nu1, nu1_), new_nu(nu2, nu2_)
        part0 = self.value_calc(z[1], G.t(), nu1[0], nu2[0], 0, -1).sum(dim=1)
        part1 = self.value_calc(z[0], G, nu1[1], nu2[1], 0, -1).sum(dim=1)
        res = (part0 + part1) * 0.5
        if self.reduction == 'sum':
            return res.sum()
        elif self.reduction == 'mean':
            return res.mean()
        else:
            return res
    
    
class KLDDerivative:
    __version__ = 2.3 # lambda -> nu
    
    def __init__(self, latent_dims, enc_config, reduction, device=_device):
        self.latent_dims = latent_dims
        self.enc_config = enc_config
        self.reduction = reduction
        self.device = device
        
    def calc(self, G, z, z_priors, nu1=None, nu2=None, mu=None, var=None,
             nu1_=None, nu2_=None, mu_=None, var_=None):
        nu1, nu2 = init_posterior(nu1, nu2, mu, var, self.enc_config)
        nu1_, nu2_ = init_posterior(nu1_, nu2_, mu_, var_, self.enc_config)
        nu1, nu2 = new_nu(nu1, nu1_), new_nu(nu2, nu2_)
        ## Creating Sufficient statistics
        T_priors, T_posts, nus = [], [], []
        for i in range(len(z)):
            T_priors.append(torch.cat((z_priors[i], torch.square(z_priors[i])), -1))
            T_posts.append(torch.cat((z[i], torch.square(z[i])), -1))
            if nu1[i] is None: 
                if nu2[i] is not None:
                    raise RuntimeError('Unmatched nus')
                
                nus.append(torch.zeros(T_posts[-1].shape[0], T_posts[-1].shape[2]).to(self.device))
            else:
                nus.append(torch.cat((nu1[i], nu2[i]), -1))
        # print('zpost max:', torch.abs(z[0]).max().item(), 'zpost mean:', torch.abs(z[0]).mean().item())
        # print('zpostp max:', torch.abs(z[1]).max().item(), 'zpostp mean:', torch.abs(z[1]).mean().item())
        # print('zprior max:', torch.abs(z_priors[0]).max().item(), 'zprior mean:', torch.abs(z_priors[0]).mean().item())
        # print('zpriorp max:', torch.abs(z_priors[1]).max().item(), 'zpriorp mean:', torch.abs(z_priors[1]).mean().item())
        # TODO: make it generic for > 2 latent spaces
        batch_size = z[0].shape[0]
        if z[1].shape[0] != batch_size:
            raise ValueError('Batch size must match.')
        
        T1_prior_unsq = T_priors[0].unsqueeze(-1)
        T2_prior_unsq = T_priors[1].unsqueeze(-2)
        T1_post_unsq  = T_posts[0].unsqueeze(-1) 
        T2_post_unsq  = T_posts[1].unsqueeze(-2)
        
        T_prior_outer = (T1_prior_unsq * T2_prior_unsq).mean(1)
        T_post_outer = (T1_post_unsq * T2_post_unsq).mean(1)
        
        T_post_mean = [T_posts[0].mean(1), T_posts[1].mean(1)]
        
        part0 = (nus[0] * T_post_mean[0]).sum(dim=-1) + \
                (nus[1] * T_post_mean[1]).sum(dim=-1)
        
        part1 = (nus[0] * T_post_mean[0].detach()).sum(dim=-1) + \
                (nus[1] * T_post_mean[1].detach()).sum(dim=-1)
        
        part2 = (T_prior_outer.detach() * G).sum(dim=(-1, -2)) - \
                (T_post_outer.detach() * G).sum(dim=(-1, -2))
        
        assert len(part0.shape) == 1 and len(part1.shape) == 1 and len(part2.shape) == 1
        if self.reduction == 'mean':
            return part0.mean() - part1.mean() + part2.mean()
        elif self.reduction == 'sum':
            return part0.sum() - part1.sum() + part2.sum()
        else:
            raise NotImplementedError
    
    def dot_product(self, tensor_1, tensor_2):
        out = torch.sum(torch.mul(tensor_1, tensor_2))
        return out
    
