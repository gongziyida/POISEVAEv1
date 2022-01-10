import torch
from torchviz import make_dot

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KLD():
    __version__ = 3.0
    
    def __init__(self, latent_dims, device=_device):
        self.latent_dims = latent_dims
        self.device = device
        
    def calc(self, G, z_priors, z, lambda1s, lambda2s):
        ## Creating Sufficient statistics
        T_priors, T_posts, lambdas, ET_posts = [], [], [], []
        for i in range(len(z)):
            T_priors.append(torch.cat((z_priors[i], torch.square(z_priors[i])), -1))
            T_posts.append(torch.cat((z[i], torch.square(z[i])), -1))
            lambdas.append(torch.cat((lambda1s[i], lambda2s[i]), -1))
            ET_posts.append(T_posts[-1].mean(dim=1))
            
        # TODO: make it generic for > 2 latent spaces
        batch_size = z[0].shape[0]
        if z[1].shape[0] != batch_size:
            raise ValueError('Batch size must match.')
        
        T1_prior_unsq = T_priors[0].unsqueeze(-1)
        T2_prior_unsq = T_priors[1].unsqueeze(-2)
        T1_post_unsq  = T_posts[0].unsqueeze(-1) 
        T2_post_unsq  = T_posts[1].unsqueeze(-2)
        
        T_prior_kron = T1_prior_unsq * T2_prior_unsq
        T_post_kron = T1_post_unsq * T2_post_unsq
        
        part0 = (lambdas[0] * ET_posts[0]).sum(dim=-1) + \
                (lambdas[1] * ET_posts[1]).sum(dim=-1)
        part0 = part0.sum(dim=0)
        
        part1 = -(lambdas[0] * ET_posts[0].detach()).sum(dim=-1) - \
                (lambdas[1] * ET_posts[1].detach()).sum(dim=-1)
        part1 = part1.sum(dim=0)
        
        part2 = (T_prior_kron.mean(dim=1).detach() * G).sum(dim=(-1, -2)) - \
                (T_post_kron.mean(dim=1).detach() * G).sum(dim=(-1, -2))
        part2 = part2.sum(dim=0)
        
        return part0, part1, part2
    