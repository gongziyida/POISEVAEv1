import torch
from torchviz import make_dot

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class kl_divergence():
    __version__ = 1.0
    
    def __init__(self, latent_dims, batch_size, device=_device):
        self.latent_dims = latent_dims
        self.batch_size = batch_size
        self.device = device

    def calc(self, G, z, z_priors, mu, var):
        ## Creating Sufficient statistics
        T_priors, T_posts, lambdas = [], [], []
        for z_i, z_prior_i, mu_i, var_i in zip(z, z_priors, mu, var):
            T_priors.append(torch.cat((z_prior_i, torch.square(z_prior_i)), 1))
            T_posts.append(torch.cat((z_i, torch.square(z_i)), 1))
            lambdas.append(torch.cat((mu_i,var_i),1))
            
        # TODO: make it generic for > 2 latent spaces
        T_prior_sqrd = torch.sum(torch.square(z_priors[0]), 1) + \
                       torch.sum(torch.square(z_priors[1]), 1) #stores z^2+z'^2
        T_post_sqrd  = torch.sum(torch.square(z[0]), 1) + \
                       torch.sum(torch.square(z[1]), 1)
        T1_prior_unsq = T_priors[0].unsqueeze(2)       
        T2_prior_unsq = T_priors[1].unsqueeze(1)       
        T1_post_unsq  = T_posts[0].unsqueeze(2)        
        T2_post_unsq  = T_posts[1].unsqueeze(1)        
        T_prior_kron = torch.zeros(self.batch_size, 2 * self.latent_dims[0], 
                                   2 * self.latent_dims[1]).to(self.device)
        T_post_kron = torch.zeros(T_prior_kron.shape).to(self.device)
       
        for i in range(self.batch_size):
            T_prior_kron[i,:] = torch.kron(T1_prior_unsq[i,:], T2_prior_unsq[i,:])
            T_post_kron[i,:] = torch.kron(T1_post_unsq[i,:], T2_post_unsq[i,:])    
            
        part_fun0 = self.dot_product(lambdas[0], T_posts[0]) + \
                    self.dot_product(lambdas[1], T_posts[1])
        part_fun1 = -self.dot_product(lambdas[0], T_posts[0].detach()) - \
                     self.dot_product(lambdas[1], T_posts[1].detach()) #-lambda*Tq-lambda'Tq'    
        part_fun2 = self.dot_product(T_prior_kron.detach(), G) - \
                    self.dot_product(T_post_kron.detach(), G)

        return part_fun0, part_fun1, part_fun2
    
    def dot_product(self, tensor_1, tensor_2):
        out = torch.sum(torch.mul(tensor_1, tensor_2))
        return out