import torch
from torchviz import make_dot

# TODO: make it generic

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class kl_divergence():
    def __init__(self, latent_dim1, latent_dim2, batch_size):
        self.latent_dim1 = latent_dim1
        self.latent_dim2 = latent_dim2
        self.batch_size = batch_size

    def calc(self,G,z1,z2,z1_prior,z2_prior,mu1,log_var1,mu2,log_var2):
        ## Creating Sufficient statistics
        T1_prior = torch.cat((z1_prior,torch.square(z1_prior)),1)     # sufficient statistics for prior of set1
        T2_prior = torch.cat((z2_prior,torch.square(z2_prior)),1)     # sufficient statistics for prior of set2
        T1_post = torch.cat((z1,torch.square(z1)),1)                  # sufficient statistics for posterior of set1
        T2_post = torch.cat((z2,torch.square(z2)),1)                  # sufficient statistics for posterior of set2
        lambda1 = torch.cat((mu1,log_var1),1)                         # Output of encoder for set1
        lambda2 = torch.cat((mu2,log_var2),1)                         # Output of encoder for set2 
        T_prior_sqrd = torch.sum(torch.square(z1_prior),1) +torch.sum(torch.square(z2_prior),1) #stores z^2+z'^2
        T_post_sqrd  = torch.sum(torch.square(z1),1) +torch.sum(torch.square(z2),1)
        T1_prior_unsq = T1_prior.unsqueeze(2)       #[128, 2]->[128, 64,1]
        T2_prior_unsq = T2_prior.unsqueeze(1)       #[128, 64]->[128, 1,32]
        T1_post_unsq  = T1_post.unsqueeze(2)        #[128, 2]->[128, 64,1]
        T2_post_unsq  = T2_post.unsqueeze(1)        #[128, 64]->[128, 1,32]
        Tprior_kron=torch.zeros(self.batch_size,2*self.latent_dim1,2*self.latent_dim2).to(device)   #[128, 64,32]
        Tpost_kron=torch.zeros(self.batch_size,2*self.latent_dim1,2*self.latent_dim2).to(device)    #[128, 64,32]  
       
        for i in range(self.batch_size):
            Tprior_kron[i,:]=torch.kron(T1_prior_unsq[i,:], T2_prior_unsq[i,:])
            Tpost_kron[i,:]=torch.kron(T1_post_unsq[i,:], T2_post_unsq[i,:])    
            
        part_fun0 = self.dot_product(lambda1,T1_post)+self.dot_product(lambda2,T2_post)
        part_fun1 = -self.dot_product(lambda1,T1_post.detach())-self.dot_product(lambda2,T2_post.detach()) #-lambda*Tq-lambda'Tq'    
        part_fun2 = self.dot_product(Tprior_kron.detach(),G)-self.dot_product(Tpost_kron.detach(),G)

        
        return part_fun0,part_fun1,part_fun2
    def dot_product(self,tensor_1,tensor_2):
        out = torch.sum(torch.mul(tensor_1,tensor_2))

        return out
