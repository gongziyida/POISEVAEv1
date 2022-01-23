import torch
import torch.nn as nn
from torch.nn import functional as F

class EncMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(EncMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.dim_MNIST = 28 * 28

        self.enc1 = nn.Linear(self.dim_MNIST, 400)
        self.enc_mu = nn.Linear(400, latent_dim)
        self.enc_var = nn.Linear(400, latent_dim)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

class DecMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(DecMNIST, self).__init__()  
        self.latent_dim = latent_dim
        self.dim_MNIST   = 28 * 28
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, 400), 
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(400, self.dim_MNIST), 
                                 nn.Sigmoid())
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)
    
    
class EncGM(nn.Module):
    def __init__(self, data_dim, emb_dim, latent_dim):
        super(EncGM, self).__init__()
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.data_dim = data_dim
        if latent_dim > data_dim:
            raise ValueError('latent_dim > data_dim')

        self.enc = nn.Sequential(nn.Linear(data_dim, emb_dim),
                                 nn.LeakyReLU(inplace=True))
        self.enc_mu = nn.Linear(emb_dim, latent_dim)
        self.enc_var = nn.Linear(emb_dim, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

class DecGM(nn.Module):
    def __init__(self, data_dim, emb_dim, latent_dim):
        super(DecGM, self).__init__()  
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.data_dim = data_dim
        if latent_dim > data_dim:
            raise ValueError('latent_dim > data_dim')
            
        self.dec = nn.Sequential(nn.Linear(latent_dim, emb_dim), 
                                 nn.LeakyReLU(inplace=True),
                                 nn.Linear(emb_dim, data_dim))
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)