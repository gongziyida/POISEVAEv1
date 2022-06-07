import torch
import torch.nn as nn

class EncMNIST1(nn.Module):
    def __init__(self, latent_dim_mnist1, latent_dim_mnist2):
        super(EncMNIST1, self).__init__()
        self.latent_dim_mnist1 = latent_dim_mnist1
        self.latent_dim_mnist2 = latent_dim_mnist2
        self.dim_MNIST = 28 * 28

        self.enc = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(512, 128),
                                 nn.ReLU(inplace=True))
        self.enc_mu_mnist1 = nn.Linear(128, latent_dim_mnist1)
        self.enc_var_mnist1 = nn.Linear(128, latent_dim_mnist1)
        self.enc_mu_mnist2 = nn.Linear(128, latent_dim_mnist2)
        self.enc_var_mnist2 = nn.Linear(128, latent_dim_mnist2)

    def forward(self, x):
        x = self.enc(x)
        mu_mnist1 = self.enc_mu_mnist1(x)
        log_var_mnist1 = self.enc_var_mnist1(x)
        mu_mnist2 = self.enc_mu_mnist2(x)
        log_var_mnist2 = self.enc_var_mnist2(x)
        return mu_mnist1, log_var_mnist1

class DecMNIST1(nn.Module):
    def __init__(self, latent_dim):
        super(DecMNIST1, self).__init__()  
        self.latent_dim = latent_dim
        self.dim_MNIST   = 28 * 28
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, 128), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 512), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512, self.dim_MNIST), 
                                 nn.Sigmoid())
        
    def forward(self, z):
        return self.dec(z).reshape(-1, 1, 28, 28), torch.tensor(0.75).to(z.device)

    
class EncMNIST2(nn.Module):
    def __init__(self, latent_dim_mnist1, latent_dim_mnist2):
        super(EncMNIST2, self).__init__()
        self.latent_dim_mnist1 = latent_dim_mnist1
        self.latent_dim_mnist2 = latent_dim_mnist2
        self.dim_MNIST = 28 * 28

        self.enc = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(512, 128),
                                 nn.ReLU(inplace=True))
        self.enc_mu_mnist1 = nn.Linear(128, latent_dim_mnist1)
        self.enc_var_mnist1 = nn.Linear(128, latent_dim_mnist1)
        self.enc_mu_mnist2 = nn.Linear(128, latent_dim_mnist2)
        self.enc_var_mnist2 = nn.Linear(128, latent_dim_mnist2)

    def forward(self, x):
        x = self.enc(x)
        mu_mnist1      = self.enc_mu_mnist1(x)
        log_var_mnist1 = self.enc_var_mnist1(x)
        mu_mnist2      = self.enc_mu_mnist2(x)
        log_var_mnist2 = self.enc_var_mnist2(x)
        return mu_mnist2, log_var_mnist2

class DecMNIST2(nn.Module):
    def __init__(self, latent_dim):
        super(DecMNIST2, self).__init__()  
        self.latent_dim = latent_dim
        self.dim_MNIST   = 28 * 28
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, 128), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 512), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512, self.dim_MNIST), 
                                 nn.Sigmoid())
        
    def forward(self, z):
        return self.dec(z).reshape(-1, 1, 28, 28), torch.tensor(0.75).to(z.device)
    