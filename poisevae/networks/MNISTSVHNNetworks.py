import torch
import torch.nn as nn

class EncMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(EncMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.dim_MNIST = 28 * 28

        self.enc = nn.Sequential(nn.Linear(self.dim_MNIST, 512),
                                 nn.ReLU(inplace=True), 
                                 nn.Linear(512, 128),
                                 nn.ReLU(inplace=True))
        self.enc_mu = nn.Linear(128, latent_dim)
        self.enc_var = nn.Linear(128, latent_dim)

    def forward(self, x):
        x = self.enc(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

class DecMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(DecMNIST, self).__init__()  
        self.latent_dim = latent_dim
        self.dim_MNIST   = 28 * 28
        
        self.dec = nn.Sequential(nn.Linear(self.latent_dim, 128), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(128, 512), 
                                 nn.ReLU(inplace=True),
                                 nn.Linear(512, self.dim_MNIST), 
                                 nn.Sigmoid())
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)
    
class EncSVHN(nn.Module):
    def __init__(self, latent_dim):
        super(EncSVHN, self).__init__()
        self.latent_dim = latent_dim
        
        n_channels = (3, 32, 64, 128)
        kernels = (4, 4, 4)
        strides = (2, 2, 2)
        paddings = (1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.Conv2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
            
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Conv2d(in_channels=128, out_channels=latent_dim, 
                                kernel_size=4, stride=1, padding=0)
        self.enc_var = nn.Conv2d(in_channels=128, out_channels=latent_dim, 
                                 kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.enc(x)
        # Be careful not to squeeze the batch dimension if batch size = 1
        mu = self.enc_mu(x).squeeze(3).squeeze(2)
        log_var = self.enc_var(x).squeeze(3).squeeze(2)
        return mu, log_var
    
class DecSVHN(nn.Module):
    def __init__(self, latent_dim):
        super(DecSVHN, self).__init__()  
        self.latent_dim = latent_dim
        
        n_channels = (latent_dim, 128, 64, 32, 3)
        kernels = (4, 4, 4, 4)
        strides = (1, 2, 2, 2)
        paddings = (0, 1, 1, 1)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.ConvTranspose2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.ReLU(inplace=True)]
        li[-1] = nn.Sigmoid()
        
        self.dec = nn.Sequential(*li)
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)