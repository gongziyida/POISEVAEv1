import torch
import torch.nn as nn

class EncImg(nn.Module):
    def __init__(self, latent_dim, input_dim=2048):
        super(EncImg, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        n = (input_dim, 1024, 512, 256)
        li = []
        for i in range(len(n)-1):
            li += [nn.Linear(n[i], n[i+1]), nn.ELU(inplace=True)]
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Linear(256, latent_dim)
        self.enc_var = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        x = self.enc(x)
        mu = self.enc_mu(x)
        log_var = self.enc_var(x)
        return mu, log_var

class DecImg(nn.Module):
    def __init__(self, latent_dim, output_dim=2048):
        super(DecImg, self).__init__()  
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        n = (output_dim, 1024, 512, 256)
        li = [nn.Linear(latent_dim, 256)]
        for i in range(len(n)-1, 0, -1):
            li += [nn.ELU(inplace=True), nn.Linear(n[i], n[i-1])]
        self.dec = nn.Sequential(*li)
        
    def forward(self, z):
        return self.dec(z), torch.tensor(0.75).to(z.device)


class EncTxt(nn.Module):
    def __init__(self, vocab_size, latent_dim, emb_dim=128):
        super(EncTxt, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        
        # 0 is for the excluded words and does not contribute to gradient
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        n_channels = (1, 32, 64, 128, 256, 512)
        kernels = (4, 4, 4, (1, 4), (1, 4))
        strides = (2, 2, 2, (1, 2), (1, 2))
        paddings = (1, 1, 1, (0, 1), (0, 1))
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li += [nn.Conv2d(n_channels[i-1], n, kernel_size=k, stride=s, padding=p), 
                   nn.BatchNorm2d(n), nn.ReLU(inplace=True)]
            
        self.enc = nn.Sequential(*li)
        self.enc_mu = nn.Conv2d(512, latent_dim, kernel_size=4, stride=1, padding=0)
        self.enc_var = nn.Conv2d(512, latent_dim, kernel_size=4, stride=1, padding=0)
        
    def forward(self, x):
        x = self.emb(x.long()).unsqueeze(1) # add channel dim
        x = self.enc(x)
        mu = self.enc_mu(x).squeeze()
        log_var = self.enc_var(x).squeeze()
        return mu, log_var

class DecTxt(nn.Module):
    def __init__(self, vocab_size, latent_dim, emb_dim=128, txt_len=32):
        super(DecTxt, self).__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.emb_dim = emb_dim
        self.txt_len = txt_len
        
        n_channels = (1, 32, 64, 128, 256, 512, latent_dim)
        kernels = (4, 4, 4, (1, 4), (1, 4), 4)
        strides = (2, 2, 2, (1, 2), (1, 2), 1)
        paddings = (1, 1, 1, (0, 1), (0, 1), 0)
        li = []
        for i, (n, k, s, p) in enumerate(zip(n_channels[1:], kernels, strides, paddings), 1):
            li = [nn.ConvTranspose2d(n, n_channels[i-1], kernel_size=k, stride=s, padding=p), 
                  nn.BatchNorm2d(n_channels[i-1]), nn.ReLU(inplace=True)] + li
            
        # No batchnorm at the first and last block
        del li[-2]
        del li[1]
        self.dec = nn.Sequential(*li)
        # self.anti_emb = nn.Sequential(nn.Linear(self.emb_dim, self.vocab_size),
        #                               nn.Sigmoid())
        self.anti_emb = nn.Linear(self.emb_dim, self.vocab_size)
        
    def forward(self, z, argmax=False):
        z = self.dec(z)
        z = self.anti_emb(z.view(-1, self.emb_dim))
        # z = z.view(-1, self.txt_len, self.vocab_size) # batch x txt len x vocab size
        z = z.view(-1, self.vocab_size)
        
        if argmax:
            z = z.argmax(-1).float()
        return (z, )