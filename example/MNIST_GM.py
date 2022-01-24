import os
import torch
import torch.nn as nn
from torch.distributions import Laplace, Normal
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import argparse

import poisevae
from poisevae.datasets import MNIST_GM

HOME_PATH = os.path.expanduser('~')
MNIST_PATH = os.path.join(HOME_PATH, 'Datasets/MNIST/%s.pt')

parser = argparse.ArgumentParser()
parser.add_argument('--foldername', type=str, help='Folder name under runs/MNIST_GM/')
parser.add_argument('--radius', type=float, help='radius')

class EncMNIST(nn.Module):
    def __init__(self, latent_dim):
        super(EncMNIST, self).__init__()
        self.latent_dim = latent_dim
        self.dim_MNIST = 28 * 28

        self.enc1 = nn.Linear(self.dim_MNIST, 400)
        self.enc_mu = nn.Linear(400, latent_dim)
        self.enc_var = nn.Linear(400, latent_dim)

    def forward(self, x):
        x = nn.functional.relu(self.enc1(x))
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
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()
    print(args)
    
    joint_dataset_train = MNIST_GM(mnist_pt_path=MNIST_PATH % 'train', data_augment=1, radius=args.radius)
    joint_dataset_test = MNIST_GM(mnist_pt_path=MNIST_PATH % 'test', data_augment=1, radius=args.radius)

    batch_size = 256
    train_loader = torch.utils.data.DataLoader(joint_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(joint_dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)

    lat1, lat2 = 2, 2
    emb_dim = 8
    enc_mnist = EncMNIST(lat1).to(device)
    dec_mnist = DecMNIST(lat1).to(device)
    enc_gm = EncGM(2, emb_dim, lat2).to(device)
    dec_gm = DecGM(2, emb_dim, lat2).to(device)

    vae = poisevae.POISEVAE([enc_mnist, enc_gm], [dec_mnist, dec_gm], likelihoods=[Laplace, Normal], 
                            latent_dims=[lat1, lat2], reduction='mean', batch_size=batch_size).to(device)

    optimizer = optim.Adam(vae.parameters(), lr=8e-4, amsgrad=True)

    PATH = os.path.join('runs/MNIST_GM', args.foldername)
    print(PATH)
    if os.path.exists(PATH):
        raise ValueError

    writer = SummaryWriter(PATH)

    epochs = 40 
    for epoch in tqdm(range(0, epochs)):
        poisevae.utils.train(vae, train_loader, optimizer, epoch, writer)
        poisevae.utils.test(vae, test_loader, epoch, writer)
        if (epoch+1) % 10 == 0 and epoch > 0:
            poisevae.utils.save_checkpoint(vae, optimizer, os.path.join(PATH, 'training_%d.pt' % (epoch+1)), epoch+1) 

    writer.flush()
    writer.close()

