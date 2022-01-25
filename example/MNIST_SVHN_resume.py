import glob 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Laplace
from tqdm import tqdm
from tensorboardX import SummaryWriter

import poisevae
from poisevae.datasets import MNIST_SVHN
from poisevae.networks.MNISTSVHNNetworks import EncMNIST, DecMNIST, EncSVHN, DecSVHN

HOME_PATH = os.path.expanduser('~')
MNIST_PATH = os.path.join(HOME_PATH, 'Datasets/MNIST/%s.pt')
SVHN_PATH = os.path.join(HOME_PATH, 'Datasets/SVHN/%s_32x32.mat')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    joint_dataset_train = MNIST_SVHN(mnist_pt_path=MNIST_PATH % 'train', svhn_mat_path=SVHN_PATH % 'train')
    joint_dataset_test = MNIST_SVHN(mnist_pt_path=MNIST_PATH % 'test', svhn_mat_path=SVHN_PATH % 'test')
    
    batch_size = 256
    train_loader = torch.utils.data.DataLoader(joint_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(joint_dataset_test, batch_size=batch_size, shuffle=True, drop_last=True)
    
    lat1, lat2 = 16, 16
    enc_mnist = EncMNIST(lat1).to(device)
    dec_mnist = DecMNIST(lat1).to(device)
    enc_svhn = EncSVHN(lat2).to(device)
    dec_svhn = DecSVHN(lat2).to(device)

    vae = poisevae.POISEVAE([enc_mnist, enc_svhn], [dec_mnist, dec_svhn], likelihoods=[Laplace, Laplace],
                            latent_dims=[lat1, (lat2, 1, 1)], batch_size=batch_size, fix_t=True).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, amsgrad=True)
    
    paths = glob.glob('runs/MNIST_SVHN/fix_t/2201251135')
    for path in paths:
        vae, optimizer, epoch = poisevae.utils.load_checkpoint(vae, optimizer, os.path.join(path, 'training_200.pt'))

        writer = SummaryWriter(path)

        epochs = 100 + epoch
        for epoch in tqdm(range(epoch, epochs)):
            poisevae.utils.train(vae, train_loader, optimizer, epoch, writer)
            poisevae.utils.test(vae, test_loader, epoch, writer)
            if (epoch+1) % 20 == 0 and epoch > 0:
                poisevae.utils.save_checkpoint(vae, optimizer, os.path.join(path, 'training_%d.pt' % (epoch+1)), epoch+1)

        writer.flush()
        writer.close()
