import os
from itertools import product
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Laplace
from tqdm import tqdm
from datetime import datetime
from tensorboardX import SummaryWriter

import poisevae
from poisevae.datasets import MNIST_SVHN
from poisevae.networks.MNISTSVHNNetworks import EncMNIST, DecMNIST, EncSVHN, DecSVHN

HOME_PATH = os.path.expanduser('~')
MNIST_PATH = os.path.join(HOME_PATH, 'Datasets/MNIST/%s.pt')
SVHN_PATH = os.path.join(HOME_PATH, 'Datasets/SVHN/%s_32x32.mat')

def run(lat1, lat2, kl_weight, n_gibbs_iter, missing_prob):
    if (lat1 == 10 and lat2 == 10 and kl_weight == 3 and n_gibbs_iter == 15 and missing_prob == 0.1) or \
       (lat1 == 20 and lat2 == 20 and kl_weight == 9 and n_gibbs_iter == 30 and missing_prob == 0.2): 
           return 
    enc_mnist = EncMNIST(lat1).to(device)
    dec_mnist = DecMNIST(lat1).to(device)
    enc_svhn = EncSVHN(lat2).to(device)
    dec_svhn = DecSVHN(lat2).to(device)

    vae = poisevae.POISEVAE_Gibbs('autograd', 
                                  [enc_mnist, enc_svhn], [dec_mnist, dec_svhn], likelihoods=[Laplace, Laplace],
                                  latent_dims=[lat1, (lat2, 1, 1)], enc_config='nu', KL_calc='derivative', 
                                  batch_size=batch_size
                                 ).to(device)
    
    optimizer = optim.Adam(vae.parameters(), lr=1e-3, amsgrad=True)
    
    def mask_missing(data):
        a = random.random()
        if a < missing_prob:
            return [None, data[1]]
        if a < 2 * missing_prob:
            return [data[0], None]
        else:
            return data
    
    s = '%d_%d_%.1f_%d_%.2f' % (lat1, lat2, kl_weight, n_gibbs_iter, missing_prob)
    foldername = datetime.now().strftime('%y%m%d%H%M__' + s)
    PATH = os.path.join('runs/MNIST_SVHN/', foldername)
    print(PATH)
    if os.path.exists(PATH):
        raise ValueError
        
    writer = SummaryWriter(PATH)
    
    epochs = 100
    for epoch in tqdm(range(0, epochs)):
        poisevae.utils.train(vae, train_loader, optimizer, epoch, kl_weight, n_gibbs_iter, writer, 
                             mask_missing)
        poisevae.utils.test(vae, test_loader, epoch, kl_weight, n_gibbs_iter, writer)
        if (epoch+1) in (50, 75, 100):
            name = os.path.join(PATH, 'training_%d.pt' % (epoch+1))
            poisevae.utils.save_checkpoint(vae, optimizer, name, epoch+1)
            
            
    writer.flush()
    writer.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    joint_dataset_train = MNIST_SVHN(mnist_pt_path=MNIST_PATH % 'train', 
                                     svhn_mat_path=SVHN_PATH % 'train')
    joint_dataset_test = MNIST_SVHN(mnist_pt_path=MNIST_PATH % 'test', 
                                    svhn_mat_path=SVHN_PATH % 'test')
    
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(joint_dataset_train, batch_size=batch_size, 
                                               shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(joint_dataset_test, batch_size=batch_size, 
                                              shuffle=True, drop_last=True)
    
    
    lats = (10, 20, 30)
    kl_weights = (3, 6, 9)
    n_gibbs_iters = (15, 30)
    missing_probs = (0.1, 0.2)
        
    for l, kl_w, n_g, p in product(lats, kl_weights, n_gibbs_iters, missing_probs):
        run(lat1=l, lat2=l, kl_weight=kl_w, n_gibbs_iter=n_g, missing_prob=p)
