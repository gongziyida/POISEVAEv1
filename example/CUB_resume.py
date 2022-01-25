import glob
import os
import torch
import torch.nn as nn
from torch.distributions import Laplace, Normal
import torch.optim as optim
from tqdm import tqdm
from tensorboardX import SummaryWriter

import poisevae
from poisevae.datasets import CUB
from poisevae.utils import NN_lookup, Categorical
from poisevae.networks.CUBNetworks import EncImg, DecImg, EncTxt, DecTxt

HOME_PATH = os.path.expanduser('~')
DATA_PATH = os.path.join(HOME_PATH, 'Datasets/CUB/')

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tx = lambda data: torch.Tensor(data)
    CUB_train = CUB(DATA_PATH, DATA_PATH, 'train', device, tx, return_idx=False)
    CUB_test = CUB(DATA_PATH, DATA_PATH, 'test', device, tx, return_idx=True)
    vocab_size, txt_len = CUB_train.CUBtxt.vocab_size, CUB_train.CUBtxt.max_sequence_length
    
    batch_size = 128
    train_loader = torch.utils.data.DataLoader(CUB_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(CUB_test, batch_size=batch_size, shuffle=True)
    
    enc_img = EncImg(128).to(device)
    dec_img = DecImg(128).to(device)
    enc_txt = EncTxt(vocab_size, 128).to(device)
    dec_txt = DecTxt(vocab_size, 128).to(device)

    for rec_weighting in (False, True):
        paths = glob.glob('runs/CUB/%s/22*' % 'wrew' if rec_weighting else 'worew')
        rec_weights = [1, 2048/txt_len] if rec_weighting else None
        
        vae = poisevae.POISEVAE([enc_img, enc_txt], [dec_img, dec_txt], likelihoods=[Laplace, Categorical], 
                                latent_dims=[128, (128, 1, 1)], rec_weights=rec_weights).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=5e-4, amsgrad=True)

        for path in paths:
            vae, optimizer, epoch = poisevae.utils.load_checkpoint(vae, optimizer, os.path.join(path, 'training_50.pt'))

            writer = SummaryWriter(path)
            
            epochs = 50 + epoch
            for epoch in tqdm(range(epoch, epochs)):
                poisevae.utils.train(vae, train_loader, optimizer, epoch, writer)
                poisevae.utils.test(vae, test_loader, epoch, writer)
                if (epoch+1) % 10 == 0 and epoch > 0:
                    poisevae.utils.save_checkpoint(vae, optimizer, os.path.join(path, 'training_%d.pt' % (epoch+1)), epoch+1)

            writer.flush()
            writer.close()
