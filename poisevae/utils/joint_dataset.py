import os
import torch
import scipy.io as sio
import random
import numpy as np
from torch.distributions import multivariate_normal as mv
from itertools import product

from .CUBDatasets import CUBSentences, CUBImageFt

def _rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
# The code is adapted from https://github.com/iffsid/mmvae, the repository for the work
# Y. Shi, N. Siddharth, B. Paige and PHS. Torr.
# Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models.
# In Proceedings of the 33rd International Conference on Neural Information Processing Systems,
# Page 15718–15729, 2019
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)

def augment_MNIST_SVHN(MNIST_PATH, SVHN_PATH, fname_MNIST_idx, fname_SVHN_idx, max_d=10000, dm=20):
    """
    max_d: int, default 10000
        Maximum number of datapoints per class
    dm: int, default 20
        Data multiplier: random permutations to match
    """
    mnist = torch.load(MNIST_PATH)
    svhn = sio.loadmat(SVHN_PATH)

    svhn['y'] = torch.LongTensor(svhn['y'].squeeze().astype(int)) % 10
    
    mnist_l, mnist_li = mnist[1].sort()
    svhn_l, svhn_li = svhn['y'].sort()
    idx_mnist, idx_svhn = _rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=max_d, dm=dm)
    torch.save(idx_mnist, os.path.join(os.path.dirname(MNIST_PATH), fname_MNIST_idx + '.pt'))
    torch.save(idx_svhn, os.path.join(os.path.dirname(SVHN_PATH), fname_SVHN_idx + '.pt'))
    
    return idx_mnist, idx_svhn


class MNIST_SVHN(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, svhn_mat_path, sampler_mnist=None, sampler_svhn=None):
        self.mnist_pt_path, self.svhn_mat_path = mnist_pt_path, svhn_mat_path
        self.sampler_mnist, self.sampler_svhn = sampler_mnist, sampler_svhn
            
        # Load the pt for MNIST and mat for SVHN 
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        
        # Reading the SVHN data
        svhn_mat_info = sio.loadmat(self.svhn_mat_path)
        self.svhn_data = svhn_mat_info['X']
        # the svhn dataset assigns the class label "10" to the digit 0
        self.svhn_targets = svhn_mat_info['y'].astype(np.int64).squeeze() % 10
        self.svhn_data = np.transpose(self.svhn_data, (3, 2, 0, 1))
        
        if sampler_mnist is None:
            # Now we have the svhn data and the SVHN Labels, for each index get the classes
            self.svhn_target_idx_mapping = self._process_svhn_labels()
            self.__len__ = lambda: len(self.mnist_data)
        else:
            self.__len__ = lambda: len(self.sampler_mnist)
        
    def _process_svhn_labels(self):
        numbers_dict = {0: [], 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[], 9:[]}
        for i in range(len(self.svhn_targets)):
            svhn_target = self.svhn_targets[i]
            numbers_dict[svhn_target].append(i)
        return numbers_dict
        
    def __len__(self):
        if self.sampler_mnist is None:
            return len(self.mnist_data)
        else:
            return len(self.sampler_mnist)
        
    def __getitem__(self, index):
        if self.sampler_mnist is None:
            mnist_img, mnist_target = self.mnist_data[index], int(self.mnist_targets[index])
            indices_list = self.svhn_target_idx_mapping[mnist_target]

            # Randomly pick an index from the indices list
            idx = random.choice(indices_list)
            svhn_img = self.svhn_data[idx]
            svhn_target = self.svhn_targets[idx]
        else:
            mnist_img = self.mnist_data[self.sampler_mnist[index]]
            mnist_target = int(self.mnist_targets[self.sampler_mnist[index]])

            svhn_img = self.svhn_data[self.sampler_svhn[index]]
            svhn_target = int(self.svhn_targets[self.sampler_svhn[index]])

        return mnist_img.view(-1)/255, svhn_img/255, mnist_target, svhn_target
    
    
class MNIST_GM(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, sample_size=800, var=1/400, radius=1, data_augment=10):

        self.mnist_pt_path = mnist_pt_path
            
        # Load the pt for MNIST
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        self.mnist_data = self.mnist_data[(self.mnist_targets!=0)&(self.mnist_targets!=9),:,:]
        self.mnist_targets = self.mnist_targets[(self.mnist_targets!=0)&(self.mnist_targets!=9)]
        
        # Generate Gaussian mixtures
        self.gm_var = var * torch.eye(2,dtype=torch.float32)
        angles = np.pi * np.arange(8) / 4
        self.gm_locs = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
        self.gms = [mv.MultivariateNormal(torch.tensor(mu, dtype=torch.float32), self.gm_var) for mu in self.gm_locs]
        self.gm_data = torch.cat([d.sample((sample_size, )) for d in self.gms], dim=0)
        self.gm_targets = np.repeat(np.arange(1, 9, dtype=np.int32), sample_size)
        
        self.gauss_target_idx_mapping = self.process_gauss_labels()
        
        self.data_augment = data_augment
        
    def process_gauss_labels(self):
        numbers_dict = { 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[]}
        for i in range(len(self.gm_targets)):
            gauss_target = self.gm_targets[i]
            numbers_dict[gauss_target].append(i)
        return numbers_dict
    
    def __len__(self):
        return len(self.mnist_data) * self.data_augment
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            Modality 1: 1-8 MNIST
            Modality 2: 8 Gaussian distributions
        """
        idx_mnist = int(np.floor(index / self.data_augment))
        mnist_img, mnist_target = self.mnist_data[idx_mnist], int(self.mnist_targets[idx_mnist])
        indices_list = self.gauss_target_idx_mapping[mnist_target]
        
        # Randomly pick an index from the indices list
        idx = random.choice(indices_list)
        gauss_data = self.gm_data[idx]
        
        gauss_target = self.gm_targets[idx]
        return mnist_img.view(-1)/255, gauss_data, mnist_target, gauss_target
    
    
class CUB(torch.utils.data.Dataset):
    def __init__(self, img_data_dir, txt_data_dir, split, device, transform=None, return_idx=False, **kwargs):
        """split: 'train' or 'test' """
        super().__init__()
        self.CUBtxt = CUBSentences(txt_data_dir, split=split, transform=transform, **kwargs)
        self.CUBimg = CUBImageFt(img_data_dir, split=split, device=device)
        self.return_idx = return_idx
        
    def __len__(self):
        return len(self.CUBtxt)
    
    def __getitem__(self, idx):
        txt = self.CUBtxt.__getitem__(idx)[0]
        img = self.CUBimg.__getitem__(idx // 10)
        if self.return_idx:
            return img, txt, idx
        else:
            return img, txt
        
