import torch
import scipy.io as sio
import random
import numpy as np
from torch.distributions import multivariate_normal as mv
from itertools import product

from CUBDatasets import CUBSentences, CUBImageFt

class MNIST_SVHN(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, svhn_mat_path):

        self.mnist_pt_path = mnist_pt_path
        self.svhn_mat_path = svhn_mat_path
            
        # Load the pt for MNIST and mat for SVHN 
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        
        # Reading the SVHN data
        svhn_mat_info = sio.loadmat(self.svhn_mat_path)

        self.svhn_data = svhn_mat_info['X']
        
        self.svhn_targets = svhn_mat_info['y'].astype(np.int64).squeeze()

        
        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.svhn_targets, self.svhn_targets == 10, 0)
        self.svhn_data = np.transpose(self.svhn_data, (3, 2, 0, 1))
        
        # Now we have the svhn data and the SVHN Labels, for each index get the classes
        self.svhn_target_idx_mapping = self.process_svhn_labels()
        
    def process_svhn_labels(self):
        numbers_dict = {0: [], 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[], 9:[]}
        for i in range(len(self.svhn_targets)):
            svhn_target = self.svhn_targets[i]
            numbers_dict[svhn_target].append(i)
        return numbers_dict
        
    def __len__(self):
        return len(self.mnist_data)
        
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        """
        mnist_img, mnist_target = self.mnist_data[index], int(self.mnist_targets[index])
        indices_list = self.svhn_target_idx_mapping[mnist_target]
        
        # Randomly pick an index from the indices list
        idx = random.choice(indices_list)
        svhn_img = self.svhn_data[idx]
        
        svhn_target = self.svhn_targets[idx]
        # What are the indices in SVHN that 
        return mnist_img.view(-1)/255, svhn_img/255, mnist_target, svhn_target

    
class MNIST_GM(torch.utils.data.Dataset):
    def __init__(self, mnist_pt_path, sample_size=800):

        self.mnist_pt_path = mnist_pt_path
            
        # Load the pt for MNIST
        self.mnist_data, self.mnist_targets = torch.load(self.mnist_pt_path)
        self.mnist_data = self.mnist_data[(self.mnist_targets!=0)&(self.mnist_targets!=9),:,:]
        self.mnist_targets = self.mnist_targets[(self.mnist_targets!=0)&(self.mnist_targets!=9)]
        
        # Generate Gaussian mixtures
        self.gm_var = 1 * torch.eye(2,dtype=torch.float32)
        radius = 20
        angles = np.pi * np.arange(8) / 4
        self.gm_locs = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])
        self.gms = [mv.MultivariateNormal(torch.tensor(mu, dtype=torch.float32), self.gm_var) for mu in self.gm_locs]
        self.gm_data = torch.cat([d.sample((sample_size, )) for d in self.gms], dim=0)
        self.gm_data = (self.gm_data - self.gm_data.mean(dim=0)) / self.gm_data.std(dim=0)
        self.gm_targets = np.repeat(np.arange(1, 9, dtype=np.int32), sample_size)
        
        self.gauss_target_idx_mapping = self.process_gauss_labels()
        
    def process_gauss_labels(self):
        numbers_dict = { 1: [], 2: [], 3:[], 4:[], 5:[], 6:[], 7: [], 8:[]}
        for i in range(len(self.gm_targets)):
            gauss_target = self.gm_targets[i]
            numbers_dict[gauss_target].append(i)
        return numbers_dict
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
            Modality 1: 1-8 MNIST
            Modality 2: 8 Gaussian distributions
        """
        mnist_img, mnist_target = self.mnist_data[index], int(self.mnist_targets[index])
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
        
