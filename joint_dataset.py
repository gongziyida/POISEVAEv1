import torch
import scipy.io as sio
import random
import numpy as np

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