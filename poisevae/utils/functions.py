# Version: 1.0
import os
import numpy as np
import torch
from tqdm import tqdm

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def _log(results, mode, writer, epoch):
    for i, val in enumerate(results['rec_losses']):
        writer.add_scalars('%s/loss/rec%d' % (mode, i), {'rec': val.item()}, epoch)
    writer.add_scalars('%s/loss/kld' % mode, {'kld': results['KL_loss'].item()}, epoch)
    writer.add_scalars('%s/loss/total' % mode, {'total': results['total_loss'].item()}, epoch)


def train(model, joint_dataloader, optimizer, epoch, writer, record_idx=(), return_latents=False, progress_bar=False, 
          device=_device, dtype=torch.float32):
    '''
    Parameters
    ----------
    model: torch.nn.Module
        Must be callable with the following signature: `model([input1, input2, ...])`
    
    joint_dataloader: torch.utils.data.DataLoader
        The inputs to the model must be in order and the initial elements of the returned data, i.e.
        ```
        for data in joint_dataloader:
             model([data[0], data[1], ...])
        ```
    
    record_idx: list or tuple of integers, optional
        The indices of the elements in the data returned by `joint_dataloader` needed to be recorded
        This is helpful if `joint_dataloader` also returns useful non-input elements
    
    return_latents: bool, optional
        If true, return a dict containing the latent representation, mean, and variance.
    
    progress_bar: bool, optional
        If true, displace training progress bar
    
    Returns
    -------
    record: list, available if `record_idx` is not empty
        Data loaded, see above
    latent_info: dict, available if `return_latents` is True
        Contains `latent`, `mu`, and `var` returned by the model for each batch
    '''
    
    record = [[] for _ in range(len(record_idx))] # Empty if len(record_idx) == 0
    returns = {'latent': [[] for _ in range(model.M)], 
               'mu': [[] for _ in range(model.M)], 
               'var': [[] for _ in range(model.M)]}
    
    model.train()
    
    for k, data in tqdm(enumerate(joint_dataloader), disable=not progress_bar):
        optimizer.zero_grad()
        
        results = model([data[i].to(device=device, dtype=dtype) for i in range(model.M)])
        results['total_loss'].backward() 
        optimizer.step()
        _log(results, 'train', writer, epoch * len(joint_dataloader) + k)
        
        if return_latents:
            for i in range(model.M):
                returns['latent'][i].append(results['z'][i].cpu().numpy())
                returns['mu'][i].append(results['mu'][i].cpu().numpy())
                returns['var'][i].append(results['var'][i].cpu().numpy())
                    
        for i, j in enumerate(record_idx): # Does not iterate if empty
            record[i].append(data[j])
        
    
    ret_buffer = []
    if len(record_idx) > 0:
        ret_buffer.append(record)
    if return_latents:
        ret_buffer.append(returns)
    return tuple(ret_buffer)



def test(model, joint_dataloader, epoch, writer, record_idx=(), return_latents=False, progress_bar=False, 
         device=_device, dtype=torch.float32):
    '''
    Parameters
    ----------
    model: torch.nn.Module
        Must be callable with the following signature: `model([input1, input2, ...])`
    
    joint_dataloader: torch.utils.data.DataLoader
        The inputs to the model must be in order and the initial elements of the returned data, i.e.
        ```
        for data in joint_dataloader:
             model([data[0], data[1], ...])
        ```
    
    record_idx: list or tuple of integers, optional
        The indices of the elements in the data returned by `joint_dataloader` needed to be recorded
        This is helpful if `joint_dataloader` also returns useful non-input elements
    
    return_latents: bool, optional
        If true, return a dict containing the latent representation, mean, and variance.
    
    progress_bar: bool, optional
        If true, displace training progress bar
    
    Returns
    -------
    record: list, available if `record_idx` is not empty
        Data loaded, see above
    latent_info: dict, available if `return_latents` is True
        Contains `latent`, `mu`, and `var` returned by the model for each batch
    '''
    
    record = [[] for _ in range(len(record_idx))] # Empty if len(record_idx) == 0
    returns = {'latent': [[] for _ in range(model.M)], 
               'mu': [[] for _ in range(model.M)], 
               'var': [[] for _ in range(model.M)]}
    
    model.eval()
    
    with torch.no_grad():
        for k, data in tqdm(enumerate(joint_dataloader), disable=not progress_bar):
            results = model([data[i].to(device=device, dtype=dtype) for i in range(model.M)])

            _log(results, 'test', writer, epoch * len(joint_dataloader) + k)
            
            if return_latents:
                for i in range(model.M):
                    returns['latent'][i].append(results['z'][i].cpu().numpy())
                    returns['mu'][i].append(results['mu'][i].cpu().numpy())
                    returns['var'][i].append(results['var'][i].cpu().numpy())
            
            for i, j in enumerate(record_idx): # Does not iterate if empty
                record[i].append(data[j])
            
    ret_buffer = []
    if len(record_idx) > 0:
        ret_buffer.append(record)
    if return_latents:
        ret_buffer.append(returns)
    return tuple(ret_buffer)


def save_latent_info(latent_info, path):
    '''
    Save the latent information collected by `train` or `test`
    '''
    for key, items in latent_info.items():
        for i, item in enumerate(items):
            np.save(path + key + str(i + 1) + '.npy', 
                    np.vstack(latent_info[key][i]).astype(np.float32))
            

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch


# The code is adapted from https://github.com/iffsid/mmvae, the repository for the work
# Y. Shi, N. Siddharth, B. Paige and PHS. Torr.
# Variational Mixture-of-Experts Autoencoders for Multi-Modal Deep Generative Models.
# In Proceedings of the 33rd International Conference on Neural Information Processing Systems,
# Page 15718–15729, 2019

def pdist(sample_1, sample_2, eps=1e-8):
    """Compute the matrix of all squared pairwise distances. Code
    adapted from the torch-two-sample library (added batching).
    You can find the original implementation of this function here:
    https://github.com/josipd/torch-two-sample/blob/master/torch_two_sample/util.py
    Arguments
    ---------
    sample_1 : torch.Tensor or Variable
        The first sample, should be of shape ``(batch_size, n_1, d)``.
    sample_2 : torch.Tensor or Variable
        The second sample, should be of shape ``(batch_size, n_2, d)``.
    norm : float
        The l_p norm to be used.
    batched : bool
        whether data is batched
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (batch_size, n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    if len(sample_1.shape) == 2:
        sample_1, sample_2 = sample_1.unsqueeze(0), sample_2.unsqueeze(0)
    B, n_1, n_2 = sample_1.size(0), sample_1.size(1), sample_2.size(1)
    norms_1 = torch.sum(sample_1 ** 2, dim=-1, keepdim=True)
    norms_2 = torch.sum(sample_2 ** 2, dim=-1, keepdim=True)
    norms = (norms_1.expand(B, n_1, n_2)
             + norms_2.transpose(1, 2).expand(B, n_1, n_2))
    distances_squared = norms - 2 * sample_1.matmul(sample_2.transpose(1, 2))
    return torch.sqrt(eps + torch.abs(distances_squared)).squeeze()  # batch x K x latent


def NN_lookup(emb_h, emb):
    dist = pdist(emb.to(emb_h.device), emb_h)
    return dist, dist.argmin(dim=0)