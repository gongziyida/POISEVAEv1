# Version: 1.0
import os
import numpy as np
import torch
from tqdm import tqdm

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, joint_dataloader, optimizer, epoch, record_idx=(), return_latents=False, progress_bar=False, 
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
    loss: float
        Total loss
    rec_loss1, rec_loss2, ...: floats
        Reconstruction losses for different modalities
    kld_loss: float
        KL loss
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
    running_total, running_recs, running_kld  = 0, np.zeros(model.M), 0
    
    for i, data in enumerate(tqdm(joint_dataloader, disable=not progress_bar)):
        optimizer.zero_grad()
        
        results = model([data[i].to(device=device, dtype=dtype) for i in range(model.M)])
        
        for i, val in enumerate(results['rec_losses']):
            running_recs[i] += val.item()
        running_kld  += results['KL_loss'].item()
        running_total += results['total_loss'].item()

        results['total_loss'].backward() 
        optimizer.step()
        
        if return_latents:
                for i in range(model.M):
                    returns['latent'][i].append(results['z'][i].cpu().numpy())
                    returns['mu'][i].append(results['mu'][i].cpu().numpy())
                    returns['var'][i].append(results['var'][i].cpu().numpy())
                    
        for i, j in enumerate(record_idx): # Does not iterate if empty
            record[i].append(data[j])
        
    train_loss = running_total / (len(joint_dataloader.dataset))
    rec_losses = running_recs / (len(joint_dataloader.dataset))
    kld_loss = running_kld / (len(joint_dataloader.dataset))
    
    ret_buffer = [train_loss, *rec_losses, kld_loss]
    if len(record_idx) > 0:
        ret_buffer.append(record)
    if return_latents:
        ret_buffer.append(returns)
    return tuple(ret_buffer)



def test(model, joint_dataloader, epoch, record_idx=(), return_latents=False, progress_bar=False, 
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
    loss: float
        Total loss
    rec_loss1, rec_loss2, ...: floats
        Reconstruction losses for different modalities
    kld_loss: float
        KL loss
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
    running_total, running_recs, running_kld  = 0, np.zeros(model.M), 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(joint_dataloader, disable=not progress_bar)):
            results = model([data[i].to(device=device, dtype=dtype) for i in range(model.M)])

            for i, val in enumerate(results['rec_losses']):
                running_recs[i] += val.item()
            running_kld  += results['KL_loss'].item()
            running_total += results['total_loss'].item()
            
            if return_latents:
                for i in range(model.M):
                    returns['latent'][i].append(results['z'][i].cpu().numpy())
                    returns['mu'][i].append(results['mu'][i].cpu().numpy())
                    returns['var'][i].append(results['var'][i].cpu().numpy())
            
            for i, j in enumerate(record_idx): # Does not iterate if empty
                record[i].append(data[j])
            
    train_loss = running_total / (len(joint_dataloader.dataset))
    rec_losses = running_recs / (len(joint_dataloader.dataset))
    kld_loss = running_kld / (len(joint_dataloader.dataset))
    
    ret_buffer = [train_loss, *rec_losses, kld_loss]
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