import torch
import torch.nn as nn
from .gibbs_sampler_poise import GibbsSampler
from .kl_divergence_calculator import KLDDerivative, KLDN01
from .gradient import KLGradient, RecGradient
from numpy import prod, sqrt

from time import time

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _latent_dims_type_setter(lds):
    ret, ret_flatten = [], []
    for ld in lds:
        if hasattr(ld, '__iter__'): # Iterable
            ld_tuple = tuple([i for i in ld])
            if not all(map(lambda i: isinstance(i, int), ld_tuple)):
                raise ValueError('`latent_dim` must be either iterable of ints or int.')
            ret.append(ld_tuple)
            ret_flatten.append(int(prod(ld_tuple)))
        elif isinstance(ld, int):
            ret.append((ld, ))
            ret_flatten.append(ld)
        else:
            raise ValueError('`latent_dim` must be either iterable of ints or int.')
    return ret, ret_flatten


def _func_type_setter(func, num, fname, concept):
    if callable(func):
        ret = [func] * num
    elif hasattr(func, '__iter__'):
        if len(func) != num:
            raise ValueError('Unmatched number of %s and datasets' % concept)
        ret = func
    else:
        raise TypeError('`%s` must be callable or list of callables.' % fname)
    return ret


class POISEVAE(nn.Module):
    __version__ = 10.0 # integrate different gibbs and kl calc.or
    
    def __init__(self, encoders, decoders, loss_funcs=None, likelihoods=None, latent_dims=None, 
                 rec_weights=None, reduction='mean', mask_missing=None, missing_data=None, 
                 batched=True, batch_size=-1, enc_config='nu', KL_calc='derivative', fix_t=True, 
                 device=_device):
        """
        Parameters
        ----------
        encoders: list of nn.Module
            Each encoder must have an attribute `latent_dim` specifying the dimension of the
            latent space to which it encodes. An alternative way to avoid adding this attribute
            is to specify the `latent_dims` parameter (see below). 
            Note that each `latent_dim` must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
            For now the model only support Gaussian distributions of the encodings. 
            The encoders must output the mean and log variance of the Gaussian distributions.
            
        decoders: list of nn.Module
            The number and indices of decoders must match those of encoders.
        
        loss_funcs: function or list of functions; default None
            The users should properly restrict the range of the output of their decoders for the loss chosen.
            If not given, `likelihoods` must be specified.
        
        likelihoods: class or list of classes, optional; default None
            The likelihood distributions. 
            The class(es) given must take exactly the output of the corresponding decoders given, and
            must contain `log_prob(target)` method to calculate the log probabilities. 
            If given, `loss_funcs` will be ignored and the loss is calculated with log probabilities.
        
        latent_dims: iterable of int, optional; default None
            The dimensions of the latent spaces to which the encoders encode. The indices of the 
            entries must match those of encoders. An alternative way to specify the dimensions is
            to add the attribute `latent_dim` to each encoder (see above).
            Note that each entry must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
        
        rec_weights: iterable of float, optional; default None
            The weights of the reconstruction loss of each modality
            
        reduction: str, optional; default 'mean'
            How to calculate the batch loss; either 'sum' or 'mean'
            
        mask_missing: callable, optional; default None
            Must be of the form `mask_missing(data)` and return the masked data
            The missing data should be None, while the present data should have the same data structures.
            
        missing_data: optional; default None
            How to fill in missing data; None treated as 0
        
        batched: bool, default True
            If the data is in batches
            
        batch_size: int, default -1
            Default: automatically determined
        
        enc_config: str, default 'nu'
            The definition of the encoder output, either 'nu' or 'mu/var'
        
        KL_calc: str, default 'derivative'
            'derivative' or 'std_normal'
        
        device: torch.device, optional
        """
        super(POISEVAE,self).__init__()

        # Modality check
        if len(encoders) != len(decoders):
            raise ValueError('The number of encoders must match that of decoders.')
        if len(encoders) > 2:
            raise NotImplementedError('> 3 latent spaces not yet supported.')
        
        # Type check
        if not all(map(lambda x: isinstance(x, nn.Module), (*encoders, *decoders))):
            raise TypeError('`encoders` and `decoders` must be lists of `nn.Module` class.')

        # Flag check
        if enc_config not in ('nu', 'mu/var', 'mu/nu2'):
            raise ValueError('`enc_config` value unreconized.')
        if KL_calc not in ('derivative', 'std_normal'): 
            raise ValueError('`KL_calc` value unreconized.')
        if reduction not in ('mean', 'sum'):
            raise NotImplementedError
            
        # Get the latent dimensions
        if latent_dims is not None:
            if not hasattr(latent_dims, '__iter__'): # Iterable
                raise TypeError('`latent_dims` must be iterable.')
            self.latent_dims = latent_dims
        else:
            self.latent_dims = tuple(map(lambda l: l.latent_dim, encoders))
        self.latent_dims, self.latent_dims_flatten = _latent_dims_type_setter(self.latent_dims)
        self.M = len(self.latent_dims)

        self.mask_missing = mask_missing
        self.missing_data = missing_data
        
        self.batched = batched
        self._batch_size = batch_size # init
            
        if likelihoods is None:
            self.loss = _func_type_setter(loss_funcs, self.M, 'loss_funcs', 'loss functions')
        elif loss_funcs is None:
            self.likelihoods = _func_type_setter(likelihoods, self.M, 
                                                 'likelihoods', 'likelihood distributions')
        else: 
            raise ValueError('One of `loss_funcs` and `likelihoods` must be given.')
        
        self.rec_weights = rec_weights
        
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        
        self.enc_config = enc_config
        self.device = device
        self.reduction = reduction
        
        self.gibbs = GibbsSampler(self.latent_dims_flatten, enc_config=enc_config, 
                                  device=self.device)
        
        if KL_calc == 'derivative':
            self.kl_div = KLDDerivative(self.latent_dims_flatten, reduction=self.reduction, 
                                        enc_config=enc_config, device=self.device)
        elif KL_calc == 'std_normal':
            self.kl_div = KLDN01(self.latent_dims_flatten, reduction=self.reduction, 
                                 enc_config=enc_config, device=self.device)
        
        
        self.g11 = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g22_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g12_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g21_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        if fix_t:
            self.t1 = [torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)]
            self.t2_hat = [torch.log(torch.tensor([0.5], device=self.device)), 
                           torch.log(torch.tensor([0.5], device=self.device))]
        else:
            self.t1 = nn.ParameterList([nn.Parameter(torch.randn(ld, device=self.device)) 
                                        for ld in self.latent_dims_flatten])
            self.t2_hat = nn.ParameterList([nn.Parameter(torch.randn(ld, device=self.device)) 
                                            for ld in self.latent_dims_flatten])
        
        
    def set_mask_missing(self, mask_missing):
        self.mask_missing = mask_missing
    
    def encode(self, x):
        """
        Encode the samples from multiple sources
        Parameter
        ---------
        x: list of torch.Tensor
        Return
        ------
        z: list of torch.Tensor
        """
        param1, param2 = [], []
        param1_, param2_ = [], []
        
        for i, xi in enumerate(x):
            if xi is None:
                param1.append(self.missing_data)
                param2.append(self.missing_data)
                param1_.append(self.missing_data)
                param2_.append(self.missing_data)
            else:
                batch_size = xi.shape[0] if self.batched else 1
                ret = self.encoders[i](xi)
                param1.append(ret[0].view(batch_size, -1))
                sign = 1 if self.enc_config == 'mu/var' else -1
                # param2.append(sign * torch.exp(ret[1].view(batch_size, -1)))
                param2.append(sign * nn.functional.softplus(ret[1].view(batch_size, -1)))
                if len(ret) == 2:
                    param1_.append(None)
                    param2_.append(None)
                elif len(ret) == 4:
                    param1_.append(ret[2].view(batch_size, -1))
                    # param2_.append(sign * torch.exp(ret[3].view(batch_size, -1)))
                    param2_.append(sign * nn.functional.softplus(ret[3].view(batch_size, -1)))
        return param1, param2, param1_, param2_
    
    def decode(self, z):
        """
        Unsqueeze the samples from each latent space (if necessary), and decode
        Parameter
        ---------
        z: list of torch.Tensor
        Return
        ------
        x_rec: list of torch.Tensor
        """
        x_rec = []
        batch_size = z[0].shape[0] if self.batched else 1
        Gibbs_dim = len(z[0].shape) == 2 + self.batched
        if Gibbs_dim:
            n_samples = z[0].shape[self.batched]
            z = [zi.flatten(0, 1) for zi in z]
        for decoder, zi, ld in zip(self.decoders, z, self.latent_dims):
            zi = zi.view(batch_size * n_samples, *ld) # Match the shape to the output
            x_ = decoder(zi)
            if Gibbs_dim: # Gibbs dimension
                x_ = (x_[0].view(batch_size, n_samples, *x_[0].shape[1:]), x_[1])
            x_rec.append(x_)
        return x_rec
        
    def _sampling(self, G, param1, param2, param1_, param2_, t2, n_iterations=30):
        batch_size = self._fetch_batch_size(param1)
        self._batch_size = batch_size
        
        z_priors, T_priors = self.gibbs.sample(G, t1=self.t1, t2=t2, 
                                               batch_size=batch_size, n_iterations=n_iterations)
        
        if self.enc_config == 'nu':
            z_posteriors, T_posteriors = self.gibbs.sample(G, nu1=param1, nu2=param2, 
                                                           nu1_=param1_, nu2_=param2_, 
                                                           batch_size=batch_size,
                                                           t1=self.t1, t2=t2, 
                                                           n_iterations=n_iterations)
            kl = self.kl_div.calc(G, z_posteriors, z_priors, 
                                  nu1=param1, nu2=param2, nu1_=param1_, nu2_=param2_)
            
        elif self.enc_config == 'mu/var':
            z_posteriors, T_posteriors = self.gibbs.sample(G, mu=param1, var=param2, 
                                                           mu_=param1_, var_=param2_, 
                                                           batch_size=batch_size,
                                                           t1=self.t1, t2=t2, 
                                                           n_iterations=n_iterations)
            kl = self.kl_div.calc(G, z_posteriors, z_priors, 
                                  mu=param1, var=param2, mu_=param1_, var_=param2_)
        elif self.enc_config == 'mu/nu2':
            z_posteriors, T_posteriors = self.gibbs.sample(G, mu=param1, nu2=param2, 
                                                           mu_=param1_, nu2_=param2_, 
                                                           batch_size=batch_size,
                                                           t1=self.t1, t2=t2, 
                                                           n_iterations=n_iterations)
            kl = self.kl_div.calc(G, z_posteriors, z_priors, 
                                  mu=param1, nu2=param2, mu_=param1_, nu2_=param2_)
            # if param1[0] is not None and param1[1] is not None:
            #     assert torch.isnan(param1[0]).sum() == 0
            #     assert torch.isnan(-0.5 / param2[0]).sum() == 0
            
        return z_posteriors, kl
            
    def get_G(self):
        g22 = -torch.exp(self.g22_hat)
        g12 = 2 / sqrt(self.latent_dims_flatten[0]) * \
              torch.exp(self.g22_hat / 2 + self.t2_hat[1].unsqueeze(0) / 2) * \
              torch.tanh(self.g12_hat) * 0.99
        g21 = 2 / sqrt(self.latent_dims_flatten[1]) * \
              torch.exp(self.g22_hat / 2 + self.t2_hat[0].unsqueeze(1) / 2) * \
              torch.tanh(self.g21_hat) * 0.99
        G = torch.cat((torch.cat((self.g11, g12), 1), torch.cat((g21, g22), 1)), 0)
        return G * 0
    
    def get_t(self):
        t2 = [-torch.exp(t2_hat) for t2_hat in self.t2_hat]
        return self.t1, t2
    
    def forward(self, x, n_gibbs_iter=15, kl_weight=1, detach_G=False):
        """
        Return
        ------
        results: dict
            z: list of torch.Tensor
                Samples from the posterior distributions in the corresponding latent spaces
            x_rec: list of torch.Tensor
                Reconstructed samples
            param1: list of torch.Tensor
                Posterior distribution parameter 1, either nu1 or mean, determined by `enc_config`
            param2: list of torch.Tensor
                Posterior distribution parameter 2, either nu2 or variance, determined by `enc_config`
            total_loss: torch.Tensor
            rec_losses: list of torch.tensor
                Reconstruction loss for each dataset
            KL_loss: torch.Tensor
        """
        batch_size = self._fetch_batch_size(x)
        
        if self.mask_missing is not None:
            param1, param2, param1_, param2_ = self.encode(self.mask_missing(x))
        else:
            param1, param2, param1_, param2_ = self.encode(x)
        if param1[0] is not None and param1[1] is not None:
            print('nu1 max:', torch.abs(param1[0]).max().item(), 'nu1 mean:', torch.abs(param1[0]).mean().item())
            print('nu1p max:', torch.abs(param1[1]).max().item(), 'nu1p mean:', torch.abs(param1[1]).mean().item())
            print('nu2 min:', torch.abs(param2[0]).min().item(), 'nu2 mean:', torch.abs(param2[0]).mean().item())
            print('nu2p min:', torch.abs(param2[1]).min().item(), 'nu2p mean:', torch.abs(param2[1]).mean().item())
            assert torch.isnan(param1[0]).sum() == 0
        
        G = self.get_G().detach() if detach_G else self.get_G()
        _, t2 = self.get_t()
    
        z_posteriors, kl = self._sampling(G, param1, param2, param1_, param2_, t2, 
                                          n_iterations=n_gibbs_iter)
        
        assert torch.isnan(G).sum() == 0
        assert torch.isnan(z_posteriors[0]).sum() == 0
        assert torch.isnan(z_posteriors[1]).sum() == 0

        x_rec = self.decode(z_posteriors) # Decoding
        # assert torch.isnan(x_rec[0][0]).sum() == 0
        # assert torch.isnan(x_rec[1][0]).sum() == 0
        
        # Reconstruction loss term *for decoder*
        dec_rec_loss = 0
        if hasattr(self, 'loss'):
            recs = [loss_func(x_rec[i], x[i]) for i, loss_func in enumerate(self.loss)]
        else:
            recs = []
            for i in range(self.M):
                x_rec[i] = self.likelihoods[i](*x_rec[i])
                if x[i] is None:
                    recs.append(torch.tensor(0).to(self.device, G.dtype))
                else:
                    dims = list(range(2, len(x_rec[i].loc.shape)))
                    negative_loglike = -x_rec[i].log_prob(x[i].unsqueeze(1)).sum(dims)
                    if self.rec_weights is not None: # Modality weighting
                        negative_loglike *= self.rec_weights[i]
                    dec_rec_loss += negative_loglike
                    recs.append(negative_loglike.detach().sum()) # For loggging 
        
        # Total loss
        if x[0] is None and x[1] is None: # No rec loss
            total_loss = kl_weight * kl
        else:
            dec_rec_loss = dec_rec_loss.mean() if self.reduction == 'mean' else dec_rec_loss.sum()
            total_loss = kl_weight * kl + dec_rec_loss

        # These will then be used for logging only. Don't waste CUDA memory!
        # z_posteriors = [i[:, -1].detach().cpu() for i in z_posteriors]
        x_rec = [i.loc[:, -1].detach().cpu() for i in x_rec]
        param1 = [i.detach().cpu() if i is not None else None for i in param1]
        param2 = [i.detach().cpu() if i is not None else None for i in param2]
        param1_ = [i.detach().cpu() if i is not None else None for i in param1_]
        param2_ = [i.detach().cpu() if i is not None else None for i in param2_]
        results = {
            'z': z_posteriors, 'x_rec': x_rec, 'param1': param1, 'param2': param2,
            'total_loss': total_loss, 'rec_losses': recs, 'KL_loss': kl
        }
        if param1[0] is not None and param1[1] is not None:
            print('total loss:', total_loss.item(), 'kl term:', kl.item())
            print('rec1 loss:', recs[0].item() / batch_size / n_gibbs_iter, 
                  'rec2 loss:', recs[1].item() / batch_size / n_gibbs_iter)
            print()
        return results

    
    def generate(self, n_samples, n_gibbs_iter=15, return_dist=False):
        self._batch_size = n_samples
        G = self.get_G()
        _, t2 = self.get_t()
        
        nones = [None] * len(self.latent_dims)
        
        z_posteriors, kl = self._sampling(G, nones, nones, t2, n_iterations=n_gibbs_iter)
        x_rec = self.decode(z_posteriors)
        
        for i in range(self.M):
            x_rec[i] = self.likelihoods[i](*x_rec[i])
        if not return_dist:
            x_rec = [i.loc[:, -1].detach().cpu() for i in x_rec]
            
        results = {'z': z_posteriors, 'x_rec': x_rec}
        
        return results

    
    def _fetch_batch_size(self, x):
        if not self.batched:
            return 1
        for xi in x:
            if xi is not None:
                return xi.shape[0]
        return self._batch_size
