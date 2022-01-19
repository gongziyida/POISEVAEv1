import torch
import torch.nn as nn
from .gibbs_sampler_poise import GibbsSampler_Z, GibbsSampler_A
from .kl_divergence_calculator import KLD
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
    __version__ = 7.0 # g21 / g12
    
    def __init__(self, encoders, decoders, loss_funcs=None, likelihoods=None, latent_dims=None, 
                 rec_weights=None, reduction='mean', batched=True, batch_size=-1, device=_device):
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
        
        batch: bool
            If the data is in batches
        
        device: torch.device, optional
        """
        super(POISEVAE,self).__init__()

        if len(encoders) != len(decoders):
            raise ValueError('The number of encoders must match that of decoders.')
        
        if len(encoders) > 2:
            raise NotImplementedError('> 3 latent spaces not yet supported.')
        
        # Type check
        if not all(map(lambda x: isinstance(x, nn.Module), (*encoders, *decoders))):
            raise TypeError('`encoders` and `decoders` must be lists of `nn.Module` class.')

        # Get the latent dimensions
        if latent_dims is not None:
            if not hasattr(latent_dims, '__iter__'): # Iterable
                raise TypeError('`latent_dims` must be iterable.')
            self.latent_dims = latent_dims
        else:
            self.latent_dims = tuple(map(lambda l: l.latent_dim, encoders))
        self.latent_dims, self.latent_dims_flatten = _latent_dims_type_setter(self.latent_dims)
        
        self.M = len(latent_dims)

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
        
        self.device = device
        
        if reduction not in ('mean', 'sum'):
            raise ValueError('`reduction` must be either "mean" or "sum".')
        self.reduction = reduction
        
        self.gibbs = GibbsSampler_Z(self.latent_dims_flatten)
        # self.gibbs = GibbsSampler_A(*self.latent_dims_flatten, batch_size)
        self.kl_div = KLD(self.latent_dims_flatten, reduction=self.reduction)
        
        self.g11 = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g22_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g12_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g21_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        
        self.flag_initialize = 1
    
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
        mu, var = [], []
        batch_size = x[0].shape[0] if self.batched else 1
        for i, xi in enumerate(x):
            _mu, _log_var = self.encoders[i](xi)
            mu.append(_mu.view(batch_size, -1))
            var.append(-torch.exp(_log_var.view(batch_size, -1)))
        return mu, var
    
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
        for decoder, zi, ld in zip(self.decoders, z, self.latent_dims):
            zi = zi.view(batch_size, *ld) # Match the shape to the output
            x_ = decoder(zi)
            x_rec.append(x_)
        return x_rec
    
    def _init_gibbs(self, G, mu, var, n_iterations=5000):
        """
        Initialize the starting points for Gibbs sampling
        """
        batch_size = mu[0].shape[0] if self.batched else 1
        self._batch_size = batch_size
        
        # Gibbs Z
        z_priors = self.gibbs.sample(G, n_iterations=n_iterations, batch_size=batch_size)
        z_posteriors = self.gibbs.sample(G, lambda1s=mu, lambda2s=var,
                                         n_iterations=n_iterations, batch_size=batch_size)
        
        # Gibbs A
        # m1, m2 = G.shape[0]//2, G.shape[1]//2
        # g11, g22, g12, g21 = G[:m1, :m2], G[m1:, m2:], G[:m1, m2:], G[m1:, :m2]
        # z_priors = self.gibbs.sample(1, torch.zeros_like(mu[0]), torch.zeros_like(mu[1]), 
        #                              g11, g22, g12, g21,
        #                              torch.zeros_like(mu[0]), torch.zeros_like(var[0]),
        #                              torch.zeros_like(mu[1]), torch.zeros_like(var[1]),
        #                              n_iterations=5000)
        # z_posteriors = self.gibbs.sample(1, torch.zeros_like(mu[0]), torch.zeros_like(mu[1]), 
        #                                  g11, g22, g12, g21, 
        #                                  mu[0], var[0], mu[1], var[1], 
        #                                  n_iterations=5000)

        self.z_priors = z_priors
        self.z_posteriors = z_posteriors
        self.flag_initialize = 0
        
    def _sampling(self, G, mu, var, n_iterations=5):
        z_priors = [z.detach() for z in self.z_priors]
        z_posteriors = [z.detach() for z in self.z_posteriors]
        
        # Gibbs Z
        z_gibbs_priors = self.gibbs.sample(G, z=z_priors, n_iterations=n_iterations)
        z_gibbs_posteriors = self.gibbs.sample(G, lambda1s=mu, lambda2s=var,
                                               z=z_posteriors, n_iterations=n_iterations)

        # Gibbs A
        # m1, m2 = G.shape[0]//2, G.shape[1]//2
        # g11, g22, g12, g21 = G[:m1, :m2], G[m1:, m2:], G[:m1, m2:], G[m1:, :m2]
        # z_gibbs_priors = self.gibbs.sample(0, *z_priors, 
        #                                  g11, g22, g12, g21, 
        #                                  torch.zeros_like(mu[0]), torch.zeros_like(var[0]),
        #                                  torch.zeros_like(mu[1]), torch.zeros_like(var[1]),
        #                                  n_iterations=n_iterations)
        # z_gibbs_posteriors = self.gibbs.sample(0, *z_posteriors,
        #                                  g11, g22, g12, g21, 
        #                                  mu[0], var[0], mu[1], var[1], 
        #                                  n_iterations=n_iterations)

        # For calculating the loss and future use
        self.z_priors = [z.detach() for z in z_gibbs_priors]
        self.z_posteriors = [z.detach() for z in z_gibbs_posteriors]
        
        return z_gibbs_priors, z_gibbs_posteriors
            
    def forward(self, x):
        """
        Return
        ------
        results: dict
            z: list of torch.Tensor
                Samples from the posterior distributions in the corresponding latent spaces
            x_rec: list of torch.Tensor
                Reconstructed samples
            mu: list of torch.Tensor
                Posterior distribution means
            var: list of torch.Tensor
                Posterior distribution variances
            total_loss: torch.Tensor
            rec_losses: list of torch.tensor
                Reconstruction loss for each dataset
            KL_loss: torch.Tensor
        """
        batch_size = x[0].shape[0] if self.batched else 1
        if batch_size != self._batch_size:
            self.flag_initialize = 1 # for the last batch whose size is often different
        
        mu, var = self.encode(x)
        
        g22 = -torch.exp(self.g22_hat)
        g12 = 2 / sqrt(self.latent_dims_flatten[1]) * torch.exp(self.g22_hat / 2) * torch.tanh(self.g12_hat)
        g21 = 2 / sqrt(self.latent_dims_flatten[0]) * torch.exp(self.g22_hat / 2) * torch.tanh(self.g21_hat)
        G = torch.cat((torch.cat((self.g11, g12), 1), torch.cat((g21, g22), 1)), 0)

        # Initializing gibbs sample
        if self.flag_initialize == 1:
            self._init_gibbs(G, mu, var) # self.z_priors and .z_posteriors are now init.ed
        # Actual sampling
        z_gibbs_priors, z_gibbs_posteriors = self._sampling(G, mu, var, n_iterations=5)

        x_rec = self.decode(z_gibbs_posteriors) # Decoding

        # KL divergence term
        kls = self.kl_div.calc(G, z_gibbs_posteriors, z_gibbs_priors, mu, var)
        KL_loss  = kls[0] + kls[1] + kls[2]

        # Reconstruction loss term
        if hasattr(self, 'loss'):
            recs = [loss_func(x_rec[i], x[i]) for i, loss_func in enumerate(self.loss)]
        else:
            recs = []
            for i in range(self.M):
                x_rec[i] = self.likelihoods[i](*x_rec[i])
                negative_loglike = -x_rec[i].log_prob(x[i]).sum()
                if self.reduction == 'mean':
                    negative_loglike /= batch_size
                recs.append(negative_loglike)
                
        rec_loss = 0
        for i in range(self.M):
            rec_loss += recs[i] if self.rec_weights is None else self.rec_weights[i] * recs[i]
        
        # Total loss
        total_loss = KL_loss + rec_loss

        results = {
            'z': self.z_posteriors, 'x_rec': x_rec, 'mu': mu, 'var': var, 
            'total_loss': total_loss, 'rec_losses': recs, 'KL_loss': KL_loss
        }

        return results
