import torch
import torch.nn as nn
from .gibbs_sampler_poise_old import GibbsSampler_Z, GibbsSampler_A
from .kl_divergence_calculator import KLDDerivative
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
    __version__ = 8.0 # t1 / t2
    
    def __init__(self, encoders, decoders, loss_funcs=None, likelihoods=None, latent_dims=None, 
                 rec_weights=None, reduction='mean', mask_missing=None, missing_data=None, 
                 batched=True, fix_t=True, batch_size=-1, device=_device):
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
        
        self.device = device
        
        if reduction not in ('mean', 'sum'):
            raise ValueError('`reduction` must be either "mean" or "sum".')
        self.reduction = reduction
        
        self.gibbs = GibbsSampler_Z(self.latent_dims_flatten, device=self.device)
        # self.gibbs = GibbsSampler_A(*self.latent_dims_flatten, batch_size)
        self.kl_div = KLDDerivative(self.latent_dims_flatten, reduction=self.reduction, device=self.device)
        
        self.g11 = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g22_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g12_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        self.g21_hat = nn.Parameter(torch.randn(*self.latent_dims_flatten, device=self.device))
        if fix_t:
            self.t1 = [torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)]
            self.t2_hat = [torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)]
        else:
            self.t1 = nn.ParameterList([nn.Parameter(torch.randn(ld, device=self.device)) 
                                        for ld in self.latent_dims_flatten])
            self.t2_hat = nn.ParameterList([nn.Parameter(torch.randn(ld, device=self.device)) 
                                            for ld in self.latent_dims_flatten])
        
        self.flag_initialize = 1
        
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
        nu1, nu2 = [], []
        
        for i, xi in enumerate(x):
            if xi is None:
                nu1.append(self.missing_data)
                nu2.append(self.missing_data)
            else:
                batch_size = xi.shape[0] if self.batched else 1
                _nu1, _log_nu2 = self.encoders[i](xi)
                nu1.append(_nu1.view(batch_size, -1))
                nu2.append(-torch.exp(_log_nu2.view(batch_size, -1)))
        return nu1, nu2
    
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
    
    def _init_gibbs(self, G, nu1, nu2, t2, n_iterations=5000):
        """
        Initialize the starting points for Gibbs sampling
        """
        batch_size = self._fetch_batch_size(nu1)
        self._batch_size = batch_size
        
        with torch.no_grad(): # this is only to pick an init value for the real sampling
            # Gibbs Z
            z_priors = self.gibbs.sample(G, t1s=self.t1, t2s=t2, 
                                         n_iterations=n_iterations, batch_size=batch_size)
            z_posteriors = self.gibbs.sample(G, nu1s=nu1, nu2s=nu2, t1s=self.t1, t2s=t2,
                                             n_iterations=n_iterations, batch_size=batch_size)

            # Gibbs A
            # m1, m2 = G.shape[0]//2, G.shape[1]//2
            # g11, g22, g12, g21 = G[:m1, :m2], G[m1:, m2:], G[:m1, m2:], G[m1:, :m2]
            # z_priors = self.gibbs.sample(1, torch.zeros_like(nu1[0]), torch.zeros_like(nu1[1]), 
            #                              g11, g22, g12, g21,
            #                              torch.zeros_like(nu1[0]), torch.zeros_like(nu2[0]),
            #                              torch.zeros_like(nu1[1]), torch.zeros_like(nu2[1]),
            #                              n_iterations=5000)
            # z_posteriors = self.gibbs.sample(1, torch.zeros_like(nu1[0]), torch.zeros_like(nu1[1]), 
            #                                  g11, g22, g12, g21, 
            #                                  nu1[0], nu2[0], nu1[1], nu2[1], 
            #                                  n_iterations=5000)

        # assert torch.isnan(z_priors[0]).sum() == 0
        # assert torch.isnan(z_priors[1]).sum() == 0
        # assert torch.isnan(z_posteriors[0]).sum() == 0
        # assert torch.isnan(z_posteriors[1]).sum() == 0
        self.z_priors = [z.detach() for z in z_priors]
        self.z_posteriors = [z.detach() for z in z_posteriors]
        self.flag_initialize = 0
        
    def _sampling(self, G, nu1, nu2, t2, n_iterations=5):
        z_priors = [z.detach() for z in self.z_priors]
        z_posteriors = [z.detach() for z in self.z_posteriors]
        del self.z_priors, self.z_posteriors # Free memory
        
        # Gibbs Z
        z_gibbs_priors = self.gibbs.sample(G, t1s=self.t1, t2s=t2, z=z_priors, n_iterations=n_iterations)
        del z_priors 
        z_gibbs_posteriors = self.gibbs.sample(G, nu1s=nu1, nu2s=nu2, t1s=self.t1, t2s=t2,
                                               z=z_posteriors, n_iterations=n_iterations)
        del z_posteriors

        # Gibbs A
        # m1, m2 = G.shape[0]//2, G.shape[1]//2
        # g11, g22, g12, g21 = G[:m1, :m2], G[m1:, m2:], G[:m1, m2:], G[m1:, :m2]
        # z_gibbs_priors = self.gibbs.sample(0, *z_priors, 
        #                                  g11, g22, g12, g21, 
        #                                  torch.zeros_like(nu1[0]), torch.zeros_like(nu2[0]),
        #                                  torch.zeros_like(nu1[1]), torch.zeros_like(nu2[1]),
        #                                  n_iterations=n_iterations)
        # del z_priors # Free memory
        # z_gibbs_posteriors = self.gibbs.sample(0, *z_posteriors,
        #                                  g11, g22, g12, g21, 
        #                                  nu1[0], nu2[0], nu1[1], nu2[1], 
        #                                  n_iterations=n_iterations)
        # del z_posteriors

        # For calculating the loss and future use
        self.z_priors = [z.detach() for z in z_gibbs_priors]
        self.z_posteriors = [z.detach() for z in z_gibbs_posteriors]
        
        return z_gibbs_priors, z_gibbs_posteriors
            
    def get_G(self):
        g22 = -torch.exp(self.g22_hat)
        g12 = 2 / sqrt(self.latent_dims_flatten[0]) * \
              torch.exp(self.g22_hat / 2 + self.t2_hat[1].unsqueeze(0) / 2) * \
              torch.tanh(self.g12_hat)
        g21 = 2 / sqrt(self.latent_dims_flatten[1]) * \
              torch.exp(self.g22_hat / 2 + self.t2_hat[0].unsqueeze(1) / 2) * \
              torch.tanh(self.g21_hat)
        G = torch.cat((torch.cat((self.g11, g12), 1), torch.cat((g21, g22), 1)), 0)
        return G
    
    def get_t(self):
        t2 = [-torch.exp(t2_hat) for t2_hat in self.t2_hat]
        return self.t1, t2
    
    def forward(self, x, n_gibbs_iter=15):
        """
        Return
        ------
        results: dict
            z: list of torch.Tensor
                Samples from the posterior distributions in the corresponding latent spaces
            x_rec: list of torch.Tensor
                Reconstructed samples
            nu: list of torch.Tensor
                Posterior distribution nu1
            nu2: list of torch.Tensor
                Posterior distribution nu2
            total_loss: torch.Tensor
            rec_losses: list of torch.tensor
                Reconstruction loss for each dataset
            KL_loss: torch.Tensor
        """
        batch_size = self._fetch_batch_size(x)
        if batch_size != self._batch_size:
            self.flag_initialize = 1 # for the last batch whose size is often different
        
        if self.mask_missing is not None:
            nu1, nu2 = self.encode(self.mask_missing(x))
        else:
            nu1, nu2 = self.encode(x)
        
        G = self.get_G()
        _, t2 = self.get_t()
        

        # Initializing gibbs sample
        if self.flag_initialize == 1:
            self._init_gibbs(G, nu1, nu2, t2) # self.z_priors and .z_posteriors are now init.ed
        # Actual sampling
        z_gibbs_priors, z_gibbs_posteriors = self._sampling(G, nu1, nu2, t2, n_iterations=n_gibbs_iter)
        # assert torch.isnan(z_gibbs_priors[0]).sum() == 0
        # assert torch.isnan(z_gibbs_priors[1]).sum() == 0
        # assert torch.isnan(z_gibbs_posteriors[0]).sum() == 0
        # assert torch.isnan(z_gibbs_posteriors[1]).sum() == 0

        x_rec = self.decode(z_gibbs_posteriors) # Decoding

        # KL divergence term
        KL_loss = self.kl_div.calc(G, z_gibbs_posteriors, z_gibbs_priors, nu1, nu2)

        # Reconstruction loss term
        if hasattr(self, 'loss'):
            recs = [loss_func(x_rec[i], x[i]) for i, loss_func in enumerate(self.loss)]
        else:
            recs = []
            for i in range(self.M):
                x_rec[i] = self.likelihoods[i](*x_rec[i])
                if x[i] is None:
                    recs.append(torch.tensor(0).to(self.device, G.dtype))
                else:
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
            'z': self.z_posteriors, 'x_rec': x_rec, 'nu1': nu1, 'nu2': nu2, 
            'total_loss': total_loss, 'rec_losses': recs, 'KL_loss': KL_loss
        }

        return results

    
    def generate(self, n_samples, n_gibbs_iter=50):
        self._batch_size = n_samples
        G = self.get_G()
        _, t2 = self.get_t()
        
        nones = [None] * len(self.latent_dims)
        
        # Initializing gibbs sample
        if self.flag_initialize == 1:
            self._init_gibbs(G, nones, nones, t2) # self.z_priors and .z_posteriors are now init.ed
        # Actual sampling
        z_gibbs_priors, z_gibbs_posteriors = self._sampling(G, nones, nones, t2, n_iterations=n_gibbs_iter)

        x_rec = self.decode(z_gibbs_posteriors) # Decoding
        
        for i in range(self.M):
            x_rec[i] = self.likelihoods[i](*x_rec[i])
            
        results = {'z': z_gibbs_posteriors, 'x_rec': x_rec}
        
        return results

    
    def _fetch_batch_size(self, x):
        if not self.batched:
            return 1
        for xi in x:
            if xi is not None:
                return xi.shape[0]
        return self._batch_size
