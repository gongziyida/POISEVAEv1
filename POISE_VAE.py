import torch
import torch.nn as nn
from .gibbs_sampler_poise import GibbsSampler
from .kl_divergence_calculator import KLD
from numpy import prod

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


def _loss_func_type_setter(loss_funcs, num):
    if callable(loss_funcs):
        ret = [loss_funcs] * num
    elif hasattr(loss_funcs, '__iter__'):
        if len(loss_funcs) != num:
            raise ValueError('Unmatched number of loss functions and datasets')
        ret = loss_funcs
    else:
        raise TypeError('`loss_funcs` must be a function or list of functions.')
    return ret

class POISEVAE(nn.Module):
    __version__ = 5.0
    
    def __init__(self, encoders, decoders, loss_funcs, latent_dims=None, batch=True,
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
        
        loss_funcs: function or list of functions
            The users should properly restrict the range of the output of their decoders for the loss chosen.
        
        latent_dims: iterable, optional; default None
            The dimensions of the latent spaces to which the encoders encode. The indices of the 
            entries must match those of encoders. An alternative way to specify the dimensions is
            to add the attribute `latent_dim` to each encoder (see above).
            Note that each entry must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
        
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

        self.batch = batch
        self._batch_size = -1 # init
            
        self.loss = _loss_func_type_setter(loss_funcs, self.M)
        
        self.encoders = nn.ModuleList(encoders)
        self.decoders = nn.ModuleList(decoders)
        
        self.device = device

        self.gibbs = GibbsSampler(self.latent_dims_flatten)
        self.kl_div = KLD(self.latent_dims_flatten)

        self.register_parameter(name='g11', 
                                param=nn.Parameter(torch.randn(*self.latent_dims_flatten, 
                                                               device=self.device)))
        self.register_parameter(name='g22', 
                                param=nn.Parameter(torch.randn(*self.latent_dims_flatten, 
                                                               device=self.device)))
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
        for i, xi in enumerate(x):
            _mu, _log_var = self.encoders[i](xi)
            batch_size = xi.shape[0] if self.batch else 1
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
        for decoder, zi, ld in zip(self.decoders, z, self.latent_dims):
            batch_size = zi.shape[0] if self.batch else 1
            zi = zi.view(batch_size, *ld) # Match the shape to the output
            x_ = decoder(zi)
            x_rec.append(x_)
        return x_rec
    
    def _init_gibbs(self, g22, mu, var, n_iterations=5000):
        """
        Initialize the starting points for Gibbs sampling
        """
        batch_size = mu[0].shape[0] if self.batch else 1
        self._batch_size = batch_size
        z_priors = self.gibbs.sample(self.g11, g22, 
                                     n_iterations=n_iterations, batch_size=batch_size)
        z_posteriors = self.gibbs.sample(self.g11, g22, lambda1s=mu, lambda2s=var,
                                         n_iterations=n_iterations, batch_size=batch_size)

        self.z_priors = z_priors
        self.z_posteriors = z_posteriors
        self.flag_initialize = 0
        
    def _sampling(self, g22, mu, var, n_iterations=5):
        z_priors = [z.detach() for z in self.z_priors]
        z_posteriors = [z.detach() for z in self.z_posteriors]

        z_gibbs_priors = self.gibbs.sample(self.g11, g22, z=z_priors, n_iterations=n_iterations)

        z_gibbs_posteriors = self.gibbs.sample(self.g11, g22, lambda1s=mu, lambda2s=var,
                                               z=z_posteriors, n_iterations=n_iterations)


        # For calculating the loss and future use
        self.z_priors = [z.detach() for z in z_gibbs_priors]
        self.z_posteriors = [z.detach() for z in z_gibbs_posteriors]
        
        return z_gibbs_priors, z_gibbs_posteriors
        
    def _loss_func(self, reduction='sum'):
        if self.loss == 'MSE':
            return nn.MSELoss(reduction=reduction)
        elif self.loss == 'BCE':
            return nn.BCLoss(reduction=reduction)
        elif self.loss == 'CE':
            return nn.CrossEntropyLoss(reduction=reduction)
        else:
            raise NotImplementedError
            
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
        batch_size = x[0].shape[0] if self.batch else 1
        if batch_size != self._batch_size:
            self.flag_initialize = 1 # for the last batch whose size is often different
        
        mu, var = self.encode(x)
        
        g22 = -torch.exp(self.g22)

        # Initializing gibbs sample
        if self.flag_initialize == 1:
            self._init_gibbs(g22, mu, var) # self.z_priors and .z_posteriors are now init.ed
        # Actual sampling
        # try:
        z_gibbs_priors, z_gibbs_posteriors = self._sampling(g22, mu, var, n_iterations=5)
        # except RuntimeError:
        #     print(mu[0].shape, var[0].shape, self.g11.shape, x[0].shape)
        #     print(mu[1].shape, var[1].shape, self.g11.shape, x[1].shape)

        x_ = self.decode(z_gibbs_posteriors) # Decoding
        
        G = torch.block_diag(self.g11, g22)

        # KL loss
        kls = self.kl_div.calc(G, z_gibbs_posteriors, z_gibbs_priors, mu,var)
        KL_loss  = torch.sum(torch.stack(kls))

        # Reconstruction loss
        recs = [loss_func(x_[i], x[i]) for i, loss_func in enumerate(self.loss)]
        rec_loss = torch.sum(torch.stack(recs))
        
        # Total loss
        total_loss = KL_loss + rec_loss

        results = {
            'z': self.z_posteriors, 'x_rec': x_, 'mu': mu, 'var': var, 
            'total_loss': total_loss, 'rec_losses': recs, 'KL_loss': KL_loss
        }

        return results
