import torch
import torch.nn as nn
from gibbs_sampler_poise import gibbs_sampler
from kl_divergence_calculator import kl_divergence
from numpy import prod

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _latent_dims_type_setter(lds): # version 1.0
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


class POISEVAE(nn.Module):
    __version__ = 1.0
    
    def __init__(self, encoders, decoders, batch_size, latent_dims=None, use_mse_loss=True,
                 device=_device):
        """
        encoders: list of nn.Module
            Each encoder must have an attribute `latent_dim` specifying the dimension of the
            latent space to which it encodes. An alternative way to avoid adding this attribute
            is to specify the `latent_dims` parameter (see below). 
            Note that each `latent_dim` must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
            
        decoders: list of nn.Module
            The number and indices of decoders must match those of encoders.
            
        batch_size: int
        
        latent_dims: iterable, optional; default None
            The dimensions of the latent spaces to which the encoders encode. The indices of the 
            entries must match those of encoders. An alternative way to specify the dimensions is
            to add the attribute `latent_dim` to each encoder (see above).
            Note that each entry must be unsqueezed, e.g. (10, ) is not the same as (10, 1).
        
        use_mse_loss: boolean, optional; default True
            To use MSE loss or not; if not, BCE loss will be used.
        
        device: torch.device, optional
        """
        super(POISEVAE,self).__init__()
        self.__version__ = 1.0

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

        self.encoders = encoders
        self.decoders = decoders

        self.batch_size = batch_size
        self.use_mse_loss = use_mse_loss
        
        self.device = device

        self.gibbs = gibbs_sampler(self.latent_dims_flatten, batch_size)
        self.kl_div = kl_divergence(self.latent_dims_flatten, batch_size)

        self.register_parameter(name='g11', 
                                param=nn.Parameter(torch.randn(*self.latent_dims_flatten, 
                                                               device=self.device)))
        self.register_parameter(name='g22', 
                                param=nn.Parameter(torch.randn(*self.latent_dims_flatten, 
                                                               device=self.device)))
        self.flag_initialize = 1

    def _decoder_helper(self):
        """
        Reshape samples drawn from each latent space, and decode with considering the loss function
        """
        ret = []
        for decoder, z, ld in zip(self.decoders, self.z_gibbs_posteriors, self.latent_dims):
            z = z.view(self.batch_size, *ld) # Match the shape to the output
            x_ = decoder(z)
            if not self.use_mse_loss: # BCE instead
                x_ = torch.sigmoid(x_)
            ret.append(x_)
        return ret

    def forward(self, x):
        mu, var = [], []
        for i, xi in enumerate(x):
            _mu, _log_var = self.encoders[i].forward(xi)
            mu.append(_mu.view(self.batch_size, -1))
            var.append(-torch.exp(_log_var.view(self.batch_size, -1)))

        g22 = -torch.exp(self.g22)

        # Initializing gibbs sample
        if self.flag_initialize == 1:
            z_priors = self.gibbs.sample(self.g11, g22, n_iterations=5000)
            z_posteriors = self.gibbs.sample(self.g11, g22, lambda1s=mu, lambda2s=var,
                                             n_iterations=5000)

            self.z_priors = z_priors
            self.z_posteriors = z_posteriors
            self.flag_initialize = 0

        z_priors = list(map(lambda z: z.detach(), self.z_priors))
        z_posteriors = list(map(lambda z: z.detach(), self.z_posteriors))

        # If lambda not provided, treat as zeros to save memory and computation
        self.z_gibbs_priors = self.gibbs.sample(self.g11, g22, z=z_priors, n_iterations=5)
        self.z_gibbs_posteriors = self.gibbs.sample(self.g11, g22, lambda1s=mu, lambda2s=var,
                                                    z=z_posteriors, n_iterations=5)

        self.z_priors = list(map(lambda z: z.detach(), self.z_gibbs_priors))
        self.z_posteriors = list(map(lambda z: z.detach(), self.z_gibbs_posteriors))

        G = torch.block_diag(self.g11, self.g22)

        x_ = self._decoder_helper() # Decoding

        # self.z2_gibbs_posterior = self.z2_gibbs_posterior.squeeze()
        for i in range(len(self.z_gibbs_posteriors)):
            self.z_gibbs_posteriors[i] = self.z_gibbs_posteriors[i].squeeze()

        # KL loss
        kls = self.kl_div.calc(G, self.z_gibbs_posteriors, self.z_gibbs_priors, mu,var)
        KL_loss  = sum(kls)

        # Reconstruction loss
        loss_func = nn.MSELoss(reduction='sum') if self.use_mse_loss else nn.BCELoss(reduction='sum')
        recs = list(map(lambda x: loss_func(x[0], x[1]), zip(x_, x)))
        rec_loss = sum(recs)
        
        # Total loss
        loss = KL_loss + rec_loss

        return self.z_posteriors, x_, mu, var, loss, recs, KL_loss