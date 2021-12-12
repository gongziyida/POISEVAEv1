import torch
import torch.nn as nn
import gibbs_sampler_poise
import kl_divergence_calculator

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _latent_dims_type_setter(lds):
    ret = []
    for ld in lds:
        if hasattr(ld, '__iter__'): # Iterable
            ld_tuple = tuple([i for i in l])
            if not all(map(lambda i: isinstance(i, int), ld_tuple)):
                raise ValueError('`latent_dim` must be either iterable of ints or int.')
            ret.append(ld_tuple)
        elif isinstance(ld, int):
            ret.append((ld, ))
        else:
            raise ValueError('`latent_dim` must be either iterable of ints or int.')
    return ret


class VAE(nn.Module):
    def __init__(self, encoders, decoders, batch_size, latent_dims=None, use_mse_loss=True,
                 device=_device):
        """
        encoders: list of nn.Module
        decoders: list of nn.Module
        """
        super(VAE,self).__init__()

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
        self.latent_dims = _latent_dims_type_setter(self.latent_dims)

        self.encoders = encoders
        self.decoders = decoders

        self.batch_size = batch_size
        self.use_mse_loss = use_mse_loss

        self.gibbs = gibbs_sampler_poise.gibbs_sampler(self.latent_dims, batch_size)
        self.kl_div = kl_divergence_calculator.kl_divergence(self.latent_dims, batch_size)

        self.register_parameter(name='g11', param = nn.Parameter(torch.randn(*self.latent_dims)))
        self.register_parameter(name='g22', param = nn.Parameter(torch.randn(*self.latent_dims)))
        self.flag_initialize = 1
        # self.g12 = torch.zeros(*self.latent_dims).to(device) # TODO: This can be rm. see below

    def _decoder_helper(self):
        ret = []
        for decoder, z, ld in zip(self.decoders, self.z_gibbs_posteriors, self.latent_dims):
            z = z.view(*ld) # Match the shape to the output
            x_ = decoder(z)
            if not self.use_mse_loss: # BCE instead
                x_ = torch.sigmoid(x_)
            ret.append(x_)
        return x_

    def forward(self, x): # TODO: gibbs_sampler and kl_divergence_calculator not yet finished
        mu, var = [], []
        for i, xi in enumerate(x):
            _mu, _log_var = self.encoders[i].forward(xi)
            mu.append(_mu)
            var.append(-torch.exp(_log_var)) # TODO why - in front of the exp

        g22 = -torch.exp(self.g22)

        # Initializing gibbs sample
        if self.flag_initialize==1:
            # TODO: function signature of gibbs_sample: optional parameters
            # flag_init. not necessary; if z not provided, init. z rand.ly
            # Not really an optimization but make the code clear
            # in case people want to look carefully in the future
            # I made an attempt in the local file `gibbs_sampler_poise.py`; debugging needed
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

        # TODO: replace the three lines with block_diag if it is valid
        # G1 = torch.cat((self.g11, self.g12), 0)
        # G2 = torch.cat((self.g12, g22), 0)
        # G  = torch.cat((G1, G2), 1)
        G = torch.block_diag(self.g11, self.g22)

        # TODO: make it generic: the users may pass different decoders,
        # with or without ConvT. input layer
        # I think the best way is to ask them specify latent_dim,
        # and reshape/expand the tensors as needed.
        # We should require that, e.g., `latent_dim = 16` != `latent_dim = (16, 1, 1)`.

        # self.z2_gibbs_posterior = self.z2_gibbs_posterior.unsqueeze(2)
        # self.z2_gibbs_posterior = self.z2_gibbs_posterior.unsqueeze(3)
        # The line below is equivalent to the two above
        # self.z2_gibbs_posterior = self.z2_gibbs_posterior[..., None, None]

        # if self.use_mse_loss:
        #     reconstruction1 = self.set1_dec3(x1)
        #     reconstruction2 = (self.set2_dec4(x2)).view(-1,3072)
        # else:
        #     reconstruction1 = torch.sigmoid(self.set1_dec3(x1))
        #     reconstruction2 = torch.sigmoid((self.set2_dec4(x2)).view(-1,3072))

        x_ = _decoder_helper() # Decoding

        # self.z2_gibbs_posterior = self.z2_gibbs_posterior.squeeze()
        for i in range(len(self.z_gibbs_posteriors)):
            self.z_gibbs_posteriors[i] = self.z_gibbs_posteriors[i].squeeze()

        # KL loss
        kls = self.kl_div.calc(G,self.z_gibbs_posteriors, self.z_gibbs_priors, mu,var)
        KL_loss  = sum(kls)

        # Reconstruction loss
        loss_func = nn.MSELoss(reduction='sum') if self.use_mse_loss else nn.BCELoss(reduction='sum')
        recs = list(map(lambda x: loss_func(x[0], x[1]), zip(x_, x)))
        rec_loss = sum(recs)

        loss = KL_loss + rec_loss

        # if self.use_mse_loss:
        #     mse_loss = nn.MSELoss(reduction='sum')
        #     MSE1 = mse_loss(reconstruction1, data1)
        #     MSE2 = mse_loss(reconstruction2, data2)
        # else:
        #     bce_loss = nn.BCELoss(reduction='sum')
        #     MSE1 = bce_loss(reconstruction1, data1)
        #     MSE2 = bce_loss(reconstruction2, data2)

        # return self.z1_posterior,self.z2_posterior,reconstruction1,reconstruction2,mu1,var1,mu2,var2,loss, MSE1, MSE2, KLD
        return self.z_posteriors, x_, mu, var,loss, recs, kls
