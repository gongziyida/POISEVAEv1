from .POISE_VAE_is import POISEVAE as POISEVAE_IS
from .POISE_VAE_gibbs_old import POISEVAE as POISEVAE_Gibbs_old
from .POISE_VAE_gibbs_autograd import POISEVAE as POISEVAE_Gibbs_autograd
from .POISE_VAE_gibbs_gradient import POISEVAE as POISEVAE_Gibbs_gradient
import poisevae._debug
import poisevae.utils
import poisevae.networks
from .pixel_cnn_wrapper import Wrapper 

def POISEVAE_Gibbs(variant, *args, **kwargs):
    if variant == 'autograd':
        model = POISEVAE_Gibbs_autograd(*args, **kwargs)
    elif variant == 'gradient':
        model = POISEVAE_Gibbs_gradient(*args, **kwargs)
    elif variant == 'old':
        model = POISEVAE_Gibbs_old(*args, **kwargs)
    else: 
        raise ValueError('variant')
    return model