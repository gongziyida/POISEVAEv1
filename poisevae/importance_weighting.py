import torch
import numpy as np

_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def sample_proposal(m1, m2, var, batch_size, n_IW_samples, device=_device):
    mn1 = torch.distributions.MultivariateNormal(torch.zeros(m1), var * torch.eye(m1))
    mn2 = torch.distributions.MultivariateNormal(torch.zeros(m2), var * torch.eye(m2))
    return [mn1.sample([batch_size, n_IW_samples]).to(device), mn2.sample([batch_size, n_IW_samples]).to(device)]


def IWq(G, z, nu1, nu2, var_proposal):
    """ Calc. the weights and expectations with importance weighting sampling.
        Assume `z`, `nu1` and `nu2` are list of tensors for different modalities.
        Assume `nu1` and `nu2` have shape (batch, latent dimension),
        and `z` has shape (batch, IW samples, latent dimension)
        $w \propto \exp(h + d) / \exp(r)$ where $h = H(z)$, $d = H(z, x) - H(z)$, 
        and $\exp(r)$ is the proposal PDF.
    """
    expand = lambda a, dim: a.unsqueeze(dim) if a is not None else 0
    
    # z = [z[i].unsqueeze(0) for i in range(len(z))] # -> (1, IW samples, lat. dim.)
    # nu -> (batch, 1, lat. dim.)
    nu1 = [expand(nu1[i], 1) for i in range(len(z))] 
    nu2 = [expand(nu2[i], 1) for i in range(len(z))]
    m = [z[i].shape[-1] for i in range(len(z))] # Latent dimensions
    z_sq = [z[i]**2 for i in range(len(z))]
    
    # TODO: make it generic
    # Memory efficiency
    h = -z_sq[0].sum(-1) - z_sq[1].sum(-1)
    assert torch.isnan(h).sum() == 0
    r = h / (var_proposal * 2) # log proposal
    
    h += (z[0] @ G[:m[0], :m[1]] * z[1]).sum(-1) # g11
    assert torch.isnan(h).sum() == 0, "%s" % G[:m[0], :m[1]]
    h += (z[0] @ G[:m[0], m[1]:] * z_sq[1]).sum(-1)  # g12
    assert torch.isnan(h).sum() == 0, "%s" % G[:m[0], m[1]:]
    h += (z_sq[0] @ G[m[0]:, :m[1]] * z[1]).sum(-1) # g21
    assert torch.isnan(h).sum() == 0, "%s" % G[m[0]:, :m[1]]
    h += (z_sq[0] @ G[m[0]:, m[1]:] * z_sq[1]).sum(-1) # g22
    assert torch.isnan(h).sum() == 0, "%s" % G[m[0]:, m[1]:]
    d = (nu1[0] * z[0]).sum(-1) + (nu1[1] * z[1]).sum(-1) + \
        (nu2[0] * z_sq[0]).sum(-1) + (nu2[1] * z_sq[1]).sum(-1)

    h = h + d
    assert torch.isnan(h).sum() == 0, "%s\n%s\n%s\n%s\n%s" % (d, nu1[0], nu1[1], nu2[0], nu2[1])
    with torch.no_grad():
        eps = 0 # -h.max(dim=-1).values.unsqueeze(-1) # precision rev
    
    log_w = h - r + eps # (batch, IW samples)
    log_normalization = torch.logsumexp(log_w, dim=1, keepdim=True)
    w = torch.exp(log_w - log_normalization)
    # print(nu1[0].mean().item(), nu1[1].mean().item(), nu2[0].mean().item(), nu2[1].mean().item())
    # print(h.mean().item(), h.median().item(), h.max().item(), h.min().item())
    # print(w)
    
    # print(torch.exp(-d - eps.unsqueeze(-1)).sum(1))
    KL_div = torch.tensor(0) # (w * d).sum(1)
    
    return w, KL_div, (w * d).sum(1)