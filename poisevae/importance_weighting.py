import torch
import numpy as np

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_LOG_2_PI = 0.79817986835

def calc_eps(val):
    with torch.no_grad():
        eps = -val.max(dim=-1).values.unsqueeze(-1)
    return eps

def sample_proposal(m1, m2, mu, var, batch_size, n_IW_samples, device=_device):
    with torch.no_grad():
        mn1 = torch.distributions.MultivariateNormal(mu[0], torch.diag_embed(var[0]))
        mn1_samples = mn1.sample([n_IW_samples]).to(device) # (IW Samples, Batch)
        mn2 = torch.distributions.MultivariateNormal(mu[1], torch.diag_embed(var[1]))
        mn2_samples = mn2.sample([n_IW_samples]).to(device) # (IW Samples, Batch)
    # Swap axis to (Batch, IW Samples)
    return [torch.swapaxes(mn1_samples, 0, 1), torch.swapaxes(mn2_samples, 0, 1)] 


def calc_weight(h, r, d, log_r_normalization):
    """ This calculation is the same as before. I just moved it here for reusing the code
    """
    log_r_normalization = log_r_normalization.unsqueeze(1) # (batch, 1)
    
    log_w_prior = h - r + log_r_normalization # (batch, IW samples)
    
    h = h + d
    assert torch.isnan(h).sum() == 0, "%s\n%s\n%s\n%s\n%s" % (d, nu1[0], nu1[1], nu2[0], nu2[1])
    
    log_w_post = h - r + log_r_normalization # (batch, IW samples)
    
    log_normalization_prior = torch.logsumexp(log_w_prior, dim=1, keepdim=True)
    log_normalization_post = torch.logsumexp(log_w_post, dim=1, keepdim=True)
    
    log_w = log_w_post - log_normalization_post # Normalized 
    w = torch.exp(log_w)
    
    return w, log_w, log_normalization_post, log_normalization_prior


def check_nu(nu1, nu2):
    """ These checking code is affecting the readability so I moved it here
    """
    assert nu2[1].mean().item() > -1e20 # Too small is pathological
    # print(nu1[0].mean().item(), nu1[1].mean().item(), nu2[0].mean().item(), nu2[1].mean().item())
    # print(h.mean().item(), h.median().item(), h.max().item(), h.min().item())
    # print(w)
    # print('_____')
    

def IWq(G, z, nu1, nu2, mu_proposal, var_proposal):
    """ Calc. the weights and expectations with importance weighting sampling.
        Assume `z`, `nu1` and `nu2` are list of tensors for different modalities.
        Assume `nu1` and `nu2` have shape (batch, latent dimension),
                `z` has shape (batch, IW samples, latent dimension),
            and `mu/var_proposal` have shape (batch, latent dimension)
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
    
    check_nu(nu1, nu2) # Just for testing
    
    # log proposal
    r = -((z[0] - mu_proposal[0].unsqueeze(1))**2 / var_proposal[0].unsqueeze(1)).sum(-1) - \
         ((z[1] - mu_proposal[1].unsqueeze(1))**2 / var_proposal[1].unsqueeze(1)).sum(-1)
    r /= 2
    
    # log prior
    h = -z_sq[0].sum(-1) - z_sq[1].sum(-1)
    assert torch.isnan(h).sum() == 0
    
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

    log_r_normalization = -0.5 * (torch.log(var_proposal[0]).sum(1) + \
                                  torch.log(var_proposal[1]).sum(1) + \
                                  (m[0] + m[1]) * _LOG_2_PI)

    ########### PART I: TRAINING POISEVAE ###########
    ret = calc_weight(h, r.detach(), d, log_r_normalization.detach()) # Uncommon calc_weight below
    # ret = calc_weight(h, r, d, log_r_normalization) # Comment calc_weight below
    w, log_w, log_normalization_post, log_normalization_prior = ret
    
    # KL_poise = torch.zeros(2) # arbitrary
    KL_poise = (w * (d - log_normalization_post + log_normalization_prior)).sum(1) # (batch,)
    
    ########### PART II: TRAINING PROPOSAL ###########
#     ########### METHOD I: MAXIMIZE ENTROPY ###########
#     # May lead to proposal var too small
#     w, log_w, _, _ = calc_weight(h.detach(), r, d.detach(), log_r_normalization)
#     neg_entropy = (w * log_w).sum(1) # (batch,)
    
#     return w, KL_poise, neg_entropy

    ########### METHOD II: MINIMIZE KL(r||q) ###########
    _, log_w, _, _ = calc_weight(h.detach(), r, d.detach(), log_r_normalization)
    
    # KL(r||h) =  E_r [r - h - log_normalization_r + log_normalization_post]
    #          = -E_r [h - r - log_normalization_post] - log_normalization_r
    #          = -E_r [h - r - log_normalization_post] - log_normalization_r
    #          = -E_r [log_w_post] - log_normalization_r
    prob_r = torch.exp(r - log_r_normalization.unsqueeze(1))
    KL_proposal = (-prob_r * log_w).sum(1) - \
                  log_r_normalization
    
    return w, KL_poise, KL_proposal

#     ########### METHOD III: MINIMIZE MSE ###########
#     _, log_w, _, _ = calc_weight(h.detach(), r, d.detach(), log_r_normalization)
#     mse_proposal_poise = (log_w**2).sum(1)
    
#     return w, KL_poise, mse_proposal_poise