import torch

class KLGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T_p, Tp_p, T_q, Tp_q, G, nu, nup):
        """
        Parameters
        ----------
        T_p, Tp_p, T_q, Tp_q, nu, nup: torch.Tensor
            Shape: (batch, n_iter, latent_dim * 2)
        G: torch.Tensor
            Shape: (batch, latent_dim * 2, latent_dim * 2)
        """
        ctx.save_for_backward(T_p, Tp_p, T_q, Tp_q, nu, nup)
        return torch.tensor([0.0], requires_grad=True, device=G.device, dtype=G.dtype)

    @staticmethod
    def backward(ctx, _):
        T_p, Tp_p, T_q, Tp_q, nu, nup = ctx.saved_tensors 
        T_q_diff, Tp_q_diff = T_q - T_q.mean(1, keepdim=True), Tp_q - Tp_q.mean(1, keepdim=True)
        cov = torch.bmm(T_q_diff.transpose(1, 2), T_q_diff) / (T_q.shape[1] - 1)
        covp = torch.bmm(Tp_q_diff.transpose(1, 2), Tp_q_diff) / (Tp_q.shape[1] - 1)
        TTp_p = (T_p.unsqueeze(-1) * Tp_p.unsqueeze(-2)).mean(1)
        TTp_q = (T_q.unsqueeze(-1) * Tp_q.unsqueeze(-2)).mean(1)
        dG = TTp_p - TTp_q
        dnu = torch.bmm(nu.unsqueeze(1), cov).squeeze(1)
        dnup = torch.bmm(nup.unsqueeze(1), covp).squeeze(1)
        # print('dnu', dnu.detach().sum().item(), nu.detach().sum().item(), cov.detach().sum().item())
        return None, None, None, None, dG, dnu, dnup
        # return None, None, None, None, None, None, None
        
        
class RecGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, T_q, Tp_q, G, nu, nup, loglike):
        """
        Parameters
        ----------
        T_q, Tp_q, nu, nup: torch.Tensor
            Shape: (batch, n_iter, latent_dim * 2)
        G: torch.Tensor
            Shape: (batch, latent_dim * 2, latent_dim * 2)
        loglike: torch.Tensor
            Shape: (batch, n_iter); correspond to the Gibbs sample used for reconstruction
        """
        ctx.save_for_backward(T_q, Tp_q, nu, nup, loglike.unsqueeze(-1))
        return torch.tensor([0.0], requires_grad=True, device=G.device, dtype=G.dtype)
    @staticmethod
    def backward(ctx, _):
        # Notice that T's shape (batch, n_iter, latent_dim)
        T_q, Tp_q, nu, nup, loglike = ctx.saved_tensors 
        dG = (T_q.unsqueeze(-1) * Tp_q.unsqueeze(-2) * loglike.unsqueeze(-1)).mean(1)
        dnu = (T_q * loglike).mean(1)
        dnup = (Tp_q * loglike).mean(1)
        
        return None, None, dG, dnu, dnup, None
        # return None, None, None, None, None, None