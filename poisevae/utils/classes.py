import torch

class Categorical(torch.distributions.OneHotCategorical):
    def __init__(self, logits):
        super(Categorical, self).__init__(logits=logits.to(dtype=torch.float64))
    def log_prob(self, target):
        shape, dtype = target.shape, target.dtype
        idx = target.view(-1).to(torch.long)
        target = torch.nn.functional.one_hot(idx, num_classes=self.probs.shape[1])
        return super(Categorical, self).log_prob(target).view(*shape).to(dtype=dtype)

    
class NULLWriter:
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
    def add_scalars(self, *args, **kwargs):
        pass