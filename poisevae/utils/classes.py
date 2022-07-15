import torch

class Categorical(torch.distributions.OneHotCategorical):
    def __init__(self, logits):
        super(Categorical, self).__init__(logits=logits.to(dtype=torch.float64))
    def log_prob(self, target):
        shape, dtype = target.shape, target.dtype
        idx = target.view(-1).to(torch.long)
        target = torch.nn.functional.one_hot(idx, num_classes=self.probs.shape[1])
        ret = super(CategoricalImage, self).log_prob(target)
        return ret.view(shape[0], -1, *shape[2:]).to(dtype=dtype)

class _CategoricalImage(torch.distributions.OneHotCategorical):
    def __init__(self, logits):
        logits = logits.flatten(-3, -1).transpose(-1, -2)
        super(CategoricalImage, self).__init__(logits=logits.to(dtype=torch.float64))
    def log_prob(self, target):
        shape, dtype = target.shape, target.dtype
        if target.max() <= 1:
            target = (target * 255).to(torch.long)
        target = torch.nn.functional.one_hot(target, num_classes=256)
        print(target.shape, self.probs.shape)
        ret = super(CategoricalImage, self).log_prob(target)
        return ret.view(shape[0], -1, *shape[2:]).to(dtype=dtype)

class CategoricalImage(torch.distributions.Categorical):
    def __init__(self, logits):
        logits = logits.flatten(-3, -1).transpose(-1, -2)
        super(CategoricalImage, self).__init__(logits=logits.to(dtype=torch.float64))
    def log_prob(self, target):
        shape, dtype = target.shape, target.dtype
        if target.max() <= 1:
            target = (target * 255).to(torch.long)
        print(target.shape, self.probs.shape)
        ret = super(CategoricalImage, self).log_prob(target)
        return ret.view(shape[0], -1, *shape[2:]).to(dtype=dtype)

    
class NULLWriter:
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        pass
    def add_scalars(self, *args, **kwargs):
        pass
