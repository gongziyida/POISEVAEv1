import torch

class Categorical(torch.distributions.OneHotCategorical):
    def __init__(self, probs=None, logits=None, **kwargs):
        super(Categorical, self).__init__(probs=probs, logits=logits, **kwargs)
    def log_prob(self, target):
        shape = target.shape
        idx = target.view(-1).to(torch.long)
        target = torch.nn.functional.one_hot(idx, num_classes=self.probs.shape[1])
        return super(Categorical, self).log_prob(target).view(*shape)