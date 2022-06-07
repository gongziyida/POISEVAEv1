import torch
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class Wrapper(nn.Module):
    def __init__(self, decoder, pixel_cnn):
        super().__init__()
        self.decoder = decoder
        self.pixel_cnn = pixel_cnn
        
    def forward(self, z, b=1):
        x = self.decoder(z)[0]
        print('x1', x.shape)
        
        x = [x[i*b:(i+1)*b] for i in range(x.shape[0]//b)]
        assert sum(map(lambda x: x.shape[0], x)) == z.shape[0]
        for i in range(len(x)):
            x[i] = self.pixel_cnn(x[i])
        x = torch.cat(x, 0)
        print('x2', x.shape)
        return x, torch.tensor(0.75).to(device)
    