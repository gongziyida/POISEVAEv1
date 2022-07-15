import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class pixelcnn_decoder(nn.Module):
    def __init__(self, pixelcnn):
        super(pixelcnn_decoder, self).__init__()
        self.pixelcnn = pixelcnn

    def forward(self, z, img,generate_mode):
        img_out = img.reshape(img.shape[0],1,28,28).to(z.device)
        if generate_mode is False:
            sample = self.pixelcnn(img_out, z )
        else:
            shape = [1,28,28]
            count = z.shape[0]
            sample = self.pixelcnn.sample(img_out,shape,count, z )
#             sample = self.pixelcnn(img_out, z )

        return (sample, )