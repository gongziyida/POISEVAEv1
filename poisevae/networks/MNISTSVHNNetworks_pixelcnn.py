import torch
import torch.nn as nn
import torch.nn.functional as F

class pixelcnn_decoder(nn.Module):
    def __init__(self, mlp, pixelcnn, img_dims):
        super(pixelcnn_decoder, self).__init__()
        self.mlp = mlp
        self.pixelcnn = pixelcnn
        self.img_dims = img_dims

    def forward(self, z):
        sample = torch.zeros(*z.shape[:1], *self.img_dims).to(z.device) # (batch * gibbs, img dims)
        sample[..., 0, 0] = self.mlp(z) # (batch * gibbs, channel)
        sample = self.pixelcnn(sample).flatten(-2 if self.img_dims[0] == 1 else -3,-1)
        sample = sample.transpose(-1, -2) # (batch * gibbs, img dims flattened, 255)
        # sample = F.softmax(sample, dim=-1).data
        #Generating images pixel by pixel
        #for i in range(self.img_dims[1]):
            #for j in range(self.img_dims[2]):
                #if i==0 and j==0:
                    #continue
                #out = self.pixelcnn(sample) # (batch * gibbs, 255, img dims)
                #probs = F.softmax(out[:,:,i,j], dim=1).data
                #sample[...,i,j] = torch.multinomial(probs, 1).float() / 255.0
        #sample = sample.flatten(-3, -1)
          
        return (sample, )
