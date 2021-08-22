import torch
import torch.nn as nn
import torch.nn.functional as F

class FSRCNN(nn.Module):
    def __init__(self, d, s, m):
        super(FSRCNN, self).__init__()

        self.feature= nn.conv2d(1,d, 5)
        self.shrink=nn.Conv2d(d,s,1)
        self.map = nn.Conv2d(s,s,3)
        self.expand = nn.Conv2d(s,d,1)
        self.deconv= nn.Conv2d(s,1,9)

    def forward(self, x):
        x = F.prelu(self.feature(x))
        x = F.prelu(self.shrink(x))
        for i in range(m):
            x = F.prelu(self.map(x))
        x = F.prelu(self.expand(x))
        x = F.prelu(self.deconv(x))
        return x