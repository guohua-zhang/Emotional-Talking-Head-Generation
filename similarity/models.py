import torch
from torch import nn
import numpy as np
from torchvision.models import resnet

class ImageEncoder(nn.Module):
    def __init__(self, depth, dim=128):
        super().__init__()
        self.model = getattr(resnet, f'resnet{depth}')(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.weight.shape[1], dim)
        self.act = nn.Tanh()

        # log
        self.register_buffer('mdist', torch.zeros(2))
        self.momentum = 0.999
        self.step = 0
        self.warmup = 500
    
    def forward(self, x):
        return self.act(self.model(x))
    
    def update_mdist(self, out):
        dist = distance(out)
        if self.step < self.warmup:
            momentum = np.linspace(0, self.momentum, self.warmup)[self.step]
            self.step += 1
        else:
            momentum = self.momentum
        self.mdist[0] = self.mdist[0] * momentum + (1 - momentum) * dist[0].detach()
        self.mdist[1] = self.mdist[1] * momentum + (1 - momentum) * dist[1].detach()


class AudioEncoder(nn.Module):
    def __init__(self, nconv=14, dim=128, nchan=64):
        super().__init__()
        self.nconv = nconv
        convs = []
        for iconv in range(nconv):
            if iconv == 0:
                chin = 1
            else:
                chin = nchan
            if (iconv+1) % 4 == 0:
                nchan *= 2
            if iconv % 2 == 0:
                stride = 2
            else:
                stride = 1
            convs.append(nn.Sequential(*[
                nn.Conv1d(chin,nchan,3,stride=stride,padding=1,bias=False),
                nn.BatchNorm1d(nchan),
                nn.ReLU(inplace=True)
            ]))
        self.convs = nn.Sequential(*convs)
        self.fc = nn.Linear(nchan, dim)
        self.act = nn.Tanh()
        
        # log
        self.register_buffer('mdist', torch.zeros(2))
        self.momentum = 0.999
        self.step = 0
        self.warmup = 500

    def forward(self, x):
        x = self.convs(x)
        x = self.fc(x.mean(-1))
        return self.act(x)
    
    def update_mdist(self, out):
        dist = distance(out)
        if self.step < self.warmup:
            momentum = np.linspace(0, self.momentum, self.warmup)[self.step]
            self.step += 1
        else:
            momentum = self.momentum
        self.mdist[0] = self.mdist[0] * momentum + (1 - momentum) * dist[0].detach()
        self.mdist[1] = self.mdist[1] * momentum + (1 - momentum) * dist[1].detach()


def distance(out):
    # same class
    dist0 = []
    for i in range(out.shape[1]):
        for j in range(out.shape[1]):
            if j == i:
                continue
            dist0.append(((out[:, i]-out[:, j])**2).sum(dim=1).sqrt())
    dist0 = torch.cat(dist0, dim=0).mean()

    # different class
    dist1 = []
    for i in range(out.shape[0]):
        for j in range(out.shape[0]):
            if j == i:
                continue
            dist1.append(((out[i]-out[j])**2).sum(dim=1).sqrt())
    dist1 = torch.cat(dist1, dim=0).mean()

    return dist0, dist1


class Loss(nn.Module):
    def __init__(self, mode='softmax', device=torch.device("cpu")):
        super(Loss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mode = mode
        self.device = device

    def forward(self, x):
        """
        x: (k/j (classes), m/i (samples), c (n_features))
        """
        # centers = torch.mean(x, 1, keepdim=False)
        # sims = -torch.cdist(x.reshape([-1, x.size(-1)]), centers, p=2)
        # sims = torch.max(sims, torch.tensor(-5.0).to(self.device))
        x1 = x[:, 1:, :].reshape(-1, x.shape[-1]).contiguous()
        x2 = x[:, 0, :].contiguous()
        sims = 1 - torch.cdist(x1, x2, p=2)

        if self.mode=="softmax":
            labels = torch.tensor(np.arange(x.size(0)).repeat(x.size(1)-1)).to(self.device)
            loss = self.ce(sims, labels)

        elif self.mode=="contrast":
            indices = list(np.arange(sims.size(0)))
            labels = list(np.arange(x.size(0)).repeat(x.size(1)-1))
            sims_clone = torch.clone(sims)
            sims_clone[indices, labels] = -1e32
            loss_self = sims[indices, labels]
            loss_others = torch.max(sims_clone, dim=1).values
            # loss = 1-torch.sigmoid(loss_self)+torch.sigmoid(loss_others)
            loss = loss_others-loss_self
            loss = torch.mean(loss)

        else:
            raise ValueError("Invalid mode.")

        return loss, sims