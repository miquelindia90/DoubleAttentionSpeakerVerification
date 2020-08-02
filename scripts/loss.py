import torch
from torch import nn

class AMSoftmax(nn.Module):
    '''
    additive margin softmax as proposed in:
    "Additive Margin Softmax for Face Verification"
    by F. Wang et al.
    https://github.com/cvqluu/Additive-Margin-Softmax-Loss-Pytorch/blob/master/AdMSLoss.py
    '''
    def __init__(self,s=30.0, m=0.4):
        super(AMSoftmax, self).__init__()
        self.s = s
        self.m = m

    def forward(self, wf, labels):
        '''
        input shape (batch,classes)
        '''
        assert len(wf) == len(labels)
        assert torch.min(labels) >= 0
        numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y+1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)


class FocalSoftmax(nn.Module):
    ''' 
    Focal softmax as proposed in:
    "Focal Loss for Dense Object Detection"
    by T-Y. Lin et al.
    https://github.com/foamliu/InsightFace-v2/blob/master/focal_loss.py
    '''
    def __init__(self, gamma=2):
        super(FocalSoftmax, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
