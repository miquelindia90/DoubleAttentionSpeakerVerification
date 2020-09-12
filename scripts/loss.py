import torch
import logging
from torch import nn

class AMSoftmax(nn.Module):

    '''
    Additve Margin Softmax as proposed in:
    https://arxiv.org/pdf/1801.05599.pdf
    Implementation Extracted From
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/cosface.py
    '''

    def __init__(self, in_feats, n_classes, m=0.3, s=15, annealing=False):
        super(AMSoftmax, self).__init__()
        self.in_feats = in_feats
        self.m = m
        self.s = s
        self.annealing = annealing
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
        self.annealing=annealing

    def getAnnealedFactor(self,step):
        alpha = self.__getAlpha(step) if self.annealing else 0.
        return 1/(1+alpha)

    def __getAlpha(self,step):
        return max(0, 1000./(pow(1.+0.0001*float(step),2.)))        

    def __getCombinedCosth(self, costh, costh_m, step):

        alpha = self.__getAlpha(step) if self.annealing else 0.
        costh_combined = costh_m + alpha*costh
        return costh_combined/(1+alpha)

    def forward(self, x, label=None, step=0):
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm)
        label_view = label.view(-1, 1)
        if label_view.is_cuda: label_view = label_view.cpu()
        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda: delt_costh = delt_costh.cuda()
        costh_m = costh - delt_costh
        costh_combined = self.__getCombinedCosth(costh, costh_m, step)
        costh_m_s = self.s * costh_combined
        return costh, costh_m_s 
 
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
    
