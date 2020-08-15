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


class AMSoftmaxV2(nn.Module):

    '''
    Alternative Implementation Extracted From
    https://github.com/clovaai/voxceleb_trainer/blob/master/loss/cosface.py
    '''

    def __init__(self, in_feats, n_classes, m=0.3, s=15):
        super(AMSoftmaxV2, self).__init__()
        self.m = m
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)


    def forward(self, x, label=None):
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
        costh_m_s = self.s * costh_m
        return costh_m_s
 
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
    
