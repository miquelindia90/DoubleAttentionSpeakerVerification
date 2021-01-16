import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import math
import copy

def new_parameter(*size):
    out = torch.nn.Parameter(torch.FloatTensor(*size))
    torch.nn.init.xavier_normal_(out)
    return out

class Attention(nn.Module):

    def __init__(self, embedding_size):

        super(Attention, self).__init__()
        self.embedding_size = embedding_size
        self.att=new_parameter(self.embedding_size,1)

    def forward(self,ht):
        attention_score = torch.matmul(ht, self.att).squeeze()
        attention_score = F.softmax(attention_score, dim=-1).view(ht.size(0), ht.size(1),1)
        ct = torch.sum(ht * attention_score,dim=1)

        return ct, attention_score

class HeadAttention(nn.Module):

    def __init__(self, encoder_size, heads_number, mask_prob = 0.25, attentionSmoothing=False):

        super(HeadAttention, self).__init__()
        self.embedding_size = encoder_size//heads_number
        self.att=new_parameter(self.embedding_size,1)
        self.mask_prob = int(1/mask_prob)
        self.attentionSmoothing = attentionSmoothing

    def __maskAttention(self, attention_score, mask_value = -float('inf')):
        
        mask = torch.cuda.FloatTensor(attention_score.size()).random_(self.mask_prob)>0
        attention_score[~mask] = mask_value
        return attention_score

    def __narrowAttention(self, new_ht):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score)
        attention_score = F.softmax(attention_score, dim=-1).view(new_ht.size(0), new_ht.size(1),1)
        return attention_score 

    def __wideAttention(self):

        attention_score = torch.matmul(new_ht, self.att).squeeze()
        if self.training:
            attention_score = self.__maskAttention(attention_score, mask_value = -1)
        attention_score /= torch.sum(attention_score, dim=1).unsqueeze(1)
        return attention_score.view(new_ht.size(0), new_ht.size(1),1)

    def forward(self,ht):

        if self.attentionSmoothing:
            attention_score = self.__wideAttention(ht)
        else:
            attention_score = self.__narrowAttention(ht)

        weighted_ht = ht * attention_score
        ct = torch.sum(weighted_ht,dim=1)

        return ct, attention_score

def innerKeyValueAttention(query, key, value):

    d_k = query.size(-1)
    scores = torch.diagonal(torch.matmul(key, query) / math.sqrt(d_k), dim1=-2, dim2=-1).view(value.size(0),value.size(1), value.size(2))
    p_attn = F.softmax(scores, dim = -2)
    weighted_vector = value * p_attn.unsqueeze(-1)
    ct = torch.sum(weighted_vector, dim=1)
    return ct, p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, encoder_size, heads_number):
        super(MultiHeadAttention, self).__init__()
        self.encoder_size = encoder_size
        assert self.encoder_size % heads_number == 0 # d_model
        self.head_size = self.encoder_size // heads_number 
        self.heads_number = heads_number
        self.query = new_parameter(self.head_size, self.heads_number)
        self.aligmment = None
        
    def getAlignments(self,ht): 
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return self.alignment 
    
    def getHeadsContextVectors(self,ht):    
        batch_size = ht.size(0)
        key = ht.view(batch_size*ht.size(1), self.heads_number, self.head_size)
        value = ht.view(batch_size,-1,self.heads_number, self.head_size)
        headsContextVectors, self.alignment = innerKeyValueAttention(self.query, key, value)
        return headsContextVectors

    def forward(self, ht):
        headsContextVectors = self.getHeadsContextVectors(ht)
        return headsContextVectors.view(headsContextVectors.size(0),-1), copy.copy(self.alignment)


class DoubleMHA(nn.Module):
    def __init__(self, encoder_size, heads_number, mask_prob=0.2):
        super(DoubleMHA, self).__init__()
        self.heads_number = heads_number
        self.utteranceAttention = MultiHeadAttention(encoder_size, heads_number)
        self.heads_size = encoder_size // heads_number
        self.headsAttention = HeadAttention(encoder_size, heads_number, mask_prob=mask_prob, attentionSmoothing=False)

    def getAlignments(self, x):

        utteranceRepresentation, alignment = self.utteranceAttention(x)
        headAlignments = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[1]
        return alignment, headAlignments

    def forward(self, x):
        utteranceRepresentation, alignment = self.utteranceAttention(x)
        compressedRepresentation = self.headsAttention(utteranceRepresentation.view(utteranceRepresentation.size(0), self.heads_number, self.heads_size))[0]    
        return compressedRepresentation, alignment

