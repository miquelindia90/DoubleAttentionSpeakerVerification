import torch
from torch import nn
from torch.nn import functional as F
from poolings import *
from CNNs import *
from ResNets import *

class SpeakerClassifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
        
        parameters.feature_size = 80 

        self.pooling_method = parameters.pooling_method
        self.device = device
        
        if parameters.front_end=='VGG3L':
            self.vector_size = getVGG3LOutputDimension(parameters.feature_size, inputChannel=1, outputChannel=parameters.kernel_size)
            self.front_end = VGG3L(parameters.kernel_size)
        
        if parameters.front_end=='VGG4L':
            self.vector_size = getVGG4LOutputDimension(parameters.feature_size, inputChannel=1, outputChannel=parameters.kernel_size)
            self.front_end = VGG4L(parameters.kernel_size)
        
        self.pooling_method = parameters.pooling_method
        self.loss = parameters.loss

        elif parameters.pooling_method == 'attention':
            self.PoolingLayer = Attention(self.vector_size)
            self.fc1 = nn.Linear(self.vector_size, parameters.embedding_size)
            self.b1 = nn.BatchNorm1d(parameters.embedding_size)
            self.preLayer = nn.Linear(parameters.embedding_size, parameters.embedding_size)
            self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        
        elif parameters.pooling_method == 'multihead_attention':
            self.PoolingLayer = MultiHeadedAttention(self.vector_size, parameters.heads_number)
            self.fc1 = nn.Linear(int(self.vector_size), parameters.embedding_size)
            self.b1 = nn.BatchNorm1d(parameters.embedding_size)
            self.preLayer = nn.Linear(parameters.embedding_size, parameters.embedding_size)
            self.b2 = nn.BatchNorm1d(parameters.embedding_size)

        elif parameters.pooling_method == 'MHAMixedAttentionV5':
            self.PoolingLayer = MHAMixedAttentionV5(self.vector_size, parameters.heads_number, mask_prob = parameters.mask_prob)
            self.fc1 = nn.Linear(self.vector_size//parameters.heads_number, parameters.embedding_size)
            self.b1 = nn.BatchNorm1d(parameters.embedding_size)
            self.preLayer = nn.Linear(parameters.embedding_size, parameters.embedding_size)
            self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        
        if parameters.loss == 'Softmax':
            self.predictionLayer = nn.Linear(parameters.embedding_size, parameters.num_spkrs)
        elif parameters.loss == 'AMSoftmax':
            self.predictionLayer = nn.Linear(parameters.embedding_size, parameters.num_spkrs, bias=False)

    def forward(self, x, label=None):

        encoder_output = self.front_end(x)
        layer, alignment = self.PoolingLayer(encoder_output)
        embedding1 = F.relu(self.fc1(layer))
        embedding1 = self.be1(embedding1)
                
        if self.loss == 'Softmax':
            embedding2 = F.relu(self.preLayer(embedding))
            embedding2 = self.b2(embedding2)
            prediction = self.prediction_Layer(embedding2)

        elif self.loss == 'AMSoftmax':
            embedding2 = self.preLayer(embedding)
            for W in self.predictionLayer.parameters():
                W = F.normalize(W, dim=1)
            embedding2 = F.normalize(embedding2, dim=1)
            prediction = self.prediction_Layer(embedding2)

        return encoder_output, self.__L2(embedding), prediction

