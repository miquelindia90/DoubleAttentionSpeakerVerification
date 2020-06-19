import torch
from torch import nn
from torch.nn import functional as F
from poolings import *
from CNNs import *

class SpeakerClassifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
        
        parameters.feature_size = 80 

        self.pooling_method = parameters.pooling_method
        self.device = device
        
        if parameters.front_end=='VGG3L':
            self.vector_size = getVGG3LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            self.front_end = VGG3L(parameters.kernel_size)
        
        if parameters.front_end=='VGG4L':
            self.vector_size = getVGG4LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            self.front_end = VGG4L(parameters.kernel_size)
        
        self.pooling_method = parameters.pooling_method

        if parameters.pooling_method == 'attention':
            self.PoolingLayer = Attention(self.vector_size)
        
        elif parameters.pooling_method == 'MHA':
            self.PoolingLayer = MultiHeadAttention(self.vector_size, parameters.heads_number)

        elif parameters.pooling_method == 'DoubleMHA':
            self.PoolingLayer = DoubleMHA(self.vector_size, parameters.heads_number)
            self.vector_size = self.vector_size//parameters.heads_number
        
        self.fc1 = nn.Linear(self.vector_size, parameters.embedding_size)
        self.b1 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc2 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        self.preLayer = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b3 = nn.BatchNorm1d(parameters.embedding_size)
        
        if parameters.loss == 'Softmax':
            self.predictionLayer = nn.Linear(parameters.embedding_size, parameters.num_spkrs)
        elif parameters.loss == 'AMSoftmax':
            self.predictionLayer = nn.Linear(parameters.embedding_size, parameters.num_spkrs, bias=False)
        
        self.loss = parameters.loss

    def forward(self, x):

        encoder_output = self.front_end(x)

        embedding0, alignment = self.PoolingLayer(encoder_output)
        embedding1 = self.b1(F.relu(self.fc1(embedding0)))
        embedding2 = self.b2(F.relu(self.fc2(embedding1)))
                
        if self.loss == 'Softmax':
            embedding3 = self.b3(F.relu(self.preLayer(embedding2)))
            prediction = self.predictionLayer(embedding3)

        elif self.loss == 'AMSoftmax':
            embedding3 = self.preLayer(embedding2)
            for W in self.predictionLayer.parameters():
                W = F.normalize(W, dim=1)
            embedding3 = F.normalize(embedding3, dim=1)
            prediction = self.predictionLayer(embedding3)

        return encoder_output, embedding2, prediction

