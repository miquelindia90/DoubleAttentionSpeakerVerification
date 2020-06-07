import sys
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def getVGG3LOutputDimension(inputDimension, outputChannel=128):

    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    return int(outputDimension) * outputChannel

def getVGG4LOutputDimension(inputDimension, outputChannel=128):

    outputDimension = np.ceil(np.array(inputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    outputDimension = np.ceil(np.array(outputDimension, dtype=np.float32)/2)
    return int(outputDimension) * outputChannel

class VGG3L(torch.nn.Module):

    def __init__(self, kernel_size):
        super(VGG3L, self).__init__()

        self.conv11 = torch.nn.Conv2d(1, int(kernel_size/4), 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/4), 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv31 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size), 3, stride=1, padding=1)
        self.conv32 = torch.nn.Conv2d(int(kernel_size), int(kernel_size), 3, stride=1, padding=1)
        
    def forward(self, paddedInputTensor):

        paddedInputTensor =  paddedInputTensor.view( paddedInputTensor.size(0),  paddedInputTensor.size(1), 1, paddedInputTensor.size(2)).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(encodedTensorLayer1, 2, stride=2, ceil_mode=True)

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(encodedTensorLayer2, 2, stride=2, ceil_mode=True)

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3))
        encodedTensorLayer3 = F.max_pool2d(encodedTensorLayer3, 2, stride=2, ceil_mode=True)
        outputTensor = encodedTensorLayer3.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(outputTensor.size(0), outputTensor.size(1), outputTensor.size(2) * outputTensor.size(3))

        return outputTensor

class VGG4L(torch.nn.Module):

    def __init__(self, kernel_size):
        super(VGG4L, self).__init__()

        self.conv11 = torch.nn.Conv2d(1, int(kernel_size/8), 3, stride=1, padding=1)
        self.conv12 = torch.nn.Conv2d(int(kernel_size/8), int(kernel_size/8), 3, stride=1, padding=1)
        self.conv21 = torch.nn.Conv2d(int(kernel_size/8), int(kernel_size/4), 3, stride=1, padding=1)
        self.conv22 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/4), 3, stride=1, padding=1)
        self.conv31 = torch.nn.Conv2d(int(kernel_size/4), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv32 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size/2), 3, stride=1, padding=1)
        self.conv41 = torch.nn.Conv2d(int(kernel_size/2), int(kernel_size), 3, stride=1, padding=1)
        self.conv42 = torch.nn.Conv2d(int(kernel_size), int(kernel_size), 3, stride=1, padding=1)
        
    def forward(self, paddedInputTensor):

        paddedInputTensor =  paddedInputTensor.view( paddedInputTensor.size(0),  paddedInputTensor.size(1), 1, paddedInputTensor.size(2)).transpose(1, 2)

        encodedTensorLayer1 = F.relu(self.conv11(paddedInputTensor))
        encodedTensorLayer1 = F.relu(self.conv12(encodedTensorLayer1))
        encodedTensorLayer1 = F.max_pool2d(encodedTensorLayer1, 2, stride=2, ceil_mode=True)

        encodedTensorLayer2 = F.relu(self.conv21(encodedTensorLayer1))
        encodedTensorLayer2 = F.relu(self.conv22(encodedTensorLayer2))
        encodedTensorLayer2 = F.max_pool2d(encodedTensorLayer2, 2, stride=2, ceil_mode=True)

        encodedTensorLayer3 = F.relu(self.conv31(encodedTensorLayer2))
        encodedTensorLayer3 = F.relu(self.conv32(encodedTensorLayer3)) 
        encodedTensorLayer3 = F.max_pool2d(encodedTensorLayer3, 2, stride=2, ceil_mode=True)

        encodedTensorLayer4 = F.relu(self.conv41(encodedTensorLayer3))
        encodedTensorLayer4 = F.relu(self.conv42(encodedTensorLayer4))
        encodedTensorLayer4 = F.max_pool2d(encodedTensorLayer4, 2, stride=2, ceil_mode=True)

        outputTensor = encodedTensorLayer4.transpose(1, 2)
        outputTensor = outputTensor.contiguous().view(outputTensor.size(0), outputTensor.size(1), outputTensor.size(2) * outputTensor.size(3))

        return outputTensor

