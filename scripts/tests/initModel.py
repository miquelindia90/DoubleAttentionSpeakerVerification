import os
import sys
import argparse
import numpy as np
sys.path.append('./scripts/')
from model import *

def main(opt):

    print('Defining Device')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None

    print('Defining Model')

    model = SpeakerClassifier(opt, device)

    print('Model Correctly Setup')
    
if __name__=="__main__":

    parser = argparse.ArgumentParser(description='VGG Attention Based Speaker Embedding Extractor')
   
    parser.add_argument('--front_end', type=str, default='VGG3L', choices = ['VGG3L','VGG4L'], help='Kind of Front-end Used')
    parser.add_argument('--kernel_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=400)
    parser.add_argument('--heads_number', type=int, default=16)
    parser.add_argument('--pooling_method', type=str, default='DoubleMHA', choices=['attention', 'MHA', 'DoubleMHA'] ,help='type of pooling methods')
    parser.add_argument('--loss', type=str, choices=['Softmax', 'AMSoftmax'], default='Softmax', help='type of loss function')
    parser.add_argument('--scalingFactor', type=float, default=30.0, help='')
    parser.add_argument('--marginFactor', type=float, default=0.4, help='')

    params=parser.parse_args()
    params.num_spkrs = 7205

    main(params)
