import pickle
import torch
import argparse
from model import *
from featureExtractor import *

def prepareInput(features, device):
    
    inputs = torch.FloatTensor(features)
    inputs = inputs.to(device)
    inputs = inputs.unsqueeze(0)
    return inputs


def getAudioEmbedding(audioPath, net, device):

    features = extractFeatures(audioPath)
    with torch.no_grad():
        networkInputs = prepareInput(features, device)
        return net.getEmbedding(networkInputs)
                

def main(opt,params):

    print('Loading Model')
    device = torch.device(params.device)
    net_dict = torch.load(params.modelCheckpoint, map_location=device)
    opt = net_dict['settings'] 

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        
    net = SpeakerClassifier(opt, device)
    net.load_state_dict(net_dict['model'])
    net.to(device)
    net.eval()
    
    embedding = getAudioEmbedding(params.audioPath, net, device)
    print(embedding)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='score a trained model')
    parser.add_argument('--audioPath', type=str, required=True)
    parser.add_argument('--modelConfig', type=str, required=True)
    parser.add_argument('--modelCheckpoint', type=str, required=True) 
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda']) 

    params = parser.parse_args()

    with open(params.modelConfig, 'rb') as handle:
        opt = pickle.load(handle)
        
    main(opt,params)

