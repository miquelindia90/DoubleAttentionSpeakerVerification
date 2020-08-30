import pickle
import numpy as np
from random import randint, randrange
from torch.utils import data
import soundfile as sf

def featureReader(featurePath, VAD=None):

    with open(featurePath,'rb') as pickleFile:
        features = pickle.load(pickleFile)
        if VAD is not None:
            filtered_features = VAD.filter(features)
        else:
            filtered_features = features

    if filtered_features.shape[1]>0.:
        return np.transpose(filtered_features)
    else:
        return np.transpose(features)

def normalizeFeatures(features, normalization='cmn'):

    mean = np.mean(features, axis=0)
    features -= mean 
    if normalization=='cmn':
       return features
    if normalization=='cmvn':
        std = np.std(features, axis=0)
        std = np.where(std>0.01,std,1.0)
        return features/std

class Dataset(data.Dataset):

    def __init__(self, utterances, parameters):
        'Initialization'
        self.utterances = utterances
        self.parameters = parameters
        self.num_samples = len(utterances)

    def __normalize(self, features):
        mean = np.mean(features, axis=0)
        features -= mean 
        if self.parameters.normalization=='cmn':
            return features
        if self.parameters.normalization=='cmvn':
            std = np.std(features, axis=0)
            std = np.where(std>0.01,std,1.0)
            return features/std

    def __sampleSpectogramWindow(self, features):
        file_size = features.shape[0]
        windowSizeInFrames = self.parameters.window_size*100
        index = randint(0, max(0,file_size-windowSizeInFrames-1))
        a = np.array(range(min(file_size, int(windowSizeInFrames))))+index
        return features[a,:]

    def __getFeatureVector(self, utteranceName):

        with open(utteranceName + '.pickle','rb') as pickleFile:
            features = pickle.load(pickleFile)
        windowedFeatures = self.__sampleSpectogramWindow(self.__normalize(np.transpose(features)))
        return windowedFeatures            
     
    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        'Generates one sample of data'
        utteranceTuple = self.utterances[index].strip().split()
        utteranceName = self.parameters.train_data_dir + '/' + utteranceTuple[0]
        utteranceLabel = int(utteranceTuple[1])
        
        return self.__getFeatureVector(utteranceName), np.array(utteranceLabel)

