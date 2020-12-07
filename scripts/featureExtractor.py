import numpy as np
import soundfile as sf
import sys
import pickle
import librosa
import argparse

def mfsc(y, sfr, window_size=0.025, window_stride=0.010, window='hamming', n_mels=80, preemCoef=0.97):
    win_length = int(sfr * window_size)
    hop_length = int(sfr * window_stride)
    n_fft = 512
    lowfreq = 0
    highfreq = sfr/2

    # melspectrogram
    y *= 32768
    y[1:] = y[1:] - preemCoef*y[:-1]
    y[0] *= (1 - preemCoef)
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=False)
    D = np.abs(S)
    param = librosa.feature.melspectrogram(S=D, sr=sfr, n_mels=n_mels, fmin=lowfreq, fmax=highfreq, norm=None)
    mf = np.log(np.maximum(1, param))
    return mf

def normalize(features):
    return features-np.mean(features, axis=0)


def extractFeatures(audioPath):

    y, sfreq = sf.read(audioPath)
    features = mfsc(y, sfreq)    
    return normalize(np.transpose(features))

def main(params):

    with open(params.audioFilesList,'r') as  filesFile:
        for featureFile in filesFile:
            print(featureFile[:-1])
            y, sfreq = sf.read('{}'.format(featureFile[:-1])) 
            mf = mfsc(y, sfreq)
            with open('{}.pickle'.format(featureFile[:-5]), 'wb') as handle:
                pickle.dump(mf,handle)

if __name__=='__main__':


    parser = argparse.ArgumentParser(description='Extract Features. Looks for .wav files and extract Features')
    parser.add_argument('--audioFilesList', '-i', type=str, required=True, default='', help='Wav Files List.')
    params=parser.parse_args()
    main(params) 



