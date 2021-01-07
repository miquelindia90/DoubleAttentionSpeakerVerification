# DoubleAttentionSpeakerVerification

Pytorch implemenation of the model proposed in the paper:

[Double Multi-Head Attention for Speaker Verification](https://arxiv.org/abs/2007.13199)

## Installation

This repository has been created using python3.6. You can find the python3
dependencies on requirements.txt. Hence you can install it by:

```bash
pip install -r requirements.txt
```

Note that soundfile library also needs the C libsndfile library. You can find
more details about its installation in [Soundfile](https://pysoundfile.readthedocs.io/en/latest/).

## Usage

This repository code shoud allow you to train a speker embedding extractor according to the implementation described in the previous cited paper. This speaker embedding extractor is based on a speaker classifier which identifies the speaker identity given a variable length utterance audio. The network used for this work uses mel-spectogram features as input. Hence we have added instructions in this code to reproduce the feature extraction, the network training and the speaker embedding step. Feel free to ask any doubt via git-hub issues, [twitter](https://twitter.com/mikiindia) or mail(miquel.angel.india@upc.edu).

### Feature Extraction

### Network Training

### Speaker Embedding Extraction

