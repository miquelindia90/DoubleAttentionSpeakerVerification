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

This repository code shoud allow you to train a speker embedding extractor according to the implementation described in the paper. This speaker embedding extractor is based on a speaker classifier which identifies the speaker identity given a variable length utterance audio. The network used for this work uses mel-spectogram features as input. Hence, we have added here tge instructions to reproduce the feature extraction, the network training and the speaker embedding extraction step. Feel free to ask any doubt via git-hub issues, [twitter](https://twitter.com/mikiindia) or mail(miquel.angel.india@upc.edu).

### Feature Extraction

You can find in `scripts/featureExtractor.py` several functions which extract and normalize the Log Mel Spectogram features. If you want to run the whole feature extraction over a set of audios you can run the following command:

```bash
python scripts/featureExtractor -i files.lst
```

where `files.lst` contains the audio paths aimed to parameterize. Each row of the file must contain an audio path without the file format extension (we assume you will be using .wav). Example:

<pre>
audiosPath/audio1
audiosPath/audio2
...
audiosPath/audioN</pre>

This script will extract a feature for each audio file and it will store it in a pickle in the same audio path.

### Network Training

Once you have extracted the features from all the audios wanted to be used, It is needed to prepare some path files for the training step. The proposed models are trained as speaker classifiers, hence a classification-based loss and an accuracy metric will be used to monitorize the training progress. However in the validaition step, an EER estimation is used to validate the network progresss. The motivation behind this is that best accuracy models do not always have the best inter/intra speaker variability. Therefore we prefer to  use directly a task based metric to validate the model instead of using a classification one. Two different kind of path files will then be needed for the training/validation procedures:

Train Labels File (`train_labels_path`): This file must have three columns separated by a blank space. The first column must contain the audio utterance paths, the second column must contain the speaker labels and the third one must be filled with None. It is assumed that the labels correspond to the output labels ids of the network. Hence if you are working with a N speakers database, the speaker labels values should be in the 0 to N-1 range.

File Example:

<pre>
audiosPath/speaker1/audio1 0 None
audiosPath/speaker1/audio2 0 None
...
audiosPath/speakerN/audio4 N-1</pre>

We have also added a `--train_data_dir` path argument. The dataloader will then look for the features on the `--train_data_dir` + `audiosPath/speakeri/audioj` paths.

Valid Labels File:

For the validation step, it will be needed a tuple of client/impostors trial files. Client trials (`valid_clients`) file must contain pair of audio utterance from same spakers and the impostors trials file (`valid_impostors`) must contain utterance pairs but from different speakers. Each pair path must be separated with a blank space:

File Example (Clients):

<pre>
audiosPath/speaker1/audio1 audiosPath/speaker1/audio2
audiosPath/speaker1/audio1 audiosPath/speaker1/audio3

  
audiosPath/speakerN/audio4 audiosPath/speakerN/audio3</pre>

Similar to the train file, we have also added a `--valid_data_dir` argument.

### Speaker Embedding Extraction

