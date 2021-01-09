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

This repository shoud allow you to train a speaker embedding extractor according to the setup described in the paper. This speaker embedding extractor is based on a VGG-based classifier which identifies speaker identities given variable length audio utterances. The network used for this work uses log mel-spectogram features as input. Hence, we have added here the instructions to reproduce the feature extraction, the network training and the speaker embedding extraction step. Feel free to ask any doubt via git-hub issues, [twitter](https://twitter.com/mikiindia) or mail(miquel.angel.india@upc.edu).

### Feature Extraction

You can find in `scripts/featureExtractor.py` several functions which extract and normalize the log mel-spectogram descriptors. If you want to run the whole feature extraction over a set of audios you can run the following command:

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

Once you have extracted the features from all the audios wanted to be used, It is needed to prepare some path files for the training step. The proposed models are trained as speaker classifiers, hence a classification-based loss and an accuracy metric will be used to monitorize the training progress. However in the validation step, an EER estimation is used to validate the network progress. The motivation behind this is that best accuracy models do not always have the best inter/intra speaker variability. Therefore we prefer to  use directly a task based metric to validate the model instead of using a classification one. Two different kind of path files will then be needed for the training/validation procedures:

Train Labels File (`train_labels_path`):

This file must have three columns separated by a blank space. The first column must contain the audio utterance paths, the second column must contain the speaker labels and the third one must be filled with -1. It is assumed that the labels correspond to the output network labels. Hence if you are working with a N speakers database, the speaker labels values should be in the 0 to N-1 range.

File Example:

<pre>
audiosPath/speaker1/audio1 0 -1
audiosPath/speaker1/audio2 0 -1
...
audiosPath/speakerN/audio4 N-1 -1</pre>

We have also added a `--train_data_dir` path argument. The dataloader will then look for the features in `--train_data_dir` + `audiosPath/speakeri/audioj` paths.

Valid Labels File:

For the validation step, it will be needed a tuple of client/impostors trial files. Client trials (`valid_clients`) file must contain pairs of audio utterances from same speakers and the impostors trials (`valid_impostors`) file must also contain audio utterance pairs but from different speakers. Each pair path must be separated with a blank space:

File Example (Clients):

<pre>
audiosPath/speaker1/audio1 audiosPath/speaker1/audio2
audiosPath/speaker1/audio1 audiosPath/speaker1/audio3

  
audiosPath/speakerN/audio4 audiosPath/speakerN/audio3</pre>

Similar to the train file, we have also added a `--valid_data_dir` argument.

Once you have all these data files ready, you can launch a model training with the following command:


```bash
python scripts/train.py
```

With this script you will launch the model training with the default setup defined in `scripts/train.py`. The model will be trained following the methods and procedures described in the paper. The best models found will be saved in the `--out_dir` directory. You will find there a `.pkl` file with the training/model configuration and several checkpoint `.pt` files which store model weghts, optimizer state values, etc. The best saved models correspond to the last saved checkpoints.

 
### Speaker Embedding Extraction

Given a trained model, this one can be used to extract a speaker embedding from a variable-length audio. We have added a script example to show how to use the models to extract speaker embeddings. This can be then used to extract similiary scores between audios computing the cosine distance between their embeddings. Run the following command:

```bash
python scripts/getEmbeddingExample.py --audioPath <path_to_audio.wav> --modelConfig <path_to_config_file.pkl> --modelCheckpoint <path_to_checkpoint_file.chkpt>
```
This script will load the model and will extract/print the embedding given the input audio.

