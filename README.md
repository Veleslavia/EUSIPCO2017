# Timbre Analysis of Music Audio Signals with Convolutional Neural Networks: Musical instrument recognition

This repository contains code for musical instrument recognition experiments for the paper 
entitled "Timbre Analysis of Music Audio Signals with Convolutional Neural Networks".

We provide the code for data preprocessing, training and evaluation of our approach. 

## Dataset

We use IRMAS dataset. Please, download the dataset from [MTG website] (www.mtg.upf.edu/download/datasets/irmas) and change the paths to the training and testing splits at the settings file: `./experiments/settings.txt`

## Preprocessing

Before to run the preprocessing script, please, create specific folder for every model which you would like to preprocess the data for. 
The following notation is using (consult `./experiments/settings.txt`): `$IRMAS_HOME_DIRECTORY/IRMAS-TrainingData-Features/model_name` and `$IRMAS_HOME_DIRECTORY/IRMAS-TestingData-Features/model_name`, where `model_name` is referenced to one of the model filenames in `./experiments/models` folder.
Usage:

```bash
python preprocessing.py -m model_name
```

Currently supported models are `han16`, `singlelayer`, `multilayer` which are referenced in the paper as han16,
single-layer and multi-layer models respectively. 

## Training

Usage:

```bash
python training.py -m model_name -o optimizer_name [-l]
```

The option `-l` states for loading data into RAM at the beginning of the experiment instead of reading it batch-by-batch from the disk.

## Evaluation
 
Usage:
 
```bash
python evaluation.py -m model_name -w /path/to/weights/file.hdf5 -s evaluation_strategy
```

The weights for the reported models can be found at `./weights/model_name` folder.

Evaluation strategies are:
The `s1` strategy computes a mean activation through whole audio excerpt and apply identification threshold
The `s2` strategy computes sum of activations, normalize it by dividing by maximum activation.

## Reference

References are coming.