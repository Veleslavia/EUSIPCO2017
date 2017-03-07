import os
import numpy as np

from keras import backend as K
from keras.models import Sequential
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, Dense
from keras.layers.advanced_activations import LeakyReLU

import librosa

from ..settings import *

MODEL_KEY = 'han_base_model'
BASE_NAME = os.path.splitext(os.path.basename(__file__))[0]
N_SEGMENTS_PER_TRAINING_FILE = 3
SAMPLES_PER_EPOCH = N_TRAINING_SET * N_SEGMENTS_PER_TRAINING_FILE * TRAIN_SPLIT
SAMPLES_PER_VALIDATION = N_TRAINING_SET * N_SEGMENTS_PER_TRAINING_FILE * VALIDATION_SPLIT
N_MEL_BANDS = 128
SEGMENT_DUR = 43


def build_model(n_classes):
    model = Sequential()
    if K.image_dim_ordering() == 'th':
        input_shape = (1, N_MEL_BANDS, SEGMENT_DUR)
    else:
        input_shape = (N_MEL_BANDS, SEGMENT_DUR, 1)

    model.add(ZeroPadding2D(padding=(1, 1), input_shape=input_shape))
    model.add(Convolution2D(32, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block1_conv1'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(32, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block1_conv2'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(MaxPooling2D((3, 3), strides=(3, 3), name='block1_pool'))
    model.add(Dropout(0.25))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block2_conv1'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(64, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block2_conv2'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(MaxPooling2D((3, 3), strides=(3, 3), name='block2_pool'))
    model.add(Dropout(0.25))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(128, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block3_conv1'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(128, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block3_conv2'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(MaxPooling2D((3, 3), strides=(3, 3), name='block3_pool'))
    model.add(Dropout(0.25))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Convolution2D(256, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block4_conv1'))
    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(LeakyReLU(alpha=0.33))
    model.add(Convolution2D(256, 3, 3,
                            border_mode='same',
                            init='glorot_uniform',
                            name='block4_conv2'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(GlobalMaxPooling2D())
    model.add(Dense(1024,
                    name='fc1'))
    model.add(LeakyReLU(alpha=0.33))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='sigmoid', name='prediction'))
    return model


def compute_spectrograms(filename):
    out_rate = 22050

    frames, rate = librosa.load(filename, sr=out_rate, mono=True)
    if len(frames) < out_rate:
        # if less then 1 second - can't process
        raise Exception("Audio duration is too short")

    normalized_audio = _normalize(frames)
    melspectr = librosa.feature.melspectrogram(y=normalized_audio, sr=out_rate, n_mels=N_MEL_BANDS, fmax=out_rate/2)
    logmelspectr = librosa.logamplitude(melspectr**2, ref_power=1.0)

    # now going through spectrogram with the stride of the segment duration
    for start_idx in range(0, logmelspectr.shape[1] - SEGMENT_DUR + 1, SEGMENT_DUR):
        yield logmelspectr[:, start_idx:start_idx + SEGMENT_DUR]


def _normalize(audio_signal):
    return audio_signal/np.amax(audio_signal)