import os

from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense
from keras.layers import Input, Dropout, merge
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2

import librosa

from ..settings import *

MODEL_KEY = 'single_layer_musically_motivated_model'
BASE_NAME = os.path.splitext(os.path.basename(__file__))[0]
N_SEGMENTS_PER_TRAINING_FILE = 1
SAMPLES_PER_EPOCH = N_TRAINING_SET * N_SEGMENTS_PER_TRAINING_FILE * TRAIN_SPLIT
SAMPLES_PER_VALIDATION = N_TRAINING_SET * N_SEGMENTS_PER_TRAINING_FILE * VALIDATION_SPLIT
N_MEL_BANDS = 96
SEGMENT_DUR = 128


def build_model(n_classes):

    if K.image_dim_ordering() == 'th':
        input_shape = (1, N_MEL_BANDS, SEGMENT_DUR)
        channel_axis = 1
    else:
        input_shape = (N_MEL_BANDS, SEGMENT_DUR, 1)
        channel_axis = 3
    melgram_input = Input(shape=input_shape)

    m_sizes = [50, 70]
    n_sizes = [1, 3, 5]
    n_filters = [128, 64, 32]
    maxpool_const = 4

    layers = list()

    for m_i in m_sizes:
        for i, n_i in enumerate(n_sizes):
            x = Convolution2D(n_filters[i], m_i, n_i,
                              border_mode='same',
                              init='he_normal',
                              W_regularizer=l2(1e-5),
                              name=str(n_i)+'_'+str(m_i)+'_'+'conv')(melgram_input)
            x = BatchNormalization(axis=channel_axis, mode=0, name=str(n_i)+'_'+str(m_i)+'_'+'bn')(x)
            x = ELU()(x)
            x = MaxPooling2D(pool_size=(N_MEL_BANDS, SEGMENT_DUR/maxpool_const), name=str(n_i)+'_'+str(m_i)+'_'+'pool')(x)
            x = Flatten(name=str(n_i)+'_'+str(m_i)+'_'+'flatten')(x)
            layers.append(x)

    x = merge(layers, mode='concat', concat_axis=channel_axis)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, init='he_normal', W_regularizer=l2(1e-5), activation='softmax', name='prediction')(x)
    model = Model(melgram_input, x)

    return model


def compute_spectrograms(filename):
    out_rate = 12000
    N_FFT = 512
    HOP_LEN = 256

    frames, rate = librosa.load(filename, sr=out_rate, mono=True)
    if len(frames) < out_rate*3:
        # if less then 3 second - can't process
        raise Exception("Audio duration is too short")

    logam = librosa.power_to_db
    melgram = librosa.feature.melspectrogram
    x = logam(melgram(y=frames, sr=out_rate, hop_length=HOP_LEN,
                      n_fft=N_FFT, n_mels=N_MEL_BANDS) ** 2,
              ref=1.0)

    # now going through spectrogram with the stride of the segment duration
    for start_idx in range(0, x.shape[1] - SEGMENT_DUR + 1, SEGMENT_DUR):
        yield x[:, start_idx:start_idx + SEGMENT_DUR]
