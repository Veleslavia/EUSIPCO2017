import os

from keras import backend as K
from keras.models import Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten
from keras.layers import Input, Dense, Dropout, merge
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2

from ..settings import *

MODEL_KEY = 'multi_layer_musically_motivated_model'
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

    maxpool_const = 4
    m_sizes = [5, 80]
    n_sizes = [1, 3, 5]
    n_filters = [128, 64, 32]

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
            x = MaxPooling2D(pool_size=(N_MEL_BANDS/maxpool_const, SEGMENT_DUR/maxpool_const),
                             name=str(n_i)+'_'+str(m_i)+'_'+'pool')(x)
            layers.append(x)

    x = merge(layers, mode='concat', concat_axis=channel_axis)

    x = Dropout(0.25)(x)
    x = Convolution2D(128, 3, 3, init='he_normal', W_regularizer=l2(1e-5), border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2')(x)

    x = Dropout(0.25)(x)
    x = Convolution2D(128, 3, 3, init='he_normal', W_regularizer=l2(1e-5), border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3')(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, init='he_normal', W_regularizer=l2(1e-5), name='fc1')(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, init='he_normal', W_regularizer=l2(1e-5), activation='softmax', name='prediction')(x)
    model = Model(melgram_input, x)

    return model
