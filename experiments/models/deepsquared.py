from keras.models import Model
from keras.layers import Input, SeparableConv2D, Conv2D, Dense
from keras.layers import merge, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
# Dropout, ZeroPadding2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2


def build_experimental_audio_model(n_classes, dense=True):

    input_shape = (96, 1366, 1)
    channel_axis = 3
    freq_axis = 1
    time_axis = 2

    audio_input = Input(shape=input_shape)

    # test rectangular convolutions
    x = Conv2D(64, 48, 3, subsample=(2, 2), bias=False, name='block1_conv1')(audio_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = ELU(name='block1_conv1_act')(x)
    x = Conv2D(64, 3, 3, bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = ELU(name='block1_conv2_act')(x)

    residual = Conv2D(128, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, 3, 24, border_mode='same', bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = ELU(name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, 3, 24, border_mode='same', bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block2_pool')(x)
    x = merge([x, residual], mode='sum')

    residual = Conv2D(256, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = ELU(name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = ELU(name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, 3, 3, border_mode='same', bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block3_pool')(x)

    x = merge([x, residual], mode='sum')

    residual = Conv2D(728, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = ELU(name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = ELU(name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), border_mode='same', name='block4_pool')(x)

    x = merge([x, residual], mode='sum')


    for i in range(4):
        residual = x
        prefix = 'block' + str(i + 5)

        x = ELU(name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = ELU(name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = ELU(name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = merge([x, residual], mode='sum')

    residual = Conv2D(1024, 1, 1, subsample=(2, 2),
                      border_mode='same', bias=False)(x)
    residual = BatchNormalization()(residual)

    x = ELU(name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, 3, 3, border_mode='same', bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = ELU(name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, 3, 3, border_mode='same', bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((2, 2), strides=(2, 2), border_mode='same', name='block13_pool')(x)
    x = merge([x, residual], mode='sum')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    if dense:
        x = Dense(n_classes, activation='softmax', name='predictions')(x)

    # Create model
    model = Model(audio_input, x)
    return model