import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.metrics import fbeta_score

#from experiments.models.han16 import *
from experiments.models.mmotivated import *
from experiments.settings import *


dataset = pd.read_csv(IRMAS_TRAINING_META_PATH, names=["filename", "class_id"])
X_train, X_val, y_train, y_val = train_test_split(list(dataset.filename),
                                                  to_categorical(np.array(dataset.class_id, dtype=int)),
                                                  test_size=VALIDATION_SPLIT,
                                                  random_state=5)
dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(BASE_NAME)))


def _load_features(filenames):
    features = list()
    for filename in filenames:
        feature_filename = os.path.join(IRMAS_TRAIN_FEATURE_BASEPATH, BASE_NAME, "{}.npy".format(filename))
        feature = np.load(feature_filename)
        feature -= dataset_mean
        features.append(feature)

    if K.image_dim_ordering() == 'th':
        features = np.array(features).reshape(-1, 1, N_MEL_BANDS, SEGMENT_DUR)
    else:
        features = np.array(features).reshape(-1, N_MEL_BANDS, SEGMENT_DUR, 1)
    return features


def _get_extended_data(inputs, targets):
    extended_inputs = list()
    for i in range(0, N_SEGMENTS_PER_TRAINING_FILE):
        extended_inputs.extend(['_'.join(list(x)) for x in zip(inputs, [str(i)]*len(inputs))])
    extended_inputs = np.array(extended_inputs)
    extended_targets = np.tile(np.array(targets).reshape(-1), N_SEGMENTS_PER_TRAINING_FILE).reshape(-1, IRMAS_N_CLASSES)
    return extended_inputs, extended_targets


def _batch_generator(inputs, targets):
    assert len(inputs) == len(targets)
    extended_inputs, extended_targets = _get_extended_data(inputs, targets)
    while True:
        indices = np.arange(len(extended_inputs))
        np.random.shuffle(indices)
        for start_idx in range(0, len(extended_inputs) - BATCH_SIZE + 1, BATCH_SIZE):
            excerpt = indices[start_idx:start_idx + BATCH_SIZE]
            yield _load_features(extended_inputs[excerpt]), extended_targets[excerpt]


def train(X_train, X_val, y_train, y_val):

    model = build_model(IRMAS_N_CLASSES)

    init_lr = 0.001
    optimizer = SGD(lr=init_lr, momentum=0.9, nesterov=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=EARLY_STOPPING_EPOCH)
    save_clb = ModelCheckpoint("{weights_basepath}/{model_path}/".format(weights_basepath=MODEL_WEIGHT_BASEPATH,
                                                                        model_path=BASE_NAME) +
                               "epoch.{epoch:02d}-{val_loss:.2f}"+"-{key}.hdf5".format(key=MODEL_KEY),
                               monitor='val_loss',
                               save_best_only=True)
    model.summary()
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy', fbeta_score])

    history = model.fit_generator(_batch_generator(X_train, y_train),
                                  samples_per_epoch=SAMPLES_PER_EPOCH,
                                  nb_epoch=MAX_EPOCH_NUM,
                                  verbose=2,
                                  callbacks=[save_clb, early_stopping],
                                  validation_data=_batch_generator(X_val, y_val),
                                  nb_val_samples=SAMPLES_PER_VALIDATION,
                                  class_weight=None,
                                  nb_worker=1)

    pickle.dump(history.history, open('{history_basepath}/{model_path}/history_{model_key}.pkl'.format(
        history_basepath=MODEL_HISTORY_BASEPATH,
        model_path=BASE_NAME,
        model_key=MODEL_KEY),
        'w'))


if __name__ == "__main__":
    train(X_train, X_val, y_train, y_val)