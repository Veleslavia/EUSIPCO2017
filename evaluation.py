import argparse
import importlib
import os
import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from experiments.settings import *


class Evaluator:

    def __init__(self, model_module, weights_path):
        """
        Test metadata format
        ---------------------
        filename : string
        class_ids: string of ints with space as a delimiter
        """
        test_dataset = pd.read_csv(IRMAS_TESTING_META_PATH, names=["filename", "class_ids"])
        self.X = list(test_dataset.filename)
        targets = [[int(category) for category in target.split()] for target in test_dataset.class_ids]
        self.ml_binarizer = MultiLabelBinarizer().fit(targets)
        self.y_true = self.ml_binarizer.transform(targets)

        self.y_pred = np.zeros(shape=self.y_true.shape)
        self.model_module = model_module
        self.weights_path = weights_path
        self.feature_filenames = os.listdir(os.path.join(IRMAS_TEST_FEATURE_BASEPATH, model_module.BASE_NAME))
        self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_module.BASE_NAME)))

    def compute_prediction_scores(self, model):
        for i, (data_for_excerpt, filename) in enumerate(self._batch_generator(self.X)):
            one_excerpt_prediction = model.predict_on_batch(data_for_excerpt)
            self.y_pred[i, :] = self._normalize_prediction_s2(one_excerpt_prediction)

    def report_metrics(self):
        print("f1 macro: \t", f1_score(self.y_true, self.y_pred, average='macro'))
        print("f1 micro: \t", f1_score(self.y_true, self.y_pred, average='micro'))
        print("precision macro: \t", precision_score(self.y_true, self.y_pred, average='macro'))
        print("precision micro: \t", precision_score(self.y_true, self.y_pred, average='micro'))
        print("recall macro: \t", recall_score(self.y_true, self.y_pred, average='macro'))
        print("recall micro: \t", recall_score(self.y_true, self.y_pred, average='micro'))

    def evaluate(self):
        model = self.model_module.build_model(IRMAS_N_CLASSES)
        model.load_weights(self.weights_path)
        model.compile(optimizer='sgd', loss='categorical_crossentropy')
        self.compute_prediction_scores(model)
        self.report_metrics()

    def _load_features(self, audio_filename):
        features = list()
        for feature_filename in self.feature_filenames:
            if audio_filename in feature_filename:
                filename_full_path = os.path.join(IRMAS_TEST_FEATURE_BASEPATH,
                                                  self.model_module.BASE_NAME,
                                                  feature_filename)
                feature = np.load(filename_full_path)
                feature -= self.dataset_mean
                features.append(feature)

        if K.image_dim_ordering() == 'th':
            features = np.array(features).reshape(-1, 1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR)
        else:
            features = np.array(features).reshape(-1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR, 1)
        return features

    def _batch_generator(self, inputs):
        for audio_filename in inputs:
            yield self._load_features(audio_filename), audio_filename

    def _normalize_prediction_s2(self, predictions):
        # TODO normalize prediction by strategy S2
        threshold = 0.5
        prediction_sum = np.zeros(IRMAS_N_CLASSES)
        for prediction in predictions:
            prediction_sum += prediction
        prediction_sum /= prediction_sum.max()
        """
        low_values_indices = prediction_sum < threshold
        prediction_sum[low_values_indices] = 0
        """
        prediction_sum[prediction_sum >= threshold] = 1
        prediction_sum[prediction_sum < threshold] = 0
        return prediction_sum


def main():

    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model to import')
    aparser.add_argument('-w',
                         action='store',
                         dest='weights_path',
                         help='-w path to file with weights for selected model')
    args = aparser.parse_args()

    if not (args.model and args.weights_path):
        aparser.error('Please, specify the model and weights path to evaluate!')
    try:
        if args.model in ALLOWED_MODELS:
            model_module = importlib.import_module(".{}".format(args.model), "experiments.models")
            print "{} imported as 'model'".format(args.model)
        else:
            print "The specified model is not allowed"
        if not os.path.exists(args.weights_path):
            print "The specified weights path doesn't exist"
    except ImportError, e:
        print e

    evaluator = Evaluator(model_module, args.weights_path)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
