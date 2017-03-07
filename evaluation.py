import argparse
import importlib
import os

import numpy as np
import pandas as pd
from keras import backend as K
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from experiments.settings import *


class Evaluator(object):

    def __init__(self, model_module, weights_path, evaluation_strategy="s2"):
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
        self.y_pred_raw = np.zeros(shape=self.y_true.shape)
        self.y_pred_raw_average = np.zeros(shape=self.y_true.shape)
        self.model_module = model_module
        self.weights_path = weights_path
        self.feature_filenames = os.listdir(os.path.join(IRMAS_TEST_FEATURE_BASEPATH, model_module.BASE_NAME))
        self.dataset_mean = np.load(os.path.join(MODEL_MEANS_BASEPATH, "{}_mean.npy".format(model_module.BASE_NAME)))
        self.evaluation_strategy = evaluation_strategy
        self.thresholds_s1 = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24]
        self.thresholds_s2 = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    def compute_prediction_scores_raw(self, model):
        for i, (data_for_excerpt, filename) in enumerate(self._batch_generator(self.X)):
            one_excerpt_prediction = model.predict_on_batch(data_for_excerpt)
            if self.evaluation_strategy == "s2":
                self.y_pred_raw[i, :] = self._compute_prediction_sum(one_excerpt_prediction)
            else:
                self.y_pred_raw_average[i, :] = self._compute_prediction_sum(one_excerpt_prediction)

    def report_metrics(self, threshold):
        for average_strategy in ["micro", "macro"]:
            print("{} average strategy, threshold {}".format(average_strategy, threshold))
            print("precision:\t{}".format(precision_score(self.y_true, self.y_pred, average=average_strategy)))
            print("recall:\t{}".format(recall_score(self.y_true, self.y_pred, average=average_strategy)))
            print("f1:\t{}".format(f1_score(self.y_true, self.y_pred, average=average_strategy)))

    def evaluate(self):
        model = self.model_module.build_model(IRMAS_N_CLASSES)
        model.load_weights(self.weights_path)
        model.compile(optimizer="sgd", loss="categorical_crossentropy")
        self.compute_prediction_scores_raw(model)
        if self.evaluation_strategy == "s2":
            for threshold in self.thresholds_s2:
                self.y_pred = np.copy(self.y_pred_raw)
                for i in range(self.y_pred.shape[0]):
                    self.y_pred[i, :] /= self.y_pred[i, :].max()
                self.y_pred[self.y_pred >= threshold] = 1
                self.y_pred[self.y_pred < threshold] = 0
                self.report_metrics(threshold)
        else:
            for threshold in self.thresholds_s1:
                self.y_pred = np.copy(self.y_pred_raw_average)
                self.y_pred[self.y_pred < threshold] = 0
                self.y_pred[self.y_pred > threshold] = 1
                self.report_metrics(threshold)

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

        if K.image_dim_ordering() == "th":
            features = np.array(features).reshape(-1, 1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR)
        else:
            features = np.array(features).reshape(-1, self.model_module.N_MEL_BANDS, self.model_module.SEGMENT_DUR, 1)
        return features

    def _batch_generator(self, inputs):
        for audio_filename in inputs:
            yield self._load_features(audio_filename), audio_filename

    def _compute_prediction_sum(self, predictions):
        prediction_sum = np.zeros(IRMAS_N_CLASSES)
        for prediction in predictions:
            prediction_sum += prediction
        if self.evaluation_strategy == "s1":    # simple averaging strategy
            prediction_sum /= predictions.shape[0]
        return prediction_sum


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument("-m",
                         action="store",
                         dest="model",
                         help="-m model to evaluate")
    aparser.add_argument("-w",
                         action="store",
                         dest="weights_path",
                         help="-w path to file with weights for selected model")
    aparser.add_argument("-s",
                         action="store",
                         dest="evaluation_strategy",
                         default="s2",
                         help="-s evaluation strategy: `s1` (simple averaging and thresholding) or `s2` ("
                              "summarization, normalization by max probability and thresholding)")

    args = aparser.parse_args()

    if not (args.model and args.weights_path):
        aparser.error("Please, specify the model and the weights path to evaluate!")
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

    evaluator = Evaluator(model_module, args.weights_path, args.evaluation_strategy)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
