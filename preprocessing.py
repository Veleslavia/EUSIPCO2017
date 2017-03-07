import argparse
import importlib
import os
import numpy as np

from experiments.settings import *


def preprocess(model_module):
    for data_path, feature_path in [(IRMAS_TRAIN_DATA_PATH, IRMAS_TRAIN_FEATURE_BASEPATH),
                                    (IRMAS_TEST_DATA_PATH, IRMAS_TEST_FEATURE_BASEPATH)]:
        for root, dirs, files in os.walk(data_path):
            files = [filename for filename in files if filename.endswith('.wav')]
            for filename in files:
                for i, spec_segment in enumerate(model_module.compute_spectrograms(os.path.join(root, filename))):
                    feature_filename = os.path.join(feature_path, model_module.BASE_NAME,
                                                    "{filename}_{segment_idx}".format(filename=filename,
                                                                                      segment_idx=i))
                    np.save(feature_filename, spec_segment)


def main():
    aparser = argparse.ArgumentParser()
    aparser.add_argument('-m',
                         action='store',
                         dest='model',
                         help='-m model for preprocessing')
    args = aparser.parse_args()

    if not args.model:
        aparser.error('Please, specify the model!')
    try:
        if args.model in ALLOWED_MODELS:
            model_module = importlib.import_module(".{}".format(args.model), "experiments.models")
            print "{} imported as 'model'".format(args.model)
        else:
            print "The specified model is not allowed"
    except ImportError, e:
        print e
    preprocess(model_module)


if __name__ == "__main__":
    main()
