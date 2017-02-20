import os
import numpy as np

from experiments.models.han16 import compute_spectrograms, BASE_NAME
#from experiments.models.mmotivated import compute_spectrograms, BASE_NAME
from experiments.settings import *


def preprocess():
    for data_path, feature_path in [(IRMAS_TRAIN_DATA_PATH, IRMAS_TRAIN_FEATURE_BASEPATH),
                                    (IRMAS_TEST_DATA_PATH, IRMAS_TEST_FEATURE_BASEPATH)]:
        for root, dirs, files in os.walk(data_path):
            files = [filename for filename in files if filename.endswith('.wav')]
            for filename in files:
                for i, spec_segment in enumerate(compute_spectrograms(os.path.join(root, filename))):
                    feature_filename = os.path.join(feature_path, BASE_NAME,
                                                    "{filename}_{segment_idx}".format(filename=filename,
                                                                                      segment_idx=i))
                    np.save(feature_filename, spec_segment)


if __name__ == "__main__":
    print(BASE_NAME)
    preprocess()
