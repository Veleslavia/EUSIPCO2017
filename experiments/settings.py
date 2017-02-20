"This file contains dataset settings, such as paths, number of categories etc"

IRMAS_TRAIN_DATA_PATH = '/home/olga/IRMAS/IRMAS-TrainingData/'
IRMAS_TEST_DATA_PATH = '/home/olga/IRMAS/IRMAS-TestingData/'
IRMAS_TRAIN_FEATURE_BASEPATH = '/home/olga/IRMAS/IRMAS-TrainingData-Features'
IRMAS_TEST_FEATURE_BASEPATH = '/home/olga/IRMAS/IRMAS-TestingData-Features'
IRMAS_TRAINING_META_PATH = '/home/olga/IRMAS/IRMAS-TrainingData/irmas_train_meta.csv'
MODEL_WEIGHT_BASEPATH = './weights/'
MODEL_HISTORY_BASEPATH = './history/'
MODEL_MEANS_BASEPATH = './means/'
IRMAS_N_CLASSES = 11
TRAIN_SPLIT = 0.85
VALIDATION_SPLIT = 0.15
N_TRAINING_SET = 6705
MAX_EPOCH_NUM = 200
EARLY_STOPPING_EPOCH = 10
SGD_LR_REDUCE = 5
BATCH_SIZE = 16