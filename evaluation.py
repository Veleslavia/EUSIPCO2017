from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

"""
Steps:
 - load model with model weights
 - load preprocessed data regarding to the model
 - compute activations for every piece
 - save activations
 - compute metrics for strategy2
 - report metrics
"""

from training import _load_features

test_dataset = None

def compute_activations():
    pass

def compute_metrics():

    y_true = [[1, 3], [0, 5], [1, 2], [4, 5]]
    """
    m.transform(y_true)
    array([[0, 1, 0, 1, 0, 0],
       [1, 0, 0, 0, 0, 1],
       [0, 1, 1, 0, 0, 0],
       [0, 0, 0, 0, 1, 1]])
    """

    y_pred = [[1], [1, 5], [1, 2, 3], [4, 5]]
    """
    Should be numpy array, overwise - raise exception!
    """

    m = MultiLabelBinarizer().fit(y_true)

    f1_score(m.transform(y_true), m.transform(y_pred), average='macro')
