
import pickle

import numpy
from sklearn import ensemble
from sklearn import metrics

import os
import alert_triage
from alert_triage.evaluation import feature
from alert_triage.evaluation import model_evaluation

_FEATURE_PLOT_FILE = "feature_importances.png"
_LABELED_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(alert_triage.__file__)), "data/")
_ROC_PLOT_FILE = "roc.png"

def read_data(datapath):
    X = numpy.load(datapath + "X.npy")
    y = numpy.load(datapath + "y.npy")
    infile = open(datapath + "feature_names.pkl", "rb")
    feature_names = pickle.load(infile)
    infile.close()
    return (X.astype(float), y.astype(float), feature_names)


class EvaluationPipeline(object):
    def __init__(self, active_learning_evaluation=None,
                 datapath=_LABELED_DATA_PATH,
                 feature_evaluation=feature.RandomForestFeatureEvaluation(),
                 evaluation=None):
        self._active_learning_evaluation = active_learning_evaluation
        self._datapath = datapath
        self._evaluation = evaluation
        self._feature_evaluation = feature_evaluation

    def run_pipeline(self):
        (X, y, feature_names) = read_data(self._datapath)
        print X.shape
        model = ensemble.RandomForestClassifier(n_estimators=100)
        evaluator = model_evaluation.FiveByTwoCrossValidation(
        	X, y, [model], model_names=["RandomForest"])
        print "Evaluate metrics..."
        #evaluator.evaluate(metric = metrics.accuracy_score)
        print "Evaluate ROC..."
        evaluator.evaluate_roc(_ROC_PLOT_FILE)
        if self._active_learning_evaluation is not None:
            self._active_learning_evaluation.evaluate(X, y)
        print "Evaluate features..."
        self._feature_evaluation.evaluate(X, y, feature_names,
                                          _FEATURE_PLOT_FILE)



if __name__ == "__main__":
    EvaluationPipeline().run_pipeline()
