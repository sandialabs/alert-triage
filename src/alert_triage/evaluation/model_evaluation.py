"""Evaluate selected model classes on a dataset using given metrics.

DEBUG: boolean variable to set if you want to print debugging statements.
ModelEvaluationException: exception type that should be
raised when exceptions or errors occur.
ModelEvaluation:

"""
import collections
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import numpy
from numpy import random
import pylab as pl
import scipy
from sklearn import cross_validation
from sklearn import metrics
from alert_triage.active_learning import metrics as mt_metrics
from sklearn import preprocessing
import numpy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.qda import QDA
import time


from matplotlib.backends.backend_pdf import PdfPages

DEBUG = True


def load_data_npy(data_file, target_file):
    X = numpy.load(data_file)
    y = numpy.load(target_file)
    return (X.astype(float), y.astype(float))


def plot_rocs(models, mean_fprs, mean_tprs, filename="roc"):
    if filename is None:
        return
    fig1 = pl.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    fig1.set_size_inches(28, 16)
    lines = ["-", "--"]

    # Plot ROC Curves
    for i in xrange(len(models)):
        name = models[i]
        mean_fpr = mean_fprs[i]
        mean_tpr = mean_tprs[i]
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        ax1.plot(mean_fpr, mean_tpr, linestyle=lines[i % 2],
                label=str(name) + " (AUC = %0.4f)" % mean_auc, lw=5.5)

    ax1.plot([0, 1], [0, 1], "--", color=(0.6, 0.6, 0.6),
             label="Random Guessing (AUC = %0.4f)" % 0.5, lw=4)
    pl.setp(ax1.get_xticklabels(), fontsize=36)
    pl.setp(ax1.get_yticklabels(), fontsize=36)
    pl.xlim([-0.01, 1.01])
    pl.ylim([-0.01, 1.01])
    pl.xlabel("False Positive Rate", fontsize=40, labelpad=15)
    pl.ylabel("True Positive Rate", fontsize=40, labelpad=15)
    #pl.title("ROC Curve", fontsize=40)
    pl.legend(loc="lower right", fontsize=40, frameon=False)

    # Zoom inset
    ax2 = pl.axes([.55, .595, .405, .35], axisbg="white")
    FPR = 0.15
    sub_indices = numpy.ravel(numpy.where(mean_fprs[0] < FPR))
    max_tpr = 0.0
    for i in xrange(len(models)):
        mean_tpr = mean_tprs[i]
        mean_fpr = mean_fprs[i]
        if numpy.max(mean_tpr[sub_indices]) > max_tpr:
            max_tpr = numpy.max(mean_tpr[sub_indices])
        ax2.plot(mean_fpr[sub_indices], mean_tpr[sub_indices], lw=4,
                 linestyle=lines[i % 2],)
    ax2.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), lw=3)
    pl.setp(ax2.get_xticklabels(), fontsize=30)
    pl.setp(ax2.get_yticklabels(), fontsize=30)
    pl.xlim([-0.01, FPR])
    pl.xticks(numpy.arange(0, FPR+.05, .05))
    pl.ylim([-0.01, max_tpr])

    pl.subplots_adjust(left=0.06, right=0.975, top=0.95, bottom=0.08)
    #pl.savefig(filename, dpi = 100)
    pp = PdfPages(filename + ".pdf")
    pl.savefig(pp, format='pdf')
    pp.close()


class NoScaler(object):
    def fit(self, X):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return X


class CrossValidation(object):

    def __init__(self, data, targets, models, model_names, num_iterations=1,
                 num_folds=10, scale=True, seed=123456):
        assert len(models) == len(model_names)
        self._data = data
        self._targets = targets
        self.models = models
        self._model_names = model_names
        self._num_iterations = num_iterations
        self._num_folds = num_folds
        self._scaler = NoScaler()
        if scale:
            self._scaler = preprocessing.MinMaxScaler()
        self._seed = seed

    def evaluate(self, metric, is_prediction_metric=True, time_series=True):
        random.seed(self._seed)
        overall_results = collections.OrderedDict()
        for model in self.models:
            for iteration in xrange(self._num_iterations):
                random_indices = numpy.arange(0, self._data.shape[0])
                random.shuffle(random_indices)
                cv_iterator = cross_validation.StratifiedKFold(
                    self._targets[random_indices], n_folds=self._num_folds,
                    random_state=self._seed)
                for fold, (train, test) in enumerate(cv_iterator):
                    train_indices = random_indices[train]
                    test_indices = random_indices[test]
                    train_X = self._scaler.fit_transform(
                        self._data[train_indices, :])
                    test_X = self._scaler.transform(self._data[test_indices, :])
                    model.fit(train_X, self._targets[train_indices])
                    importances = model.feature_importances_

                    std = numpy.std([tree.feature_importances_ for tree in model.estimators_],
                                 axis=0)
                    indices = numpy.argsort(importances)[::-1]

                    # Print the feature ranking
                    print("Feature ranking:")
                    print indices

                    not_used = [i for i, j in enumerate(importances) if j == 0]
                    print not_used

                    for f in range(len(indices)):
                        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.title("Feature importances")
                    plt.bar(range(len(indices)), importances[indices],
                    color="r", yerr=std[indices], align="center")
                    plt.xticks(range(len(indices)), indices, rotation='vertical', fontsize=7)
                    plt.xlim([-1, 70])
                    plt.savefig('feature_importance.png')

                    if is_prediction_metric:
                        predictions = model.predict(test_X)
                    else:
                        if hasattr(model, "predict_proba"):
                            predictions = model.predict_proba(test_X)[:, 1]
                        else:
                            predictions = model.decision_function(test_X)
                    result = metric(self._targets[test_indices], predictions)
                        
                    if DEBUG:
                        print '==============================================='
                        print '\t\tIteration', iteration+1, 'FOLD', fold+1
                        print '==============================================='
                        print '\t\tModel'
                        print '==============================================='
                        print str(model) + '\n'
                        print "CAA", result
                        print "Recall", metrics.recall_score(self._targets[test_indices], predictions)
                    if overall_results.get(repr(model)) is None:
                        overall_results[repr(model)] = []
                    overall_results[repr(model)].append(result)
        return overall_results

    def evaluate_roc(self, filename="roc"):
        random.seed(self._seed)
        overall_results = collections.OrderedDict()
        mean_tprs = []
        mean_fprs = []
        for model in self.models:
            mean_tpr = 0.0
            mean_fpr = numpy.linspace(0, 1, int(self._data.shape[0] / 2))
            for iteration in xrange(self._num_iterations):
                random_indices = numpy.arange(0, self._data.shape[0])
                random.shuffle(random_indices)
                cv_iterator = cross_validation.StratifiedKFold(
                    self._targets[random_indices], n_folds=self._num_folds,
                    random_state=self._seed)
                for fold, (train, test) in enumerate(cv_iterator):
                    train_indices = random_indices[train]
                    test_indices = random_indices[test]
                    train_X = self._scaler.fit_transform(
                        self._data[train_indices, :])
                    test_X = self._scaler.transform(self._data[test_indices, :])
                    model.fit(train_X, self._targets[train_indices])
                    if hasattr(model, "predict_proba"):
                        probas_ = model.predict_proba(test_X)[:, 1]
                    else:
                        probas_ = model.decision_function(test_X)
                    # Compute ROC curve and area the curve
                    fpr, tpr, thresholds = metrics.roc_curve(
                        self._targets[test_indices], probas_)
                    mean_tpr += scipy.interp(mean_fpr, fpr, tpr)
                    mean_tpr[0] = 0.0
                    result = metrics.auc(fpr, tpr)
                    if DEBUG:
                        print "================================================"
                        print "\t\tIteration", iteration+1, "Fold", fold+1
                        print "================================================"
                        print "\t\tModel"
                        print "================================================"
                        print str(model)
                        print "AUC", result
                    if overall_results.get(repr(model)) is None:
                        overall_results[repr(model)] = []
                    overall_results[repr(model)].append(result)
            overall_results[repr(model)] = numpy.array(
                overall_results[repr(model)])
            mean_tpr /= float(len(overall_results[repr(model)]))
            mean_tpr[-1] = 1.0
            mean_tprs.append(mean_tpr)
            mean_fprs.append(mean_fpr)
        plot_rocs(self._model_names, mean_fprs, mean_tprs, filename=filename)
        return overall_results


class TenFoldCrossValidation(CrossValidation):

    def __init__(self, data, targets, models, model_names, scale=True):
        super(TenFoldCrossValidation, self).__init__(
            data=data, targets=targets, models=models,
            model_names=model_names,
            num_iterations=1, num_folds=10, scale=scale)


class FiveByTwoCrossValidation(CrossValidation):

    def __init__(self, data, targets, models, model_names, scale=True):
        super(FiveByTwoCrossValidation, self).__init__(
            data=data, targets=targets, models=models,
            model_names=model_names,
            num_iterations=5, num_folds=2, scale=scale)


if __name__ == "__main__":
    data, targets = load_data_npy("data/X.npy",
        "data/y.npy")

    print dir(data)
    models = [
     #   SVC(probability=True, class_weight="auto", kernel="linear"),
     #   LogisticRegression(class_weight="auto"),
     #   GaussianNB(),
     #   KNeighborsClassifier(),
        RandomForestClassifier(n_estimators=100, criterion="entropy",
                n_jobs=-1, random_state=123456),
     #   SVC(probability=True, class_weight="auto")
    ]

    model_names = [
    #    "Linear SVM",
    #    "Logistic Regression",
    #    "Naive Bayes",
    #    "k-NN",
        "Random Forest",
    #    "SVM w/ RBF"
    ]

    # evaluate using ModelEvaluation class
    mevaluator = TenFoldCrossValidation(
        data=data, targets=targets, models=models,
        model_names=model_names, scale=True)

    start = time.time()
    caa_eval = mevaluator.evaluate(mt_metrics.class_averaged_accuracy_score)
    for key, value in caa_eval.iteritems():
        model_str = key.split("(")[0]
        print model_str, (str(numpy.around(numpy.mean(value), decimals=3)) +
            " (" + str(numpy.around(numpy.std(value), decimals=3)) + ")")
    #mevaluator.evaluate_roc()
    print "Overall running time:", (time.time() - start)
