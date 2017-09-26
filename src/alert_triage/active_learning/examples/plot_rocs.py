import time

import numpy
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.qda import QDA

from alert_triage.active_learning import metrics
from alert_triage.evaluation import model_evaluation

def evaluate(data, targets):
    prior = numpy.bincount(y.astype(int)) / float(len(targets))
    models = [
        LDA(priors=prior),
        SVC(probability=True, class_weight="auto", kernel="linear"),
        LogisticRegression(class_weight="auto"),
        GaussianNB(),
        KNeighborsClassifier(),
        QDA(priors=prior),
        RandomForestClassifier(n_estimators=100, criterion="entropy",
                n_jobs=-1, random_state=123456),
        SVC(probability=True, class_weight="auto")
    ]

    model_names = [
        "LDA",
        "Linear SVM",
        "Logistic Regression",
        "Naive Bayes",
        "k-NN",
        "QDA",
        "Random Forest",
        "SVM w/ RBF"
    ]

    # evaluate using ModelEvaluation class
    mevaluator = model_evaluation.TenFoldCrossValidation(
        data=data, targets=targets, models=models,
        model_names=model_names, scale=True)

    start = time.time()
    caa_eval = mevaluator.evaluate(metrics.class_averaged_accuracy_score)
    for key, value in caa_eval.iteritems():
        model_str = key.split("(")[0]
        print model_str, (str(numpy.around(numpy.mean(value), decimals=3)) +
            " (" + str(numpy.around(numpy.std(value), decimals=3)) + ")")
    mevaluator.evaluate_roc()
    print "Overall running time:", (time.time() - start)


def load_data_csv(data_file, target_file):
    X = numpy.loadtxt(data_file, delimiter=',')
    y = numpy.loadtxt(target_file, delimiter=',')
    return (X, y)


def load_data_npy(data_file, target_file):
    X = numpy.load(data_file)
    y = numpy.load(target_file)
    if "malware_skew" in data_file:
        rf = RandomForestClassifier(n_estimators=100, criterion="entropy",
                n_jobs=-1)
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = numpy.argsort(importances)[::-1]
        X = X[:, indices[0:233]]
    return (X.astype(float), y.astype(int))


if __name__ == "__main__":
    #Data are unavaible in production version
    print "Data are unavailable to run this test"
    X, y = load_data_npy("../../data/scot2/data.npy",
        "../../data/scot2/labels.npy")
    evaluate(data=X, targets=y)
