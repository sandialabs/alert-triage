import numpy
import pylab as pl
from sklearn import ensemble

_DEBUG = True


def plot_importance(importances, stdev, filename):
    indices = numpy.ravel(numpy.where(numpy.around(importances,
                                                   decimals=4) > 0))
    pl.figure()
    pl.title("Feature importances")
    pl.bar(range(len(importances[indices])), importances[indices],
           color="r", yerr=stdev[indices], align="center")
    pl.xticks(range(len(importances[indices])), indices)
    pl.xlim([-1, len(importances[indices])])
    pl.savefig(filename)


class RandomForestFeatureEvaluation(object):
    def __init__(self, rf_model=ensemble.RandomForestClassifier(
        n_estimators = 500, criterion = "entropy", compute_importances = True)):
        self._rf_model = rf_model

    def evaluate(self, X, y, feature_names=None, filename=None):
        self._rf_model.fit(X, y)
        importances = self._rf_model.feature_importances_
        individual_importances = [tree.feature_importances_
                                  for tree in self._rf_model.estimators_]
        stdev = numpy.std(individual_importances, axis=0)
        indices = numpy.argsort(importances)[::-1]

        if feature_names is None:
            feature_names = ["feature " + str(i) for i in xrange(len(indices))]

        if _DEBUG:
            # Print the feature ranking
            print "Feature ranking:"
            for i in xrange(len(importances)):
                print("%d. %s (%f)" % (i + 1, feature_names[indices[i]],
                                       importances[indices[i]]))

        feature_names = numpy.array(feature_names)
        if filename is not None:
            plot_importance(importances[indices], stdev[indices], filename)
