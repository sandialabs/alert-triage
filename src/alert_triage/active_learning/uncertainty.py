# Need to use numpy random_state
import math

import numpy
from scipy import stats

from alert_triage.active_learning import abstract

def entropy2(X):
    logs = numpy.where(X > 0, numpy.log(X), 0)
    return -numpy.sum(X*logs, axis=1)


def entropy(X):
    return numpy.apply_along_axis(stats.entropy, axis=1, arr=X)


def kl_divergence(X, Y):
    KLs = []
    for i in xrange(X.shape[0]):
        KLs.append(stats.entropy(X[i], Y[i]))
    return numpy.array(KLs)


def renyi_entropy(X, alpha=2):
    if alpha == 1.0:
        return entropy(X)
    entropies = []
    for i in xrange(X.shape[0]):
        ent = 0.0
        for j in xrange(X.shape[1]):
            freq = X[i, j]
            if freq != 0:
                ent = ent + freq ** alpha
        entropies.append(alpha / (1.0 - alpha) * math.log(ent ** (1.0 / alpha)))
    return numpy.array(entropies)


def least_confident(X):
    return 1.0 - X.max(1)


def margin(X):
    maxes = numpy.argsort(-X)[:, 2]
    return numpy.diff(X[maxes])


class UncertaintySampling(abstract.ActiveLearner):
    def __init__(self, budget=1, measure=entropy, model=None):
        self._budget = budget
        self._measure = measure
        #self.model = model
        self._queries = []
        super(UncertaintySampling, self).__init__(budget=budget, model=model)

    def __str__(self):
        return "Uncertainty (entropy)"

    def query(self, X, y, unlabeled_indices):
        if len(unlabeled_indices) == 0:
            return numpy.array([])
        if hasattr(self.model, "predict_proba"):
            probs = self.predict_proba(X[unlabeled_indices, :])
            results = self._measure(probs)
            indices = numpy.argsort(-results)
        else:
            results = self.model.decision_function(X[unlabeled_indices, :])
            results = numpy.absolute(results.flatten())
            indices = numpy.argsort(results)
        assert len(indices) == len(unlabeled_indices)
        min_index = min(len(unlabeled_indices), self._budget)
        query = unlabeled_indices[indices[0:min_index]]
        self._queries.append(query)
        return query
