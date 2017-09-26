import numpy

from alert_triage.active_learning import abstract

class NoSampling(abstract.ActiveLearner):
    def __str__(self):
        return "No Sampling"

    def query(self, X, y, unlabeled_indices):
        return numpy.array([])


class FullSampling(abstract.ActiveLearner):
    def __str__(self):
        return "Full Sampling (all labels)"

    def query(self, X, y, unlabeled_indices):
        self._queries.append(unlabeled_indices)
        return unlabeled_indices


class RandomSampling(abstract.ActiveLearner):
    def __str__(self):
        return "Random"

    def query(self, X, y, unlabeled_indices):
        if len(unlabeled_indices) == 0:
            return numpy.array([])
        N = self._budget
        if N > len(unlabeled_indices):
            N = len(unlabeled_indices)
        indices = numpy.random.choice(unlabeled_indices, N, replace=False)
        self._queries.append(indices)
        return indices
