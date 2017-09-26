import numpy

from alert_triage.active_learning import abstract
from alert_triage.active_learning import uncertainty

def hard_vote_entropy(X, ensemble):
    pass


def soft_vote_entropy(X, ensemble):
    pass


def kl_disagreement(X, ensemble):
    ensemble_probs = ensemble.predict_proba(X)
    kl_div = None
    model_count = 0.
    for model in ensemble:
        model_probs = model.predict_proba(X)
        divs = uncertainty.kl_divergence(model_probs, ensemble_probs)
        if kl_div is None:
            kl_div = divs
        else:
            kl_div += divs
        model_count += 1.
    return kl_div / model_count


class QueryByCommittee(abstract.ActiveLearner):
    def __str__(self):
        return "QBC"

    def query(self, X, y, unlabeled_indices):
        if len(unlabeled_indices) == 0:
            return numpy.array([])
        results = kl_disagreement(X[unlabeled_indices, :], self.model)
        indices = numpy.argsort(-results)
        min_index = min(len(unlabeled_indices), self._budget)
	unlabeled_indices = numpy.asarray(unlabeled_indices)
        query = unlabeled_indices[numpy.array(indices[0:min_index])]
        self._queries.append(query)
        return query
