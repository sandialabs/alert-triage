


class ActiveLearner(object):
    def __init__(self, budget=1, model=None):
        self._budget = budget
        self.model = model
        self._queries = []

    def fit(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def query(self, X, y, unlabeled_indices):
        raise NotImplementedError()
