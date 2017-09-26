
import copy
import time

from sklearn.cross_validation import StratifiedKFold
import numpy as np

from alert_triage.active_learning import baseline
from alert_triage.active_learning import uncertainty
from alert_triage.active_learning import qbc
from alert_triage.active_learning import metrics

ACTIVE_LEARNING_STRATEGIES = [baseline.NoSampling(),
                              baseline.RandomSampling(),
                              #qbc.QueryByCommittee(),
                              uncertainty.UncertaintySampling(),
                              baseline.FullSampling()]

class ActiveLearningEvaluation(object):

    def __init__(self, data=None, targets=None, estimator=None,
                 max_budget=20, step_budget=1, percent_train=0.1, folds=10):
        self._estimator = estimator
        self._data = data
        self._targets = targets
        self._label_budget = max_budget
        self._cv = StratifiedKFold(self._targets, n_folds=folds)
        self._perc_train = percent_train
        self.metric = metrics.class_averaged_accuracy_score
        self._step_budget = step_budget


    def evaluate(self):
        overall_results = {}
        for iteration, (train, test) in enumerate(self._cv):
            print "Fold", iteration+1
            for subiter in xrange(10):
                print subiter+1
                select_size = int(self._perc_train * len(train))
                al_train = np.random.choice(train, size=select_size,
                                            replace=False)
                al_unlabeled = np.array([index for index in train.tolist()
                                if index not in al_train.tolist()])
                if len(al_unlabeled) < self._label_budget:
                    raise Exception("Query budget too big.")
                for al_strategy in ACTIVE_LEARNING_STRATEGIES:
                    print str(al_strategy)
                    results = []
                    train_data = copy.deepcopy(al_train)
                    unlabeled = copy.deepcopy(al_unlabeled)
                    al_strategy.model = copy.deepcopy(self._estimator)
                    al_strategy.model.fit(X=self._data[train_data],
                        y=self._targets[train_data])
                    preds = al_strategy.model.predict(self._data[test])
                    results.append(self.metric(self._targets[test], preds))
                    al_strategy.queries = []
                    start = time.time()
                    for _ in xrange(0, self._label_budget,
                        self._step_budget):
                        al_strategy._budget = self._step_budget
                        add_data = al_strategy.query(self._data, self._targets,
                            unlabeled)
                        if len(add_data) > 0:
                            train_data = (np.hstack([train_data, add_data]))
                            for datum in add_data.tolist():
                                unlabeled = unlabeled[unlabeled != datum]
                            al_strategy.model.fit(X=self._data[train_data],
                                                y=self._targets[train_data])
                        preds = al_strategy.model.predict(self._data[test])
                        results.append(self.metric(self._targets[test], preds))
                    print time.time() - start
                    key = str(al_strategy)
                    if overall_results.get(key) is None:
                        overall_results[key] = np.array(results)
                        print results
                    else:
                        overall_results[key] = np.vstack([overall_results[key],
                                                     np.array(results)])
        return overall_results
