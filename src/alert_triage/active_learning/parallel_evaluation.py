
import collections
import copy
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Manager
import time
import traceback

from sklearn.cross_validation import StratifiedKFold
import numpy

from alert_triage.active_learning import baseline
from alert_triage.active_learning import metrics
from alert_triage.active_learning import qbc
from alert_triage.active_learning import uncertainty


class ActiveLearningEvaluation(object):

    def __init__(self, data=None, targets=None, estimator=None,
                 max_budget=20, step_budget=1, percent_train=0.01,
                 folds=10):
        self._estimator = estimator
        self._data = data
        self._targets = targets
        self._label_budget = max_budget
        self._cv = StratifiedKFold(self._targets, shuffle=True, n_folds=folds)
        self._perc_train = percent_train
        self.metric = metrics.class_averaged_accuracy_score
        self._step_budget = step_budget


def evaluate(num_sub_iterations, evaluator):
    overall_results = collections.OrderedDict()
    start = time.time()
    for iteration, (train, test) in enumerate(evaluator._cv):
        start = time.time()
        manager = Manager()
        results = manager.dict()
        lock = manager.RLock()
        pool = Pool(processes=int(multiprocessing.cpu_count()))
        for _ in xrange(0, num_sub_iterations):
            pool.apply_async(_run_subiteration,
                        args=(lock, results, evaluator, train, test))
        pool.close()
        pool.join()
        for key, value in results.items():
            if overall_results.get(key) is None:
                overall_results[key] = numpy.array(value)
            else:
                overall_results[key] = numpy.vstack([overall_results[key],
                                              numpy.array(value)])
        print "Iteration:", iteration + 1, " Time:", (time.time() - start)
    return overall_results


def _run_subiteration(lock, strategy_results, evaluator, train, test):
    select_size = int(evaluator._perc_train * len(train))
    al_train = numpy.random.choice(train, size=select_size, replace=False)
    al_unlabeled = numpy.setdiff1d(train, al_train)
    try:
        models = [
            baseline.FullSampling(
                budget=evaluator._step_budget, model=evaluator._estimator),
            uncertainty.UncertaintySampling(
                budget=evaluator._step_budget, model=evaluator._estimator),
            qbc.QueryByCommittee(
                budget=evaluator._step_budget, model=evaluator._estimator),
            baseline.RandomSampling(
                budget=evaluator._step_budget, model=evaluator._estimator),
            baseline.NoSampling(
                budget=evaluator._step_budget, model=evaluator._estimator),
        ]
        for active_learner in models:
            results = []
            train_data = copy.deepcopy(al_train)
            unlabeled = copy.deepcopy(al_unlabeled)
            active_learner.fit(X=evaluator._data[train_data, :],
                               y=evaluator._targets[train_data])
            preds = active_learner.predict(evaluator._data[test])
            results.append(evaluator.metric(evaluator._targets[test], preds))
            for _ in xrange(0, evaluator._label_budget,
                evaluator._step_budget):
                add_data = active_learner.query(evaluator._data,
                                                evaluator._targets,
                                                unlabeled)
                if len(add_data) > 0:
                    train_data = numpy.unique(numpy.hstack([train_data,
                                                            add_data]))
                    #for datum in add_data.tolist():
                    #print "Before query:", unlabeled.shape
                    unlabeled = numpy.setdiff1d(al_unlabeled, train_data)
                    #print "After query:", unlabeled.shape
                    active_learner.fit(X=evaluator._data[train_data, :],
                                       y=evaluator._targets[train_data])
                preds = active_learner.predict(evaluator._data[test])
                results.append(evaluator.metric(evaluator._targets[test],
                                                preds))
            key = str(active_learner)
            lock.acquire()
            if strategy_results.get(key) is None:
                strategy_results[key] = numpy.array(results)
            else:
                old_results = strategy_results.get(key)
                strategy_results[key] = numpy.vstack([old_results,
                                                     numpy.array(results)])
            lock.release()
    except Exception:
        print traceback.format_exc()

