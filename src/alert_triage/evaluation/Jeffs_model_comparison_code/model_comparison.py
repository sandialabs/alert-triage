"""Evaluate selected model classes on a dataset using given metrics upon
which models will be compared.

DEBUG: boolean variable to set if you want to print debugging statements.
ModelComparisonException: exception type that should be
raised when exceptions or errors occur.

"""

import numpy
import copy
from sklearn import cross_validation
from sklearn.utils import shuffle


import metrics
#from ..active_learning import metrics

DEBUG = True


class ModelComparisonException(Exception):

    """Exception type for the ModelComparison class."""

    pass


class ModelComparison(object):

    def __init__(self, data=None, targets=None, folds=10, numCV=3, models=[]):
        print "Creating model comparison..."
        self._data = data
        self._targets = targets
        self._folds = folds
        self._numCV = numCV
        self._models = copy.deepcopy(models)
        self._metric = metrics.class_averaged_accuracy_score

    def evaluate(self):
        print "Starting evaluation..."
        accResults = {}
        for model in self._models:
            accResults[str(model)] = {'accuracies':[]}
        for iteration in xrange(self._numCV):
            self._data, self._targets = shuffle(self._data, self._targets,
                                                random_state=iteration)
            self._cv = cross_validation.StratifiedKFold(self._targets,
                                                        indices=True,
                                                        n_folds=self._folds)
            for fold, (train, test) in enumerate(self._cv):
                if DEBUG:
                    print '\n'
                    print '===================================================='
                    print "'Iteration ' %s '  ---  FOLD ' %s" (str(iteration+1),
                                                               str(fold+1))
                    print '----------------------------------------------------'
                for originalModel in self._models:
                    model = copy.deepcopy(originalModel)
                    model.fit(self._data[train], self._targets[train])
                    predictions = model.predict(self._data[test])
                    result = self._metric(self._targets[test], predictions)
                    if DEBUG:
                        print str(model) + ": " + str(result)
                    accResults[str(model)]['accuracies'].append(result)
        for k, v in accResults.iteritems():
            accResults[k]['avgAccuracy'] = numpy.mean(accResults[k]['accuracies'])
            accResults[k]['stdDev'] = numpy.std(accResults[k]['accuracies'])
        #results = [ \
        #            { \
        #             'model':m, \
        #             'accuracy':sum(a['accuracies'])/float(len(a['accuracies'])), \
        #             'pValues':dict((str(m2),wilcoxon(a['accuracies'],a['accuracies'])[1]) for m2 in self._models if m!=m2) \
        #            } \
        #            for m,a in accResults.iteritems() \
        #          ]
        #results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        f = open('output.txt', 'w')
        for k, v in accResults.iteritems():
            f.write(str(k)+'\n')
            f.write("    Accuracy: " + str(v['avgAccuracy']) + '\n')
            f.write("   Std Dev:  " + str(v['stdDev']) + '\n')
            #f.write("    pValues:\n")
            #for m,p in result['pValues'].iteritems():
            #    f.write("        " + str(m) + ": " + str(p) + '\n')
        f.write("\n\n==============================================\n\n")
        #for k,v in accResults.iteritems():
        #    f.write(k+'\n')
        #    for k2,v2 in v.iteritems():
        #        f.write(k2 + ': ' + str(v2) + '\n')
        f.close()
        #return results
