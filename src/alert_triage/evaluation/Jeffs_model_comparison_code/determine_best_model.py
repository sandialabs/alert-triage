import numpy
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

from scipy import sparse

#import matplotlib.pyplot as plt

#from .. import active_learning_evaluation
#from .. import model_evaluation
#from ..active_learning import metrics
from model_comparison import ModelComparison

# neural network
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork


def evaluate(data, targets):
    print "Creating models..."
    models = []
    models.append(LinearSVC())
    models.append(SVC(kernel='rbf'))
    models.append(GaussianNB())
    models.append(LDA())
    models.append(QDA())
    models.append(LogisticRegression())
    models.append(KNeighborsRegressor())
    models.append(RandomForestClassifier(n_estimators=100, criterion="entropy",
                                   random_state=1234, n_jobs=-1))

    if sparse.issparse(data):
        data = data.toarray()

    mc = ModelComparison(data, targets, folds=10, numCV=3, models=models)
    mc.evaluate()

    #print "\n\n\n"
    #for d in results:
    #    print d['model']
    #    for m,p in d['pValues'].iteritems():
    #        print "    " + str(m)[:10] + " - " + str(p)


    # evaluate using ModelEvaluation class
    #mevaluator = model_evaluation.ModelEvaluation(data=data, targets=targets,
    #                                                     estimator = model)
    #evaluator = active_learning_evaluation.ActiveLearningEvaluation(max_budget=1000,
    #                                                                data=data,
    #                                                                targets=targets,
    #                                                                estimator=model)
    #evaluator._cv = copy.deepcopy(mevaluator._cv)

    #cv_results = mevaluator.evaluate()
    #al_results = evaluator.evaluate()
    #do_plot(dataset_name, "Random Forest", cv_results, al_results)

def load_data(data_file, target_file):
    print "Reading in data..."
    X = numpy.loadtxt(data_file, delimiter=',')
    y = numpy.loadtxt(target_file, delimiter=',')
    return (X, y)


def neuralNetwork(X, Y):
    print "Creating dataset..."
    ds = SupervisedDataSet(len(X[0]), 1)

    for x, y in zip(X, Y):
        ds.addSample(x, y)

    print "Creating neural network..."
    n = buildNetwork(ds.indim, int(ds.indim), ds.outdim)
    print "Training neural network..."
    t = BackpropTrainer(n, ds, verbose=True)
    errors = t.trainUntilConvergence(maxEpochs=10)
    return n


if __name__ == "__main__":
    X, Y = load_data("normalized_data.csv", "/opt/triage/targets.csv")
    evaluate(X, Y)



def nn(X, Y):
    n = neuralNetwork(X, Y)
    #N = 20
    #print("Predictions:")

    num_actually_zero = 0
    num_actually_one = 0

    guess_zero_actually_one = 0
    guess_zero_actually_zero = 0
    guess_one_actually_one = 0
    guess_one_actually_zero = 0

    for x, y in zip(X, Y):
        p = round(n.activate(x)[0])
        if y == 0.0:
            num_actually_zero += 1
            if p == 0.0:
                guess_zero_actually_zero += 1
            elif p == 1.0:
                guess_one_actually_zero += 1
        elif y == 1.0:
            num_actually_one += 1
            if p == 0.0:
                guess_zero_actually_one += 1
            elif p == 1.0:
                guess_one_actually_one += 1

    acc = ((guess_zero_actually_zero / float(num_actually_zero)) + (guess_one_actually_one / float(num_actually_one)))/(2.0)
    print "Accuracy: " + str(acc)
