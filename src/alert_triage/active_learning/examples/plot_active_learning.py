import time

import numpy
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
import matplotlib.pyplot as plt

from alert_triage.active_learning import parallel_evaluation

def do_plot(dataset_name, al_results, step, x_max, N):
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = fig1.add_subplot(111)
    fig1.set_size_inches(28, 16)

    minimum = 1.00
    maximum = 0
    x = numpy.arange(0, x_max, step)
    for key, value in al_results.iteritems():
        n = value.shape[0]
        y = numpy.mean(value, axis=0)
        if key == "Full Sampling (all labels)":
            y[0] = y[1]
        if numpy.max(y) > maximum:
            maximum = numpy.max(y)
        if numpy.min(y) < minimum:
            minimum = numpy.min(y)
        yerr = stats.sem(value, axis=0)
        #yerr = numpy.std(value, axis = 0)/math.sqrt(n)
        numpy.save(dataset_name + key.lower(), value)
        plt.errorbar(x, y, xerr=0, yerr=yerr, label=key, errorevery=3,
            lw=3, capsize=6, capthick=3)

    plt.setp(ax1.get_xticklabels(), fontsize=30)
    plt.setp(ax1.get_yticklabels(), fontsize=30)
    plt.subplots_adjust(left=0.075, right=0.925, top=0.925, bottom=0.075)
    plt.ylim([max(0.00, minimum - 0.25), min(maximum + 0.2, 1.00)])
    plt.xlim((0, x_max-step))
    max_ticks = x_max/(step*5)
    plt.xticks(numpy.arange(0, x_max-step+1, int(step*max_ticks)))
    plt.xlabel("# of Queried Labels", fontsize=40, labelpad=10)
    plt.ylabel("CAA", fontsize=40, labelpad=10)
    plt.legend(loc="lower right", fontsize=36, frameon=False)
    pp = PdfPages(dataset_name + 'avp.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()


def evaluate(data, targets, dataset_name):
    model = RandomForestClassifier(n_estimators=100, criterion="entropy",
                                   random_state=1234, n_jobs=-1)
    evaluator = parallel_evaluation.ActiveLearningEvaluation(
        max_budget=4000, step_budget=100, data=data,
        targets=targets, estimator=model, percent_train=0.01)

    start = time.time()
    print "\n10-fold cross-validation, active learning"
    print "------------------------------------------------"
    al_results = parallel_evaluation.evaluate(10, evaluator)

    do_plot(dataset_name, al_results, evaluator._step_budget,
            evaluator._label_budget + evaluator._step_budget,
            data.shape[0])
    print evaluator.__dict__
    print "Overall running time:", (time.time() - start)


def load_data_npy(data_file, target_file):
    X = numpy.load(data_file)
    y = numpy.load(target_file)
    return (X.astype(float), y.astype(float))


if __name__ == "__main__":
    prefix = "../../data/"
    dataset = "registry"
    (X, y) = load_data_npy(prefix + dataset + "/data.npy",
        prefix + dataset + "/labels.npy")
    print X.shape, numpy.sum(y)
    evaluate(X, y, prefix + dataset + "/results/")
