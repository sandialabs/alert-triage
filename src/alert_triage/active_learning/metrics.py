
import numpy


def class_averaged_accuracy_score(targets, preds):
    labels = numpy.unique(targets)
    average = 0.0
    for label in labels:
        indices = targets == label
        mistakes = numpy.sum((targets[indices] == preds[indices]).astype(float))
        average += mistakes / float(numpy.sum(indices.astype(float)))
    return average / float(len(labels))

def class_averaged_mae_score(targets, preds):
    labels = numpy.unique(targets)
    average = 0.0
    for label in labels:
        indices = targets == label
        mistakes = numpy.sum(abs(targets[indices] - preds[indices]))
        average += mistakes / float(numpy.sum(indices.astype(float)))
    return average / len(labels)
