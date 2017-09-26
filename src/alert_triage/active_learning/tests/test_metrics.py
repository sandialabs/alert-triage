"""

This contains the unit tests for the .

"""


import unittest

import numpy

from alert_triage.active_learning import metrics


class MetricTests(unittest.TestCase):

    def test_class_averaged_accuracy(self):
        targets = numpy.array([1, 0, 1, 1, 1, 0, 1, 1])
        preds = numpy.array([1, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual(metrics.class_averaged_accuracy_score(targets, preds),
                         0.5)
        preds = numpy.array([1, 0, 1, 1, 1, 1, 1, 1])
        self.assertEqual(metrics.class_averaged_accuracy_score(targets, preds),
                         0.75)
        preds = numpy.array([1, 0, 1, 1, 1, 0, 1, 1])
        self.assertEqual(metrics.class_averaged_accuracy_score(targets, preds),
                         1.0)
        preds = numpy.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(metrics.class_averaged_accuracy_score(targets, preds),
                         0.5)


if __name__ == '__main__':
    unittest.main()
