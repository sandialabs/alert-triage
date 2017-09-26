"""Define ranking metrics.

This is where we're keeping our project-specific implementations of
various metrics for ranking algorithms.

RankingMetricsException: this is the exception type that should be
raised when exceptions or errors occur. This isn't currently used.

RankingMetrics: each method in this class implements a specific
ranking metric.

"""

import copy

import numpy

_DEBUG = True

class RankingMetricsException(Exception):

    """Exception type for the RankingMetrics class."""

    pass


class RankingMetrics(object):

    """Define ranking metrics.

    The methods in this class implement various metrics for ranking
    algorithms.

    calculate_auc(): this method implements the AUC metric as defined
    in the Scored AUC paper.

    """

    def __init__(self):
        """Create an instance of the RankingMetrics class."""
        pass

    def _cmp(self, x, y):
        """Define compare function for sort."""
        if x[1] < y[1]:
            return -1
        elif x[1] == y[1]:
            return 0
        else:
            return 1

    def to_list(self, data):
        """Convert input tuple, list, or numpy array to list.

        arguments:

            input: tuple, list, or numpy array to convert

        """

        # Take tuple, list, or numpy array and create a list copy
        if type(data) is tuple:
            input_copy = list(data)
        elif type(data) is list:
            input_copy = copy.copy(data)
        elif type(data) is numpy.ndarray:
            input_copy = list(data)

        return input_copy

    def get_pos_and_neg(self, ground_truth):
        """Return number of positive and negative ground truth instances.

        arguments:

            ground_truth: list of ground truth values where 1 = positive
            and 0 = negative.

        """

        # Calculate number of pos (m) and neg (n) instances.
        m = 0
        n = 0
        for truth in ground_truth:
            if truth == 1:
                m += 1
            else:
                n += 1
        return m, n


    def calculate_scored_auc(self, predictions, ground_truth):
        """Calculate the scored AUC as defined in the Scored AUC paper.

        arguments:

            predictions: this is a tuple, list, or numpy array
            containing the scores predicted by the system.

            ground_truth: this is a tuple, list, or numpy array
            of the known, correct labels.

        """

        # Take tuple, list, or numpy array and create a list copy
        predictions_copy = self.to_list(predictions)
        ground_truth_copy = self.to_list(ground_truth)

        # Create a version of the predictions with the original indices
        for i, prediction in enumerate(predictions_copy):
            predictions_copy[i] = (i, prediction)

        # Sort predictions in descending order
        predictions_copy.sort(cmp=self._cmp, reverse=True)

        tie_value = -1
        aoc = 0
        auc = 0
        r = 0
        c = 0
        for index, prediction in predictions_copy:
            if ground_truth_copy[index] == 1:
                if prediction == tie_value:
                    tie_count += 1
                else:
                    tie_count = 1
                    tie_value = prediction
                c += prediction
                aoc += r
            else:
                r += prediction
                auc += c

        # m = num actual positive instances,
        # n = num actual negative instances
        m, n = self.get_pos_and_neg(ground_truth_copy)

        # Check for divide by zero
        if m == 0 or n == 0:
            return 1.0

        # Weighted average of negative scores
        avg_neg_score = float(m*r - aoc)/(m*n)

        # Weighted average of positive scores
        avg_pos_score = float(auc)/(m*n)

        # Return scored AUC
        return avg_pos_score - avg_neg_score



    def calculate_auc(self, predictions, ground_truth):
        """Calculate the AUC score as defined in the Scored AUC paper.

        arguments:

            predictions: this is a tuple, list, or numpy array
            containing the labels predicted by the system.

            ground_truth: this is a tuple, list, or numpy array of the
            known, correct labels.

        """
        # Take tuple, list, or numpy array and create a list copy.
        predictions_copy = self.to_list(predictions)
        ground_truth_copy = self.to_list(ground_truth)

        # Sort the predicted labels in descending order.

        # First, we need to create a version of the predictions with
        # the original indices.
        for i, prediction in enumerate(predictions_copy):
            predictions_copy[i] = (i, prediction)

        predictions_copy.sort(cmp=self._cmp, reverse=True)

        tie_value = -1
        auc = 0
        c = 0
        for index, prediction in predictions_copy:
            if ground_truth_copy[index] == 1:
                if prediction == tie_value:
                    tie_count += 1
                else:
                    # tie_count should be initialized to 1 here the
                    # first time through the loop.  The first
                    # prediction should never match -1.  Also, note
                    # that tie_count needs to be initialized to 1, not
                    # 0.  This is because the very first positive
                    # instance we run into that is not tied with the
                    # positive instance before it needs to count for
                    # 1.
                    tie_count = 1
                    tie_value = prediction
                c += 1
            else:
                if prediction == tie_value:
                    auc += c - tie_count
                else:
                    auc += c

        # m = num actual positive instances
        # n = num actual negative instances
        m, n = self.get_pos_and_neg(ground_truth_copy)


        # If you have either all negative or all positive instances in
        # ground truth, you'll end up dividing by 0 unless we do this.
        # When we are completely missing instances from one of the
        # classes, the algorithm should just return 1.0 since by
        # default the system ranking is perfect, no matter what it is.
        if m == 0 or n == 0:
            return 1.0

        return float(auc) / (m * n)
