"""Test RankingMetrics methods.

This contains the unit tests for the RankingMetrics class.

RankingMetricsTests is the actual class that defines the unit tests.

To Do:

    * Handle multi-class problem.

    * Did I write code for something we aren't going to need in an
      attempt to handle lists, tuples, or arrays?

"""

import unittest

import numpy

from alert_triage.evaluation import ranking_metrics

_DEBUG = True

# These are various module-level constants.  Some are used as the true
# values in the various assertion statements.

# _PREDICTIONS represents what the system predicted for the labels.
_PREDICTIONS = [1, 1, 1, 1, 0, 0, 1, 0, 0, 1]
_PREDICTIONS_TUPLE = (1, 1, 1, 1, 0, 0, 1, 0, 0, 1)
_PREDICTIONS_NUMPY_ARRAY = numpy.array((1, 1, 1, 1, 0, 0, 1, 0, 0, 1))
# What the labels actually are.
_GROUND_TRUTH = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
_GROUND_TRUTH_TUPLE = (0, 1, 1, 0, 1, 1, 1, 0, 1, 1)
_GROUND_TRUTH_NUMPY_ARRAY = numpy.array((0, 1, 1, 0, 1, 1, 1, 0, 1, 1))
_GROUND_TRUTH_ONE_CLASS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_REFERENCE_AUC_SCORE = 0.190
_REFERENCE_AUC_SCORE_ONE_CLASS = 1.0


# _SCORED_PREDICTIONS represents what the system predicted for scores.
_SCORED_PREDICTIONS_1 = [1.0, 0.7, 0.6, 0.5, 0.4, 0.0]
_SCORED_PREDICTIONS_TUPLE_1 = (1.0, 0.7, 0.6, 0.5, 0.4, 0.0)
_SCORED_PREDICTIONS_NUMPY_ARRAY_1 = numpy.array((1.0, 0.7, 0.6, 0.5, 0.4, 0.0))
_SCORED_PREDICTIONS_2 = [1.0, 0.9, 0.6, 0.5, 0.2, 0.0]
_SCORED_PREDICTIONS_TUPLE_2 = (1.0, 0.9, 0.6, 0.5, 0.2, 0.0)
_SCORED_PREDICTIONS_NUMPY_ARRAY_2 = numpy.array((1.0, 0.9, 0.6, 0.5, 0.2, 0.0))
# What the labels actually are.
_SCORED_GROUND_TRUTH_1 = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
_SCORED_GROUND_TRUTH_TUPLE_1 = (1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
_SCORED_GROUND_TRUTH_NUMPY_ARRAY_1 = numpy.array((1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
_SCORED_GROUND_TRUTH_2 = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
_SCORED_GROUND_TRUTH_TUPLE_2 = (1.0, 1.0, 0.0, 1.0, 0.0, 0.0)
_SCORED_GROUND_TRUTH_NUMPY_ARRAY_2 = numpy.array((1.0, 1.0, 0.0, 1.0, 0.0, 0.0))
_SCORED_GROUND_TRUTH_ONE_CLASS = [0, 0, 0, 0, 0, 0]
_REFERENCE_SCORED_AUC_SCORE_1 = 0.47
_REFERENCE_SCORED_AUC_SCORE_2 = 0.54
_REFERENCE_SCORED_AUC_SCORE_ONE_CLASS = 1.0


class RankingMetricsTests(unittest.TestCase):

    """Test RankingMetrics methods.

    This class tests the various methods in the RankingMetrics class.

    Methods

    setUp: this method constructs a ranking_metrics instance before
    each test is run.

    tearDown: this method doesn't need to do anything (yet). Note that
    we decided as a team NOT to delete objects in tearDown that were
    created in setUp. Since each test gets its own instance of the
    RankingMetricsTests class, the instance is destroyed after
    tearDown completes.  Any objects created by setUp will be garbage
    collected at this time.

    test_calculate_auc: this calls the calculate_auc method of the
    RankingMetrics class, then compares the calculated score to the
    reference score.

    test_calculate_auc_one_class: this calls the calculate_auc method
    of the RankingMetrics class, then compares the calculated score to
    the reference score.  This is for the specific case ground truth
    is either all negative or all positive instances.

    test_calculate_auc_tuples: this calls the calculate_auc method of
    the RankingMetrics class, then compares the calculated score to
    the reference score.  It is designed to test the case when both
    the system predictions and ground truth are tuples instead of
    lists.

    test_calculate_auc_numpy_arrays: this calls the calculate_auc
    method of the RankingMetrics class, then compares the calculated
    score to the reference score.  It is designed to test the case
    when both the system predictions and ground truth are numpy arrays
    instead of lists.

    """

    def setUp(self):
        """Create an instance of the RankingMetrics class."""
        self._ranking_metrics = ranking_metrics.RankingMetrics()

    def tearDown(self):
        """Don't do anything."""
        pass

    def test_calculate_auc(self):
        """Test the calculate_auc method of the RankingMetrics class.

        This compares the calculated AUC score to the known correct
        value calculated by hand.

        """
        auc_score = self._ranking_metrics.calculate_auc(_PREDICTIONS,
                                                        _GROUND_TRUTH)
        self.assertAlmostEqual(auc_score, _REFERENCE_AUC_SCORE, places=3)

    def test_calculate_scored_auc_1(self):
        """Test the calculate_scored_auc method of the RankingMetrics class.

        This compares the calculated scored AUC to the known correct
        value calculated by hand.

        """
        scored_auc_score = self._ranking_metrics.calculate_scored_auc(
                                                        _SCORED_PREDICTIONS_1,
                                                        _SCORED_GROUND_TRUTH_1)
        self.assertAlmostEqual(scored_auc_score, _REFERENCE_SCORED_AUC_SCORE_1,
                               places=2)

    def test_calculate_scored_auc_2(self):
        """Test the calculate_scored_auc method of the RankingMetrics class.

        This compares the calculated scored AUC to the known correct
        value calculated by hand.

        """
        scored_auc_score = self._ranking_metrics.calculate_scored_auc(
                                                        _SCORED_PREDICTIONS_2,
                                                        _SCORED_GROUND_TRUTH_2)
        self.assertAlmostEqual(scored_auc_score, _REFERENCE_SCORED_AUC_SCORE_2,
                               places=2)

    def test_calculate_auc_one_class(self):
        """Test the calculate_auc method when ground truth is all one
        class.

        This compares the calculated AUC score to the known correct
        value calculated by hand for the case when ground truth is either all
        positive or all negative instances.

        """
        auc_score = self._ranking_metrics.calculate_auc(_PREDICTIONS,
                                                        _GROUND_TRUTH_ONE_CLASS)
        self.assertEqual(auc_score, _REFERENCE_AUC_SCORE_ONE_CLASS)
        
    def test_calculate_scored_auc_one_class(self):
        """Test the calculate_scored_auc method when ground truth is all one
        class.

        This compares the calculated scored AUC to the known correct
        value calculated by hand for the case when ground truth is either all
        positive or all negative instances.

        """
        scored_auc = self._ranking_metrics.calculate_scored_auc(_SCORED_PREDICTIONS_1,
                                                        _SCORED_GROUND_TRUTH_ONE_CLASS)
        self.assertEqual(scored_auc, _REFERENCE_SCORED_AUC_SCORE_ONE_CLASS)

    def test_calculate_auc_tuples(self):
        """Test the calculate_auc method of the RankingMetrics class
        using tuples for the predictions and ground truth.

        This compares the calculated AUC score to the known correct
        value calculated by hand when the system predictions and
        ground truth are tuples.

        """
        auc_score = self._ranking_metrics.calculate_auc(_PREDICTIONS_TUPLE,
                                                        _GROUND_TRUTH_TUPLE)
        self.assertAlmostEqual(auc_score, _REFERENCE_AUC_SCORE, places=3)

    def test_calculate_scored_auc_tuples_1(self):
        """Test the calculate_scored_auc method of the RankingMetrics class
        using tuples for the predictions and ground truth.

        This compares the calculated scored AUC to the known correct
        value calculated by hand when the system predictions and
        ground truth are tuples.

        """
        scored_auc = self._ranking_metrics.calculate_scored_auc(_SCORED_PREDICTIONS_TUPLE_1,
                                                        _SCORED_GROUND_TRUTH_TUPLE_1)
        self.assertAlmostEqual(scored_auc, _REFERENCE_SCORED_AUC_SCORE_1,
                               places=2)

    def test_calculate_scored_auc_tuples_2(self):
        """Test the calculate_scored_auc method of the RankingMetrics class
        using tuples for the predictions and ground truth.

        This compares the calculated scored AUC to the known correct
        value calculated by hand when the system predictions and
        ground truth are tuples.

        """
        scored_auc = self._ranking_metrics.calculate_scored_auc(_SCORED_PREDICTIONS_TUPLE_2,
                                                        _SCORED_GROUND_TRUTH_TUPLE_2)
        self.assertAlmostEqual(scored_auc, _REFERENCE_SCORED_AUC_SCORE_2,
                               places=2)

    def test_calculate_auc_numpy_arrays(self):
        """Test the calculate_auc method of the RankingMetrics class
        using numpy arrays for the predictions and ground truth.

        This compares the calculated AUC score to the known correct
        value calculated by hand when the system predictions and
        ground truth are numpy arrays.

        """
        auc_score = self._ranking_metrics.calculate_auc(
            _PREDICTIONS_NUMPY_ARRAY,
            _GROUND_TRUTH_NUMPY_ARRAY)
        self.assertAlmostEqual(auc_score, _REFERENCE_AUC_SCORE, places=3)

    def test_calculate_scored_auc_numpy_arrays_1(self):
        """Test the calculate_scored_auc method of the RankingMetrics class
        using numpy arrays for the predictions and ground truth.

        This compares the calculated scored AUC to the known correct
        value calculated by hand when the system predictions and
        ground truth are numpy arrays.

        """
        scored_auc = self._ranking_metrics.calculate_scored_auc(
            _SCORED_PREDICTIONS_NUMPY_ARRAY_1,
            _SCORED_GROUND_TRUTH_NUMPY_ARRAY_1)
        self.assertAlmostEqual(scored_auc, _REFERENCE_SCORED_AUC_SCORE_1,
                               places=2)

    def test_calculate_scored_auc_numpy_arrays_2(self):
        """Test the calculate_scored_auc method of the RankingMetrics class
        using numpy arrays for the predictions and ground truth.

        This compares the calculated scored AUC to the known correct
        value calculated by hand when the system predictions and
        ground truth are numpy arrays.

        """
        scored_auc = self._ranking_metrics.calculate_scored_auc(
            _SCORED_PREDICTIONS_NUMPY_ARRAY_2,
            _SCORED_GROUND_TRUTH_NUMPY_ARRAY_2)
        self.assertAlmostEqual(scored_auc, _REFERENCE_SCORED_AUC_SCORE_2,
                               places=2)

if __name__ == "__main__":
    unittest.main()
