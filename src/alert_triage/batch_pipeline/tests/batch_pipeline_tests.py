"""Test BatchPipeline methods.

This contains the unit tests for the BatchPipeline class.

To Do:

    * What is a good metric for testing batch pipeline? (Dustin)

"""

import unittest

from alert_triage.batch_pipeline import batch_pipeline

_DEBUG = False

class BatchPipelineTests(unittest.TestCase):

    def setUp(self):
        self._batch_pipeline = batch_pipeline.BatchPipeline(testing=True)

    def tearDown(self):
        pass

    def test_batch_pipeline(self):
        self._batch_pipeline.run_pipeline()

if __name__ == '__main__':
    unittest.main()
