"""Test NearRealtimePipeline methods.

This contains the unit tests for the BatchPipeline class.

"""

import unittest

from alert_triage.near_realtime_pipeline import near_realtime_pipeline

_DEBUG = False
_STOP_TIME = 40. * 60.

class NearRealtimeTests(unittest.TestCase):

    def setUp(self):
        self._near_realtime_pipeline = near_realtime_pipeline.NearRealtimePipeline(testing=True)

    def tearDown(self):
        pass

    def test_batch_pipeline(self):
        self._near_realtime_pipeline.run_pipeline(_STOP_TIME)

if __name__ == '__main__':
    unittest.main()
