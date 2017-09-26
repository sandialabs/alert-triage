"""

This contains the unit tests for the ModelBuilder class.

"""

DEBUG = True

import unittest

from alert_triage.database import database
from alert_triage.model_builder import model_builder

class ModelBuilderTests(unittest.TestCase):

    def setUp(self):
        self._alerts_db = database.Database()
        self._model = model_builder.ModelBuilder(self._alerts_db)

    def tearDown(self):
        del self._model
        self._alerts_db.close()

    # def test_build_tree(self):
    #     self._model.build_model()
    #     assert self._model.scores

    # def test_build_svm(self):
    #     self._model._model = 'svm'
    #     self._model.build_model()
    #     assert self._model.scores

if __name__ == '__main__':
    unittest.main()
