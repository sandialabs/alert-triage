"""

This contains the unit tests for the Database class.

"""


import unittest

from alert_triage.database import database
from alert_triage.feature_extraction import label_extraction

_ALERTS_COUNT = 492781
_ALERTS_LIMIT_COUNT = 10
_LABELS_COUNT = 492781


class DatabaseTests(unittest.TestCase):

    def setUp(self):
        """Create an instance of the Database class"""
        print "Initializing scot3"
        self.db = database.Database(database="scot3")
        print "Finished initializing"

    def tearDown(self):
        """Delete the instance of the Database class"""
        self.db.delete_collection(collection='le')
        self.db.delete_collection(collection='rfe')

        self.db.close()

    def test_labeled_alert_query(self):
	"""This is empty but needs to run"""
	print database.labeled_alert_query(self.db, limit=2)

    def test_find(self):
        """Test the find method of the Database class

        Check that the number of number of alerts found matches the expected
        expected value.

        """
        print "Testing the ability for db to find"
        cursor = self.db.find({})
        self.assertEqual(cursor.count(), _ALERTS_COUNT)

    def test_find_limit(self):
        """Test the find method of the Database class using a limit

        Note that count() sets with_limit_and_skip. count() will ignore
        skips and limits unless this is manually set to True.

        """
        print "Testing the ability for find limits"
        cursor = self.db.find({}, limit=_ALERTS_LIMIT_COUNT)
        self.assertEqual(cursor.count(with_limit_and_skip=True),
                         _ALERTS_LIMIT_COUNT)

    def test_delete_collection(self):
        """Test the delete_collection method of the Database class

        Extract labels, create a collection of labels, delete it,
        do a find() on the previously deleted collection, then
        check that the cursor returns zero results.

        """
        print "Testing the ability to delete collection"
        le = label_extraction.LabelExtraction(self.db)
        le.extract_labels()
        self.db.write_labels(le.alert_labels, 'le')
        self.db.delete_collection(collection='le')

        cursor = self.db.find(collection='le')
        self.assertEqual(cursor.count(), 0)

    def test_write_labels(self):
        """Test the write_labels method of the Database class

        Perform label extraction, write to a collection in the database,
        check that the number of records in the collection matches the
        number of expected number of extracted labels.

        """
        print "Testing the ability to write labels"
        le = label_extraction.LabelExtraction(self.db)
        le.extract_labels()

        self.db.write_labels(le.alert_labels, 'le')

        cursor = self.db.find(collection='le')
        self.assertEqual(cursor.count(), _LABELS_COUNT)

        self.db.delete_collection(collection='le')

    def test_read_labels(self):
        """Test the read_labels method of the Database class

        Perform label extraction, write to a collection in the database,
        then read the labels back into into a dictionary. Check that the
        dictionary's length matches the expected value.

        """
        print "Testing the ability to read labels"
        le = label_extraction.LabelExtraction(self.db)
        le.extract_labels()

        self.db.write_labels(le.alert_labels, 'le')
        data = self.db.read_labels(collection='le')

        self.assertEqual(len(data), _LABELS_COUNT)

if __name__ == '__main__':
    unittest.main()
