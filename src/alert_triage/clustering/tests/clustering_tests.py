import unittest
from alert_triage.clustering import clustering_utils

class ClusteringTests(unittest.TestCase):

    def setUp(self):
        self.infile = "./testData.csv"

    def tearDown(self):
        pass

    def testGetDimensions(self):
        # Testing getting under fewer lines than available in the file.
        limit = 1
        nRows, nCols = clustering_utils.getDimensions(self.infile, limit) 
        self.assertEqual(nRows, 1)
        self.assertEqual(nCols, 5)
       
        # Testing getting exactly the number of lines available in the file
        limit = 2 
        nRows, nCols = clustering_utils.getDimensions(self.infile, limit) 
        self.assertEqual(nRows, 2)
        self.assertEqual(nCols, 5)


        # Testing getting more than available.
        limit = 3
        nRow, nCols = clustering_utils.getDimensions(self.infile, limit) 
        self.assertEqual(nRows, 2)
        self.assertEqual(nCols, 5)

    def testCreateMatrix(self):
        limit = 3
        m = clustering_utils.createMatrix(self.infile, limit)
        self.assertEqual(m.shape[0], 2)
        self.assertEqual(m.shape[1], 5)
        self.assertEqual(m[0,0], 1)
        self.assertEqual(m[0,1], 2)
        self.assertEqual(m[0,2], 3.5)
        self.assertEqual(m[0,3], -4)
        self.assertEqual(m[0,4], 0)
        self.assertEqual(m[1,0], 1)
        self.assertEqual(m[1,1], 0) 
        self.assertEqual(m[1,2], 0)
        self.assertEqual(m[1,3], 0)
        self.assertEqual(m[1,4], 0)


if __name__ == '__main__':
    unittest.main()
