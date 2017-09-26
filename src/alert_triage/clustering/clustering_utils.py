

'''
This reads a snapshot of scot data that has already had features extracted.
The file is expected to be csv, with the firs line being the feature names.
There are many empty features.
'''

from time import time
import argparse
import csv
import numpy as np
import collections

np.set_printoptions(threshold=np.inf)

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

'''
    This gets the number of rows and columns from the csv file.
'''
def getDimensions(infile, limit):
    with open(infile, 'rb') as csvfile:
        csvreader = csv.DictReader(csvfile)
        i = 0
        nCols = 0
        for row in csvreader:
            if i < limit:
                i = i + 1
                nCols = len(row)
            else:
                break
        csvfile.close()
        yield i
        yield nCols

def createMatrix(infile, limit):

    nRows, nCols = getDimensions(infile, limit)

    print "Number of rows:", nRows
    print "Number of columns:", nCols

    m = np.zeros(shape=(nRows, nCols))

    with open(infile, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        header = csvreader.next()
        i = 0
        for row in csvreader:
            if i < limit:
                j = 0;
                for value in row:
                    if value != "":
                        m[i,j] = float(value)
                    else: 
                        m[i,j] = 0.0
                    j = j + 1
            else:
                break
            i = i + 1
        csvfile.close()
    
    return m

'''
    From scikit-learn example
'''
def clusteringMetrics(estimator, name, data):
    t0 = time()
    y = estimator.fit_predict(data)
    print('% 9s   %.2fs    %i   '
          % (name, (time() - t0), estimator.inertia_))
    counter = collections.Counter(y)
    print "Label counts"
    for key in counter:
        print counter[key],
    print


'''
    Plots the results of clustering by reducing to two dimensions
'''
#def plotPCA(estimator, data):
#    reducedData = PCA(n_components=2).fit_transform(data):
