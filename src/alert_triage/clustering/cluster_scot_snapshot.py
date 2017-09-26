

'''
This reads a snapshot of scot data that has already had features extracted.
The file is expected to be csv, with the firs line being the feature names.
There are many empty features.
'''

import argparse
import csv
import numpy as np
import collections
from alert_triage.clustering import clustering_utils
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from time import time

def main():
    parser = argparse.ArgumentParser(description='Clustering options')
    parser.add_argument('--method', 
                        choices=['kmeans'],
                        default='kmeans')
    parser.add_argument('--init', 
                        choices=['k-means++', 'random', 'pca'],
                        default='k-means++')
    parser.add_argument('--input',
                        type=str,
                        help='The location of the csv file',
                        required=True)
    parser.add_argument('--limit',
                        type=int,
                        help='The number of lines to process (0 for all)',
                        default=10000)
    parser.add_argument('--nclusters',
                        type=int,
                        default=2,
                        help='The number of cluters to create (if algorithm'
                             ' needs it')



    args = parser.parse_args()
    infile         = args.input
    limit          = args.limit
    nclusters      = args.nclusters
    method         = args.method
    init           = args.init

    X = clustering_utils.createMatrix(infile, limit)
    print "Finished loading matrix"
    
    beg = time()
    if method == 'kmeans':
        print "Clustering method: kmeans"
        if init == 'k-means++':
            print "init method: k-means++"
            clustering_utils.clusteringMetrics(KMeans(n_clusters = nclusters,
                                     init="k-means++", n_init=10),
                                     name="k-means++", data=X)
        if init == 'random':
            print "init method: random"
            clustering_utils.clusteringMetrics(KMeans(n_clusters = nclusters,
                                     init="random", n_init=10),
                                     name="random", data=X)
        if init == 'pca':
            print "init method: PCA"
            pca = PCA(n_components=nclusters).fit(X)
            clustering_utils.clusteringMetrics(KMeans(
                                               init=pca.components_, n_init=1),
                                               name="PCA-based", data=X)
    end = time()
    print "Time (excluding creating the matrix)", end-beg 




main()
