# Example of rare class detection from the paper:
# Nearest-Neighbor-Based Active Learning for Rare Category Detection
#
# Prints class label queries to screen where:
#     0 = majority class (common class)
#     1 = minority class (rare class)
#
# Author: Jeff Shelburg
# Date:   11/01/2013
#

from scipy.spatial import KDTree
import numpy as np
import random


random.seed(12345)

def main():
    majority_x_center = 0.0
    majority_x_stddev = 1.0
    majority_y_center = 0.0
    majority_y_stddev = 1.0
    majority_size = 1000
    majority_x = [random.normalvariate(majority_x_center, majority_x_stddev) for
                  _ in xrange(majority_size)]
    majority_y = [random.normalvariate(majority_y_center, majority_y_stddev) for
                  _ in xrange(majority_size)]
    majority_data = [{'class': '0', 'x': x, 'y': y} for x, y in
                     zip(majority_x, majority_y)]

    minority_x_center = 2.0
    minority_x_stddev = 0.1
    minority_y_center = 2.0
    minority_y_stddev = 0.1
    minority_size = 25
    minority_x = [random.normalvariate(minority_x_center, minority_x_stddev) for
                  _ in xrange(minority_size)]
    minority_y = [random.normalvariate(minority_y_center, minority_y_stddev) for
                  _ in xrange(minority_size)]
    minority_data = [{'class': '1', 'x': x, 'y': y} for x, y in zip(minority_x,
                     minority_y)]

    x = majority_x + minority_x
    y = majority_y + minority_y
    data = majority_data + minority_data

    t = KDTree(zip(x, y))
    k = minority_size

    for datum in data:
        datum['indices'], datum['knn_dist'] = get_knn_indices_and_dist(t,
                                                                       datum, k)

    r = get_radius(data)
    found_classes = {}
    i = 1

    while len(data) > 0:
        for datum in data:
            datum['n'], datum['neighbor_indices'] = get_neighbors_within_radius(t, datum, (1+i/20.0)*r)

        for datum in data:
            datum['max_n_diff'] = max(map(lambda index: datum['n'] -
                                      data[index]['n'],
                                      datum['neighbor_indices']))

        sorted_data = sorted(data, key=lambda d: d['max_n_diff'], reverse=True)
        found_class = sorted_data.pop(0)['class']
        found_classes[found_class] = True
        print found_class
        i += 1

def get_neighbors_within_radius(kdtree, datum, r):
    indices = kdtree.query_ball_point([datum['x'], datum['y']], r)
    return len(indices), indices

def get_radius(data):
    return min(map(lambda d: d['knn_dist'], data))

def get_knn_indices_and_dist(kdtree, datum, k):
    dists, indices = kdtree.query(np.array([datum['x'], datum['y']]), k=k)
    return indices, dists[-1]

if __name__ == '__main__':
    main()
