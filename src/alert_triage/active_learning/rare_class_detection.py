# RARE CLASS DETECTION
#
# Create two 2D Guassians (one majority class, one minorty class)
# Use rare class detection to find minority class instances

# libraries
from scipy.spatial import KDTree
import numpy as np
import random

# seed RNG for repeatability
random.seed(12345)


def main():
    # setup majority class parameters
    majority_x_center = 0.0
    majority_x_stddev = 1.0
    majority_y_center = 0.0
    majority_y_stddev = 1.0
    majority_size = 1000
    majority_x = [random.normalvariate(majority_x_center, majority_x_stddev) for
                  _ in xrange(majority_size)]
    majority_y = [random.normalvariate(majority_y_center, majority_y_stddev) for
                  _ in xrange(majority_size)]
    majority_data = [{'class':'0', 'x':x, 'y':y} for x, y in
                     zip(majority_x, majority_y)]

    # setup minority class parameters
    minority_x_center = 2.0
    minority_x_stddev = 0.1
    minority_y_center = 2.0
    minority_y_stddev = 0.1
    minority_size = 25
    minority_x = [random.normalvariate(minority_x_center, minority_x_stddev) for
                  _ in xrange(minority_size)]
    minority_y = [random.normalvariate(minority_y_center, minority_y_stddev) for
                  _ in xrange(minority_size)]
    minority_data = [{'class':'1', 'x':x, 'y':y} for x, y in
                     zip(minority_x, minority_y)]

    # combine data
    x = majority_x + minority_x
    y = majority_y + minority_y
    data = majority_data + minority_data

    # create KD-Tree (k = query neighborhood)
    t = KDTree(zip(x, y))
    k = minority_size

    # get indices of knn for each instance as well as the distance to the kth
    # nearest neighbor
    for datum in data:
        datum['indices'], datum['knn_dist'] = get_knn_indices_and_dist(t, datum,
                                                                       k)

    # set the initial radius as the smallest distance to a kth nearest neighbor
    # over all instances
    r = get_radius(data)

    # iteration
    i = 1

    # keep querying until all data has been queried (you will want to ctrl-c
    # before this finishes)
    while len(data) > 0:
        # get number of neighbors within "radius" distance as well as the
        # indices of those neighbors (radius grows with each iteration)
        for datum in data:
            datum['n'], datum['neighbor_indices'] = get_neighbors_within_radius(
                                                        t, datum, (1+i/20.0)*r)

        # for each datum, calculate the max difference between its 'n' value and
        # each of its neighbors within the radius neighborhood (trying to find
        # the instances whose neighborhood is most unlike its neighbors to
        # determine degree of outlierness)
        for datum in data:
            datum['max_n_diff'] = max(map(lambda index: datum['n'] - data[index]['n'],
                                      datum['neighbor_indices']))

        # sort the data based on the max difference between n and its neighbors'
        # n from greatest to least
        sorted_data = sorted(data, key=lambda d: d['max_n_diff'], reverse=True)

        # print to screen the class of the instance selected for query (1 = rare
        # class, 0 = common class)
        print sorted_data.pop(0)['class']
        i += 1


# return the number of instances within "r" of instance as well as the indices
# of those neighbors
def get_neighbors_within_radius(kdtree, datum, r):
    indices = kdtree.query_ball_point([datum['x'], datum['y']], r)
    return len(indices), indices


# return the minimum distance to a kth nearest neighbor out of all instances
def get_radius(data):
    return min(map(lambda d: d['knn_dist'], data))


# return indices of knn as well as the distance to the kth nearest neighbor
def get_knn_indices_and_dist(kdtree, datum, k):
    dists, indices = kdtree.query(np.array([datum['x'], datum['y']]), k=k)
    return indices, dists[-1]


# ain't nuthin but a main thang
if __name__ == '__main__':
    main()
