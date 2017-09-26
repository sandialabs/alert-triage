# Create 2D Guassian dataset

import matplotlib.pyplot as plt
import random

# number of datasets to create
numDatasets = 10

# random number generator seed
rngSeed = 12345

random.seed(rngSeed)

data_dir = "datasets/"         # directory to save datasets
dataFile = "data_"        # data file prefix (X)
targetsFile = "targets_"  # targets file prefix (Y)
graphFile = "graph_"      # png graph file prefix
extension = ".csv"        # extension for data and target files

# decision boundary parameters (Y = AX^2 + BX + C)
aMin = -5
aMax = 5
bMin = -5
bMax = 5
cMin = -1
cMax = 1

# cluster parameters
numPointsMin = 250
numPointsMax = 500


# get decision boundary and Guassian parameters
def getParams():
    # decision boundary
    a = (random.random()*(aMax-aMin)) - aMin
    b = (random.random()*(bMax-bMin)) - bMin
    c = (random.random()*(cMax-cMin)) - cMin

    # Guassian
    centerX = 0.0
    centerY = random.choice([1, -1]) * random.random() + c
    stddevX = random.random() * abs(abs(a)-abs(b))
    stddevY = random.random() * abs(abs(a)-abs(b))

    # number of points in dataset
    numPoints = int((random.random()*(numPointsMax-numPointsMin))+numPointsMin)

    return a, b, c, centerX, centerY, stddevX, stddevY, numPoints

# generate dataset points
def getPoints():
    points = []
    a, b, c, centerX, centerY, stddevX, stddevY, numPoints = getParams()

    # genereate each point
    for point in xrange(numPoints):
        # each point's coordinates
        x = random.gauss(centerX, stddevX)
        y = random.gauss(centerY, stddevY)

        # each point's class (as dictated by quadratic decision boundary)
        c = 1.0 if (y > a*(x**2) + b*x + c) else 0.0

        # add point to list of points
        points.append({'x':x, 'y':y, 'c':c})

    return points, a, b, c

# do work, yo
def main():
    # make "numDatasets" many datasets
    for datasetIndex in xrange(numDatasets):
        # get dataset points
        points, _, _, _ = getPoints()

        # get those file handles!
        fd = open(data_dir+dataFile+str(datasetIndex)+extension, 'w')
        ft = open(data_dir+targetsFile+str(datasetIndex)+extension, 'w')

        # let's write those points to some files
        for point in points:
            fd.write(str(point['x'])+','+str(point['y'])+'\n')
            ft.write(str(point['c'])+'\n')

        # don't forget to close the file handles (or else)
        fd.close()
        ft.close()

        # let's draw some pictures of the 2D datasets
        plt.figure

        # segregate those points by class
        group1 = [point for point in points if point['c'] == 0.0]
        group2 = [point for point in points if point['c'] == 1.0]

        # plot those beautiful points!
        plt.plot(map(lambda x: x['x'], group1), map(lambda x: x['y'], group1),
                 'co')
        plt.plot(map(lambda x: x['x'], group2), map(lambda x: x['y'], group2),
                 'rx')

        # a graph without axes or a title is just a meaningless picture
        plt.xlabel("1st Dimension")
        plt.ylabel("2nd Dimension")
        plt.title("Dataset "+str(datasetIndex))

        # let's save and close this
        plt.savefig(data_dir+graphFile+str(datasetIndex)+".png")
        plt.close()


if __name__ == '__main__':
    main()
