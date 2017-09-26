# QUERY BY COMMITTEE
# "If the committee can't agree on your class, prepared to be queried"

# without libraries, Python is nothing
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import math
import random
import time

initialTrainingPercentage = 0.03   # percentage of dataset used to train initial model
datasetSize = 200                  # size of input dataset
numTrees = 10                      # number of base decision trees to use in random forest
data_dir = "./datasets/"                # directory where datasets are stored
datasetIndex = 4                   # index of dataset to use in directory
dataFile = data_dir+"data_"+str(datasetIndex)+".csv"         # input features
targetsFile = data_dir+"targets_"+str(datasetIndex)+".csv"   # respective classes
queryDir = "./queries/"            # directory to save intermediate query snapshots
numQueries = 20                    # number of query iterations
digits = len(str(numQueries))      # used to pad filename string with zeros so image names are listed in order

def main():
    # get our dataset
    originalData = readData()
    originalTargets = readTargets()

    # split data and targets into (1) a set that will be used to train the
    # initial model; and (2) a set that will be used for active learning
    trainingData, trainingTargets, data, targets = splitDataAndTargets(
                                                                originalData,
                                                                originalTargets)

    # train initial model
    classifier = getClassifier()
    classifier.fit(trainingData, trainingTargets)

    # let's take a look at that accuracy
    printAccuracy(classifier, originalData, originalTargets)

    # let's keep track of model accuracies from previous iterations so we can
    # graph it
    iteration = 0
    accuracies = [accuracy_score(targets, classifier.predict(data))]

    # iterate over query iterations
    for _ in xrange(numQueries):
        # let's keep track of those iterations
        iteration += 1
        print "Iteration: " + str(iteration)

        # perform a query
        queryData, queryTarget, data, targets = query(classifier, data, targets)
        print "Got query"

        # data used to train the model thus far
        trainingData0 = [datum for datum, target in
                         zip(trainingData, trainingTargets) if target == 0.0]
        trainingData1 = [datum for datum, target in
                         zip(trainingData, trainingTargets) if target == 1.0]

        # unlabeled data
        data0 = [datum for datum, target in zip(data, targets) if target == 0.0]
        data1 = [datum for datum, target in zip(data, targets) if target == 1.0]

        # let's plot these points
        fig = plt.figure(figsize=(16, 6.5))
        ax1 = fig.add_subplot(121)
        grayscale = '0.85'

        # plot unlabaled data as gray (and its respective shape)
        ax1.plot(map(lambda x: x[0], data0), map(lambda x: x[1], data0),
                 marker='^', markerfacecolor=grayscale,
                 markeredgecolor=grayscale, linestyle='None')
        ax1.plot(map(lambda x: x[0], data1), map(lambda x: x[1], data1),
                 marker='o', markerfacecolor=grayscale,
                 markeredgecolor=grayscale, linestyle='None')

        # plot training data as colored shapes
        ax1.plot(map(lambda x: x[0], trainingData0),
                 map(lambda x: x[1], trainingData0), 'r^')
        ax1.plot(map(lambda x: x[0], trainingData1),
                 map(lambda x: x[1], trainingData1), 'co')

        # plot the queried points for this iteration
        ax1.plot([queryData[0]], [queryData[1]], 'g^' if
                 queryTarget == 0.0 else 'go', markersize=15)

        # title, yo
        plt.title("Dataset %s - Query %s/%s", (str(datasetIndex),
                                               str(iteration),
                                               str(int(numQueries))))

        # add queried points to training points and retrain our model
        trainingData.append(queryData)
        trainingTargets.append(queryTarget)
        classifier = getClassifier()
        classifier.fit(trainingData, trainingTargets)

        # print this iteration's accuracy and calculate new one
        printAccuracy(classifier, originalData, originalTargets)
        accuracy = accuracy_score(targets, classifier.predict(data))
        accuracies.append(accuracy)

        # finishing touches for the graph
        ax2 = fig.add_subplot(122)
        ax2.set_xlim([0, numQueries])
        ax2.set_ylim([0.0, 1.0])
        ax2.plot(range(len(accuracies)), accuracies)
        plt.xlabel("Number of Queries")
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy")

        # pad the iteration number in the image file so they all stay in order
        iterationDigits = len(str(iteration))
        if iterationDigits == digits:
            paddingString = ''
        else:
            paddingString = '0' * (digits-iterationDigits)

        # let's save and close this
        fig.savefig("%squery_%s_%s%s.png" % (queryDir, str(datasetIndex),
                                             paddingString, str(iteration)))
        plt.close()


# calculate class-average accuracy (for two classes)
def getClassAvgAccuracy(classifier, data, targets):
    predicted = classifier.predict(data)
    class0correct = 0.0
    class0wrong = 0.0
    class1correct = 0.0
    class1wrong = 0.0
    for prediction, actual in zip(predicted, targets):
        if actual == 0.0:
            if prediction == 0.0:
                class0correct += 1.0
            else:
                class0wrong += 1.0
        else:
            if prediction == 1.0:
                class1correct += 1.0
            else:
                class1wrong += 1.0
    return ((class0correct/(class0correct+class0wrong)) + (class1correct/(class1correct+class1wrong)))/2.0


# return random forest classifier
def getClassifier():
    return RandomForestClassifier(n_estimators=numTrees, criterion="entropy",
                                  random_state=random.randint(1, 99999),
                                  n_jobs=-1)


# perform query
def query(classifier, data, targets):
    # record initial time
    start0 = time.time()

    # let's keep track of the best query
    bestValue = float("inf")
    bestIndex = -1
    iteration = 0
    queryTime = 0

    # let's look through each datum to see if we should query it
    for i in xrange(len(data)):
        # increment our iteration
        iteration += 1
        print "Query Iter: " + str(iteration)

        # get class probabilities and add query time to overall query time
        start1 = time.time()
        classProbs = classifier.predict_proba(data[i])[0]
        queryTime += time.time() - start1

        # record best query values and its index in the data
        value = 0.0
        if 0.0 in classProbs:
            value = 1.0
        else:
            for classProb in classProbs:
                value += classProb * math.log(classProb)
        if value < bestValue:
            bestValue = value
            bestIndex = i

    # get best query
    queryData = data.pop(bestIndex)
    queryTarget = targets.pop(bestIndex)

    # output query time and total time
    print "Query Time: " + str(queryTime)
    print "Total Time: " + str(time.time() - start0)

    return queryData, queryTarget, data, targets


# split data into training set and query set
def splitDataAndTargets(data, targets):
    # randomize data
    combinedData = [[d, t] for d, t in zip(data, targets)]
    random.shuffle(combinedData)

    # get training data
    trainingSetSize = int(len(data)*initialTrainingPercentage)
    combinedTrainingData = combinedData[:trainingSetSize]
    combinedData = combinedData[trainingSetSize:]
    trainingData = [x[0] for x in combinedTrainingData]
    trainingTargets = [x[1] for x in combinedTrainingData]

    # get querying data
    newData = [x[0] for x in combinedData]
    newTargets = [x[1] for x in combinedData]

     # return what we got
    return trainingData, trainingTargets, newData, newTargets


# let's take a look at the accuracy
def printAccuracy(classifier, data, targets):
    predictedTargets = classifier.predict(data)
    print "Accuracy: " + str(accuracy_score(targets, predictedTargets))


# this function is silly
def getData():
    data = getInitialData()


# read data from file
def readData():
    return csvRead(dataFile)


# read target values from file
def readTargets():
    return map(lambda x: x[0], csvRead(targetsFile))


# each element in the returned list is a line; each line is a list of CSV
def csvRead(filename):
    return [[float(x) for x in line.split(',')] for
            line in open(filename, 'r')][:datasetSize]


# ain't no function like a main
if __name__ == '__main__':
    main()
