# Implement Nearest Neighbour classifier here!
# region Packages
#This Region contains all imported packages that is used inside the code.
import numpy as np
import math
import lsh
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import operator
# endregion

def euclideanDistance(d1, d2, length):
    '''
    :param d1:
    :param d2:
    :param length:
    :return: distance
    Author: KD
    Details: This Procedure Calculates the Eucledean Distance between two points.
    '''
    distance=0
    D=np.subtract(d1,d2)
    distance = math.sqrt(np.dot(D,np.transpose(D)))
    return distance


def getAllNeighborsWithDistance(trainData,trainLabels,testRow):
    '''
    :param trainData:
    :param trainLabels:
    :param testRow:
    :return:distance
    Author: KD
    Details: This Procedure calculates all
    '''
    distance=[]
    distanceLabe=[]
    length = len(testRow)
    for i in range(len(trainData)):
        dist = euclideanDistance(testRow, trainData[i], length)
        distance.append((trainLabels[i],dist))
    distance.sort(key=operator.itemgetter(1))
    return distance

def getKNearestNeighbor(distance,k):
    '''
    :param distance:
    :param k:
    :return:
    '''
    neighbors = []
    for i in range(0,k):
        neighbors.append(distance[i][0])
    return neighbors

def getCloseLabels(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] is predictions[x]:
            correct=correct+1
    return (correct/float(len(testSet))) * 100.0


def nearestNeighbour(trainData,trainLabels,testData,testLabels):
    k = 7
    predictions = []
    accuracies = []
    correctPrediction = 0
    # region My KNN Classifier Code
    for i in range(0, len(testData)):
        distance = getAllNeighborsWithDistance(trainData, trainLabels, testData[i])
        kNearestNeighbors = getKNearestNeighbor(distance, k)
        result = getCloseLabels(kNearestNeighbors)
        predictions.append(result)
        #print("Predicted: ",result,"Actual:",testLabels[i])
        # predictions.append(result)
        if (testLabels[i] == result):
            correctPrediction = correctPrediction + 1
    accuracyOfMyCode = (correctPrediction / len(testData)) * 100.0
    f1_score_macro=f1_score(testLabels, predictions, average='macro')
    f1_score_micro = f1_score(testLabels, predictions, average='micro')
    print("K-Nearest Neighbour:Custom Code Test Accuracy: ", accuracyOfMyCode)
    print("K-Nearest Neighbour:Custom Code Test F1-Score(Macro): ", f1_score_macro)
    print("K-Nearest Neighbour:Custom Code Test F1-Score(Micro): ", f1_score_micro)
    # endregion

    # region Sklearn Package KNN
    '''
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(trainData, trainLabels)
    predictions = model.predict(testData)
    score = model.score(testData, testLabels)
    accuracies.append(score * 100)
    print("K-Nearest Neighbour:Sklearn Library Code Test Accuracy: ", accuracies)
    print("K-Nearest Neighbour:Sklearn Library Test F1-Score(Macro): ",
          f1_score(testLabels, predictions, average='macro'))
    print("K-Nearest Neighbour:Sklearn Library Test F1-Score(Micro): ",
          f1_score(testLabels, predictions, average='micro'))
    '''
    # endregion

    return accuracyOfMyCode,f1_score_macro,f1_score_micro
