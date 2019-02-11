# Implement code for Locality Sensitive Hashing here!
# region Package
from sklearn.neighbors import LSHForest
import operator
import random
import math
import collections
import numpy as np
from sklearn.metrics import f1_score
# endregion


def preProcessing(dataMatrix, trainLabels):
    '''
   :param dataMatrix:
   :param trainLabels:
   :return:hashDetails,hashTableSet
   Author: KD
   Details: This Procedure Preprocess the Train Data points and Hash Points in HashTables based on Random Projections.
            We Build 20 such Hash Tables.
   '''
    hashDetails = {}
    hashTableSet = {}
    w = 4
    vectorSize = len(dataMatrix[0])
    for i in range(20):
        hashTable = {}
        dataIndex = 0
        r = np.random.normal(0, 1, (1, vectorSize))
        b = np.random.uniform(0, w)
        info = []
        info.append(r)
        info.append(b)
        info.append(w)
        hashDetails[i] = info
        for data in dataMatrix:
            data = np.reshape(data, len(data), 1)
            p = np.dot(data, np.transpose(r))
            #hashIndex = int(np.floor((p / w) + b))
            hashIndex = int(np.ceil((p + b)/w))
            if (hashIndex not in hashTable):
                element = []
                element.append(dataIndex)
                # element.append(trainLabels[dataIndex])
                hashTable[hashIndex] = element
            else:
                hashTable[hashIndex].append(dataIndex)
            dataIndex += 1
        od = collections.OrderedDict(sorted(hashTable.items()))
        hashTableSet[i] = od
    return hashDetails, hashTableSet

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

def queryTestVector(testVector,dataMatrix,trainLabels,hashDetails,hashTableSet):
    '''
    :param testVector:
    :param dataMatrix:
    :param trainLabels:
    :param hashDetails:
    :param hashTableSet:
    :return: neighbors :
    Author: KD
    Details: This Procedure maps the testVector(Query Point) into each HashTable based on Random Projection of same
            Random Vector which we used in prepocessing.Then it take majority labels of the points mapped in that hash
            slot.It returns all those labels from each hash table in neighbours list.
    '''
    L=len(hashDetails)
    neighbors = []
    k=5
    length = len(testVector)
    hashParam=list(hashDetails.values())
    hashTables=list(hashTableSet.values())
    for i in range(L):
        limit=k
        distance = []
        r=hashParam[i][0]
        b=hashParam[i][1]
        w=hashParam[i][2]
        hashTB=hashTables[i]
        data = np.reshape(testVector, len(testVector), 1)
        p = np.dot(data, np.transpose(r))
        #hashIndex = int(np.floor((p / w) + b))
        hashIndex = int(np.ceil((p + b) / w))
        if (hashIndex not in hashTB):
            continue
        dataIndices=hashTB[hashIndex]

        # region Take Majoirity Label from mapped labels
        neighbourLabel={}
        for j in dataIndices:
            classLabel=trainLabels[j]
            if (classLabel not in neighbourLabel):
                neighbourLabel[classLabel]=1
            else:
                neighbourLabel[classLabel] += 1
        neighbors.append(max(neighbourLabel.items(), key=operator.itemgetter(1))[0])
        # endregion

    return neighbors

def getCloseLabel(neighbors):
    score={}
    for i in range(len(neighbors)):
        dataIndex=neighbors[i][2]
        if (dataIndex not in score):
            score[dataIndex]=1
        else:
            score[dataIndex] += 1
    score_Up = sorted(score.items(), key=operator.itemgetter(1),reverse=True)
    return score_Up[0][0]

def majorityLabel(nearestNeighborslabels):
    '''
        :param nearestNeighborslabels:
        :return: Majority Of all Class Labels.
        Author: KD
        Details:This methos will find majority of all Nearest Labels of a test vector and return it as its predicted Class Label.
        '''
    labelCount = {}
    for labels in nearestNeighborslabels:
        if (labels not in labelCount):
            labelCount[labels] = 1
        else:
            labelCount[labels] += 1
    return max(labelCount.items(), key=operator.itemgetter(1))[0]



def lsh(dataMatrix,trainLabels,testMatrix,testLabels):

    # region Custome LSH Code
    print("Executing Custom LSH Classifier")
    hashDetails,hashTableSet=preProcessing(dataMatrix,trainLabels)
    predictionLevel = {}
    predictions = []
    correctPrediction = 0
    labelSet=set(trainLabels)
    for k in labelSet:
        predictionLevel[k]=0
    for x in range(len(testMatrix)):
        # Take Majority Of labels in Hash Slot Points
        nearestNeighborslabels=queryTestVector(testMatrix[x],dataMatrix,trainLabels,hashDetails,hashTableSet)
        result=majorityLabel(nearestNeighborslabels)
        predictions.append(result)
        if (int(testLabels[x]) == int(result)):
            correctPrediction = correctPrediction + 1
    accuracyOfMyCode = (correctPrediction / len(testMatrix)) * 100.0
    f1_score_macro = f1_score(testLabels, predictions, average='macro')
    f1_score_micro = f1_score(testLabels, predictions, average='micro')
    return accuracyOfMyCode, f1_score_macro, f1_score_micro
    # endregion



