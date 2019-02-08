# Implement code for Locality Sensitive Hashing here!
# region Package
from sklearn.neighbors import LSHForest
import operator
import random
import math
import collections
import numpy as np
# endregion

# region Junk Code
#TRying Hash Function by myself
'''k=10
    L=20
    hashIndex=[]
    for i in range(L):
        h = []
        for j in range(k):
            randNum=random.randrange(len(dataMatrix[0]))
            while(randNum in hashIndex):
                randNum = random.randrange(len(dataMatrix[0]))
            h.append(randNum)
        hashIndex.append(h)
    for g in hashIndex:
        hashTable={}
        for data in dataMatrix:
            minHash=0
            for j in range(k):
                hindex=g[j]
                d=data[hindex]
                minHash=min(minHash,d)'''
# endregion

def preProcessing(dataMatrix,trainLabels):
    '''
    :param dataMatrix:
    :param trainLabels:
    :return:hashDetails,hashTableSet
    Author: KD
    Details: This Procedure Preprocess the Train Data points and Hash Points in HashTables based on Random Projections.
             We Build 20 such Hash Tables.
    '''
    hashDetails = {}
    hashTableSet={}
    w=5
    vectorSize=len(dataMatrix[0])
    k=int(vectorSize)
    for i in range(20):
        randomIndexes = random.sample(range(k), k)
        hashTable = {}
        dataIndex=0
        #r = np.random.normal(0, 1, (1,vectorSize))
        r = np.random.normal(0, 1, (1, k))
        b = np.random.uniform(0, w)
        info=[]
        info.append(r)
        info.append(b)
        info.append(w)
        info.append(randomIndexes)
        hashDetails[i]=info
        sign={}
        sign['pos']=0
        sign['neg'] = 0
        for data in dataMatrix:
            reducedData =[data[x] for x in randomIndexes]
            #data=np.reshape(data,len(data),1)
            #p=np.dot(data,np.transpose(r))
            p = np.dot(reducedData, np.transpose(r))
            if(p>=0):
                sign['pos'] += 1
            else:
                sign['neg'] += 1
            #hashIndex = int(np.floor((p/w)+b))
            hashIndex = int(np.floor((p + b) / w))
            if (hashIndex not in hashTable):
                element=[]
                element.append(dataIndex)
                #element.append(trainLabels[dataIndex])
                hashTable[hashIndex]=element
            else:
                hashTable[hashIndex].append(dataIndex)
            dataIndex +=1
        od = collections.OrderedDict(sorted(hashTable.items()))
        hashTableSet[i]=od
    return hashDetails,hashTableSet

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
        randomIndexes=hashParam[i][3]
        hashTB=hashTables[i]
        reducedData = [testVector[x] for x in randomIndexes]
        #data = np.reshape(testVector, len(testVector), 1)
        p = np.dot(reducedData, np.transpose(r))
        #hashIndex = int(np.floor((p / w) + b))
        hashIndex = int(np.floor((p + b)/ w))
        if (hashIndex not in hashTB):
            continue
        dataIndices=hashTB[hashIndex]

        # region Take Majoirity Label from mapped labels
        '''
        neighbourLabel={}
        for j in dataIndices:
            classLabel=trainLabels[j]
            if (classLabel not in neighbourLabel):
                neighbourLabel[classLabel]=1
            else:
                neighbourLabel[classLabel] += 1
        neighbors.append(max(neighbourLabel.items(), key=operator.itemgetter(1))[0])
        '''
        # endregion

        # region Calculate Eucledian Distance with mapped points and take closed distance point
        for j in dataIndices:
            dist = euclideanDistance(testVector, dataMatrix[j], length)
            distance.append((trainLabels[j], dist,j))
        distance.sort(key=operator.itemgetter(1))
        limit=min(k,len(distance))
        for l in range(0, limit):
            neighbors.append(distance[l])
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
    hashDetails,hashTableSet=preProcessing(dataMatrix,trainLabels)
    predictionLevel={}
    correctPrediction = 0
    labelSet=set(trainLabels)
    for k in labelSet:
        predictionLevel[k]=0
    for x in range(len(testMatrix)):
        # Take Majority Of labels in Hash Slot Points
        '''nearestNeighborslabels=queryTestVector(testMatrix[x],dataMatrix,trainLabels,hashDetails,hashTableSet)
        result=majorityLabel(nearestNeighborslabels)'''
        # Take Nearest Neighbour of labels in Hash Slot Points and Take Closest One.
        nearestNeighbors = queryTestVector(testMatrix[x], dataMatrix, trainLabels, hashDetails, hashTableSet)
        result=trainLabels[getCloseLabel(nearestNeighbors)]
        count=0
        if (int(testLabels[x]) == int(result)):
            correctPrediction = correctPrediction + 1
    accuracyOfMyCode = (correctPrediction / len(testMatrix)) * 100.0
    print("Locality Sensitive Hashing:Custom Library Accuracy ", accuracyOfMyCode)
    # endregion

    # region Sklearn Library LSH Code
    accuracy=0
    lshf = LSHForest(random_state=42)
    lshf.fit(dataMatrix)
    LSHForest(min_hash_match=4, n_candidates=50, n_estimators=10,
              n_neighbors=7, radius=1.0, radius_cutoff_ratio=0.9,
              random_state=42)
    distances, indices = lshf.kneighbors(testMatrix, n_neighbors=7)
    for x in range(len(testMatrix)):
        closestNeighbourIndices=indices[x]
        for i in closestNeighbourIndices:
            predictionLevel[trainLabels[i]]=predictionLevel[trainLabels[i]]+1
        predictionLabel=max(predictionLevel.items(), key=operator.itemgetter(1))[0]
        if(predictionLabel==testLabels[x]):
            accuracy += 1
    accuracyOfLibraryCode = 0
    accuracyOfLibraryCode = (accuracy / len(testMatrix)) * 100.0
    print("Locality Sensitive Hashing:Sklearn Library Accuracy ", accuracyOfLibraryCode)
    # endregion
