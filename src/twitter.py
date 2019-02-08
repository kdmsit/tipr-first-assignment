#Implement Naive Bayes for Twitter Classifier here!
# region Packages
#This Region contains all imported packages that is used inside the code.
import numpy as np
import sklearn
import lsh
import nn
import collections
import math
# endregion

def classPartition(trainDataSet,trainLabels):
    '''
    :param trainDataSet:
    :param trainLabels:
    :return: ClassPartitionSet-A Dict having data-elements of a particular class grouped together.
    Author: KD
    Details: This Code Partitions the Train Data as per Class Labels
    '''
    ClassPartitionSet={}
    for i in range(len(trainDataSet)):
        dataElement=list(trainDataSet[i])
        dataLabel=trainLabels[i]
        if(dataLabel!=3):
            if(dataLabel not in ClassPartitionSet):
                ClassPartitionSet[dataLabel]=[]
            ClassPartitionSet[dataLabel].append(dataElement)
    return ClassPartitionSet


def priorDensity(trainLabels):
    sampleSize=len(trainLabels)
    priorDict={}
    labelSize={}
    for label in trainLabels:
        if label not in labelSize:
            labelSize[label]=1
        else:
            labelSize[label] += 1
    for classValue, count in labelSize.items():
        priorDict[classValue] = count / sampleSize
    return priorDict

def twitter():
    # region handle older versions of sklearn
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split
    # endregion
    # region Twitter Data Set

    inputFilePath = 'data/twitter/'
    inputFileName = 'twitter.txt'
    out = 'twitter'
    inputLabelFileName = 'twitter_label.txt'
    testFileName = 'twitter_test.txt'
    testLabelFileName = 'twitter_test_label.txt'
    vocabulary = {}
    inputFile=open(inputFilePath + inputFileName, 'r+')
    fileLines=inputFile.readlines()
    wordMatrix=[]
    for line in fileLines:
        line=line.strip()
        sentence = line.split(" ")
        wordMatrix.append(sentence)
    labelMatrix = np.genfromtxt(inputFilePath + inputLabelFileName, delimiter=' ')
    (trainData, testData, trainLabels, testLabels) = train_test_split(wordMatrix,
                                                                      labelMatrix, test_size=0.20, random_state=42)

    #priorDict = priorDensity(trainLabels)
    for sentence in trainData:
        for word in sentence:
            if (word not in vocabulary):
                vocabulary[word] = 1
            else:
                vocabulary[word] = vocabulary[word] + 1
    #orderedDictionary = collections.OrderedDict(sorted(vocabulary.items(), key=lambda x: x[1]))
    updatedVoc = {k: v for k, v in vocabulary.items() if v >3}
    dataMatrix = np.zeros((len(trainData), len(updatedVoc)))
    testMatrix = np.zeros((len(testData), len(updatedVoc)))
    i=0
    for sentence in trainData:
        for word in sentence:
            if (word not in updatedVoc):
                continue
            index = list(updatedVoc.keys()).index(word)
            dataMatrix[i][index] = dataMatrix[i][index] + 1
        i = i + 1
    i=0
    for sentence in testData:
        for word in sentence:
            if (word not in updatedVoc):
                continue
            index = list(updatedVoc.keys()).index(word)
            testMatrix[i][index] = testMatrix[i][index] + 1
        i = i + 1

    classPartitionSet = classPartition(dataMatrix, trainLabels)
    priorProbSet = {}
    for classValue, features in classPartitionSet.items():
        priorProbSet[classValue] = len(features) / len(trainData)
    # endregion

    # region Bayes On Twitter
    '''classDensitiesOfWords = {}
    for classValue, features in classPartitionSet.items():
        VocSize = len(features[0])
        matrix = np.matrix(features)
        sumofallElements = np.sum(matrix)
        deno = VocSize + sumofallElements
        columnsum = matrix.sum(axis=0).tolist()
        densities = []
        for i in range(0, len(columnsum[0])):
            num = columnsum[0][i] + 1
            prob = (num) / deno
            densities.append(prob)
        classDensitiesOfWords[classValue] = densities

    predictions = []
    accuracy = 0
    for i in range(len(testMatrix)):
        testVector=np.zeros((len(updatedVoc)))
        #testVector = list(testData[0])
        for word in testData[i]:
            if (word not in updatedVoc):
                continue
            index = list(updatedVoc.keys()).index(word)
            testVector[index] = testVector[index] + 1
        probabilities = {}
        for classValue, densities in classDensitiesOfWords.items():
            prob = priorProbSet[classValue]
            for i in range(0, len(testVector)):
                if (testVector[i] != 0):
                    if (testVector[i] == 1):
                        prob = prob * densities[i]
                    else:
                        prob = prob * np.power(densities[i], testVector[i])
            probabilities[classValue] = prob
        predictions.append(max(probabilities, key=probabilities.get))
    for x in range(len(testData)):
        if testLabels[x] == predictions[x]:
            accuracy += 1
    accuracyOfMyCode = (accuracy / len(testData)) * 100.0
    print("Accuracy of My Code ", accuracyOfMyCode)'''
    # endregion

    lsh.lsh(dataMatrix,trainLabels,testMatrix,testLabels)
    #nn.nearestNeighbour(dataMatrix,trainLabels,testMatrix,testLabels)
