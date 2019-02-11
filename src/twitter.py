#Implement Naive Bayes for Twitter Classifier here!
# region Packages
#This Region contains all imported packages that is used inside the code.
import numpy as np
import sklearn
import lsh
import bayes
import nn
import collections
import projections
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
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

def naiveBayesMultiNomial(trainData,trainLabels,testData,testLabels,updatedVoc):
    '''
    :param trainData: Mulinomial representation of train word matrix.
    :param trainLabels: Train Labels
    :param testData: Mulinomial representation of test word matrix.
    :param testLabels: Test Labels
    :param testdataword: Test Word Matrix
    :param updatedVoc:  Vocabulary Dictionary
    :return:
    '''
    classPartitionSet = classPartition(trainData, trainLabels)
    priorProbSet = {}
    for classValue, features in classPartitionSet.items():
        priorProbSet[classValue] = len(features) / len(trainData)
    classDensitiesOfWords = {}
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
    for i in range(len(testData)):
        testVector=testData[i]
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
        # print(testLabels[x]," ",predictions[x])
        if testLabels[x] == predictions[x]:
            accuracy += 1
    accuracyOfMyCode = (accuracy / len(testData)) * 100.0
    f1_score_macro = f1_score(testLabels, predictions, average='macro')
    f1_score_micro = f1_score(testLabels, predictions, average='micro')
    return accuracyOfMyCode, f1_score_macro, f1_score_micro

def priorDensity(trainLabels):
    sampleSize = len(trainLabels)
    priorDict = {}
    labelSize = {}
    for label in trainLabels:
        if label not in labelSize:
            labelSize[label] = 1
        else:
            labelSize[label] += 1
    for classValue, count in labelSize.items():
        priorDict[classValue] = count / sampleSize
    return priorDict

def plotData(xVal,yVal,xLabel,yLabel,title,fileName,yticky):
    max = 0
    index = 0
    for i in range(len(yVal)):
        if (yVal[i] > max):
            max = yVal[i]
            index = xVal[i]
    text = "Maximum "
    plt.bar(xVal, yVal)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks(xVal)
    plt.yticks(yticky)
    for i in range(len(yVal)):
        plt.text(xVal[i], yVal[i], str(round(yVal[i], 2)))
    plt.text(index, max / 2, text)
    # plt.xticks(accuracyList,fontsize=5, rotation=30)
    # plt.xticks(index, accuracyList, fontsize=5, rotation=30)
    plt.title(title)
    plt.savefig(fileName)
    plt.close()

def twitter(testMatrix,testlabelMatrix):
    vocabulary = {}
    Accuracy = {}
    F1ScoreMacro = {}
    F1ScoreMicro = {}
    # region handle older versions of sklearn
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split
    # endregion
    # region Twitter Data Set

    outputFileName = "Twitter_stat.txt"
    f = open(outputFileName, "w")
    print("Processing Twitter DataSet.......")
    f.write("Processing Twitter DataSet.......")
    f.write("\n")

    # region Input File INformation
    inputFilePath = '../data/twitter/'
    inputFileName = 'twitter.txt'
    out = 'twitter'
    inputLabelFileName = 'twitter_label.txt'
    labelMatrix = np.genfromtxt(inputFilePath + inputLabelFileName, delimiter=' ')
    # endregion

    # region Generate Word Matrix from txt file

    inputFile=open(inputFilePath + inputFileName, 'r+')
    fileLines=inputFile.readlines()
    wordMatrix=[]
    for line in fileLines:
        line=line.strip()
        sentence = line.split(" ")
        wordMatrix.append(sentence)
    # endregion

    # region Original Dimension Analysis KNN/Bayes/LSH

    trainData=wordMatrix
    trainLabels=labelMatrix
    testData=testMatrix
    testLabels=testlabelMatrix

    # region Generate Vocubulary Dictionary
    for sentence in trainData:
        for word in sentence:
            if (word not in vocabulary):
                vocabulary[word] = 1
            else:
                vocabulary[word] = vocabulary[word] + 1
    updatedVoc = {k: v for k, v in vocabulary.items() if v >2}
    # endregion

    # region Convert Word Matrix into MultiNomial Form
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
    # endregion


    noOfColumns = len(dataMatrix[0])

    # region K-Nearest Neighbour Classifier
    print("Custom K-NN Classifier Statistcs:")
    print("K-Value Taken:5")
    f.write("Custom K-NN Classifier Statistcs:")
    f.write("\n")
    f.write("K-Value Taken:5")
    f.write("\n")
    accuracyNN, f1_score_macroNN, f1_score_microNN = nn.nearestNeighbour(dataMatrix, trainLabels, testMatrix,testLabels)
    print("KNN Test Accuracy ", str(accuracyNN))
    print("KNN Test Macro F1-score ", str(f1_score_macroNN))
    print("KNN Test Micro F1-score ", str(f1_score_microNN))
    f.write("\n")
    f.write("KNN Test Accuracy " + str(accuracyNN))
    f.write("\n")
    f.write("KNN Test Macro F1-score " + str(f1_score_macroNN))
    f.write("\n")
    f.write("KNN Test Micro F1-score " + str(f1_score_microNN))
    f.write("\n")
    f.write("\n")
    # endregion

    # region Bayes Classifier
    print("Custom Bayes Classifier Statistcs:")
    f.write("Custom Bayes Classifier Statistcs:")
    f.write("\n")
    accuracyBayes, f1_score_macroBayes, f1_score_microBayes = naiveBayesMultiNomial(dataMatrix, trainLabels, testMatrix,testLabels,updatedVoc)
    print("Bayes Test Accuracy ", str(accuracyBayes))
    print("Bayes Test Macro F1-score ", str(f1_score_macroBayes))
    print("Bayes Test Micro F1-score ", str(f1_score_microBayes))
    f.write("Original Dimension :" + str(noOfColumns))
    f.write("\n")
    f.write("Bayes Test Accuracy " + str(accuracyBayes))
    f.write("\n")
    f.write("Bayes Test Macro F1-score " + str(f1_score_macroBayes))
    f.write("\n")
    f.write("Bayes Test Micro F1-score " + str(f1_score_microBayes))
    f.write("\n")
    f.write("\n")
    # endregion

    # region LSH Custom Code
    print("Custom LSH Classifier Statistcs:")
    f.write("Custom LSH Classifier Statistcs:")
    f.write("\n")
    f.write("\n")
    accuracyLSH, f1_score_macroLSH, f1_score_microLSH = lsh.lsh(dataMatrix, trainLabels, testMatrix, testLabels)
    print("LSH Test Accuracy ", str(accuracyLSH))
    print("LSH Test Macro F1-score ", str(f1_score_macroLSH))
    print("LSH Test Micro F1-score ", str(f1_score_microLSH))
    f.write("Original Dimension :" + str(noOfColumns))
    f.write("\n")
    f.write("LSH Test Accuracy " + str(accuracyLSH))
    f.write("\n")
    f.write("LSH Test Macro F1-score " + str(f1_score_macroLSH))
    f.write("\n")
    f.write("LSH Test Micro F1-score " + str(f1_score_microLSH))
    f.write("\n")
    f.write("\n")
    # endregion


    # endregion

    # region Random Projection
    '''
    K = 2
    while (K <= int(noOfColumns / 2)):

        Accuracy[K] = 0
        F1ScoreMacro[K] = 0
        F1ScoreMicro[K] = 0
        randomDataMatrix,randomTestMatrix = projections.randomProjectionTwitter(dataMatrix,testMatrix, noOfColumns, K, inputFilePath, out)
        #randomTestMatrix = projections.randomProjection(testMatrix, noOfColumns, K, inputFilePath, out)

        #accuracyOfMyCode, f1_score_macro, f1_score_micro = nn.nearestNeighbour(randomDataMatrix, trainLabels,randomTestMatrix,testLabels)

        accuracyOfMyCode, f1_score_macro, f1_score_micro = naiveBayesMultiNomial(randomDataMatrix, trainLabels,randomTestMatrix, testLabels,updatedVoc)

        print("Reduced Dimension :", K)
        f.write("Reduced Dimension :" + str(K))
        f.write("\n")
        print("Test Accuracy ", str(accuracyOfMyCode))
        print("Test Macro F1-score ", str(f1_score_macro))
        print("Test Micro F1-score ", str(f1_score_micro))
        f.write("Test Accuracy " + str(accuracyOfMyCode))
        f.write("\n")
        f.write("Test Macro F1-score " + str(f1_score_macro))
        f.write("\n")
        f.write("Test Micro F1-score " + str(f1_score_micro))
        f.write("\n")
        f.write("\n")
        Accuracy[K] = accuracyOfMyCode
        F1ScoreMacro[K] = f1_score_macro
        F1ScoreMicro[K] = f1_score_micro
        K = K * 2
        '''
    # endregion

    # region Plot and Save Data
    '''
    index = []
    accuracyList = []
    for k, accuracy in Accuracy.items():
        index.append(k)
        accuracyList.append(accuracy)
    #title = "Accuracy:Different D values Of custom K-NN for Twitter Data Set."
    #figname = "custom_K-NN_Accuracy_Twitter.png"
    #title = "Accuracy:Different D values Of Sklearn K-NN for Twitter Data Set."
    #figname = "Sklearn_K-NN_Accuracy_Twitter.png"
    #title = "Accuracy: Different D values Of custom Bayes for Twitter Data Set."
    #figname="custom_Bayes_Accuracy_Twitter.png"
    # title = "Accuracy: Different D values Of Sklearn Bayes for Twitter Data Set."
    # figname="Sklearn_Bayes_Accuracy_Twitter.png"
    plotData(index, accuracyList, 'D-Value', 'Accuracy', title, figname, np.arange(0, 100, step=10))

    index = []
    F1MacroList = []
    for k, f1_score_macro in F1ScoreMacro.items():
        index.append(k)
        F1MacroList.append(f1_score_macro)
    #title = "F1-Score(Macro):Different D values Of custom K-NN for Twitter Data Set."
    #figname = "custom_K-NN_F1-Score(Macro)_Twitter.png"
    #title = "F1-Score(Macro):Different D values Of Sklearn K-NN for Twitter Data Set."
    #figname = "Sklearn_K-NN_F1-Score(Macro)_Twitter.png"
    #title = "F1-Score(Macro): Different D values Of custom Bayes for Twitter Data Set."
    #figname = "custom_Bayes_F1-Score(Macro)_Twitter.png"
    # title = "F1-Score(Macro): Different D values Of Sklearn Bayes for Twitter Data Set."
    # figname = "Sklearn_Bayes_F1-Score(Macro)_Twitter.png"
    plotData(index, F1MacroList, 'D-Value', 'F1-Score(Macro)', title, figname, np.arange(0, 1, step=0.1))

    index = []
    F1MicroList = []
    for k, f1_score_micro in F1ScoreMicro.items():
        index.append(k)
        F1MicroList.append(f1_score_micro)
    #title = "F1-Score(Micro):Different D values Of custom K-NN for Twitter Data Set."
    #figname = "custom_K-NN_F1-Score(Micro)_Twitter.png"
    #title = "F1-Score(Micro):Different D values Of Sklearn K-NN for Twitter Data Set."
    #figname = "Sklearn_K-NN_F1-Score(Micro)_Twitter.png"
    #title = "F1-Score(Micro): Different D values Of custom Bayes for Twitter Data Set."
    #figname = "custom Bayes_F1-Score(Micro)_Twitter.png"
    # title = "F1-Score(Micro): Different D values Of Sklearn Bayes for Twitter Data Set."
    # figname = "Sklearn Bayes_F1-Score(Micro)_Twitter.png"
    plotData(index, F1MicroList, 'D-Value', 'F1-Score(Micro)', title, figname, np.arange(0, 1, step=0.1))
    '''
    # endregion

    # region Task-VII PCA Analysis:Twitter
    '''
    K = 2
    while (K <= int(noOfColumns / 2)):
        pca = PCA(n_components=K).fit(dataMatrix)
        data_reduce = pca.transform(dataMatrix)
        pca2 = PCA(n_components=K).fit(testMatrix)
        test_reduce = pca2.transform(testMatrix)
        #accuracyOfMyCode, f1_score_macro, f1_score_micro = nn.nearestNeighbour(data_reduce, trainLabels, test_reduce,testLabels)
        accuracyOfMyCode, f1_score_macro, f1_score_micro = naiveBayesMultiNomial(data_reduce, trainLabels, test_reduce,testLabels,updatedVoc)
        print("PCA Reduced Dimension :", K)
        print("Test Accuracy ", str(accuracyOfMyCode))
        print("Test Macro F1-score ", str(f1_score_macro))
        print("Test Micro F1-score ", str(f1_score_micro))
        f.write("Reduced Dimension :" + str(K))
        f.write("\n")
        f.write("Test Accuracy " + str(accuracyOfMyCode))
        f.write("\n")
        f.write("Test Macro F1-score " + str(f1_score_macro))
        f.write("\n")
        f.write("Test Micro F1-score " + str(f1_score_micro))
        f.write("\n")
        f.write("\n")
        Accuracy[K] = accuracyOfMyCode
        F1ScoreMacro[K] = f1_score_macro
        F1ScoreMicro[K] = f1_score_micro
        K = K * 2
    index = []
    accuracyList = []
    for k, accuracy in Accuracy.items():
        index.append(k)
        accuracyList.append(accuracy)
    #title = "Accuracy:Different D values Of PCA K-NN for Twitter Data Set."
    #figname = "PCA_K-NN_Accuracy_Twitter.png"
    title = "Accuracy: Different D values Of PCA Bayes for Twitter Data Set."
    figname = "PCA_Bayes_Accuracy_Twitter.png"
    plotData(index, accuracyList, 'D-Value', 'Accuracy', title, figname, np.arange(0, 100, step=10))

    index = []
    F1MacroList = []
    for k, f1_score_macro in F1ScoreMacro.items():
        index.append(k)
        F1MacroList.append(f1_score_macro)
    #title = "F1-Score(Macro):Different D values Of PCA K-NN for Twitter Data Set."
    #figname = "PCA_K-NN_F1-Score(Macro)_Twitter.png"
    title = "F1-Score(Macro): Different D values Of PCA Bayes for Twitter Data Set."
    figname = "PCA_Bayes_F1-Score(Macro)_Twitter.png"
    plotData(index, F1MacroList, 'D-Value', 'F1-Score(Macro)', title, figname, np.arange(0, 1, step=0.1))

    index = []
    F1MicroList = []
    for k, f1_score_micro in F1ScoreMicro.items():
        index.append(k)
        F1MicroList.append(f1_score_micro)
    #title = "F1-Score(Micro):Different D values Of PCA K-NN for Twitter Data Set."
    #figname = "PCA_K-NN_F1-Score(Micro)_Twitter.png"
    title = "F1-Score(Micro): Different D values Of PCA Bayes for Twitter Data Set."
    figname = "PCA Bayes_F1-Score(Micro)_Twitter.png"
    plotData(index, F1MicroList, 'D-Value', 'F1-Score(Micro)', title, figname, np.arange(0, 1, step=0.1))
    '''
    # endregion


    f.close()


    #lsh.lsh(dataMatrix,trainLabels,testMatrix,testLabels)
