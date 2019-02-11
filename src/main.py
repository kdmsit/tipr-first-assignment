# region Packages
#This Region contains all imported packages that is used inside the code.
import numpy as np
import math
import sys
import sklearn
from sklearn.decomposition import PCA
import nn
import time
import  lsh
import bayes
import twitter
import collections
import projections
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import operator
# endregion

def priorDensity(trainLabels):
    '''
    :param trainLabels:     list of Labels Of Train Data
    :return:priorDict:      A dictionary with <key,value> pair <classLabel,priorDensity>
    Author: KD
    Details: This code calculates prior density of given data.
    '''
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

def plotData(xVal,yVal,xLabel,yLabel,title,fileName,yticky):
    '''
    :param xVal:        Value across X-axis.
    :param yVal:        Value across Y-axis.
    :param xLabel:      Labels of X-axis.
    :param yLabel:      Labels of Y-axis.
    :param title:       Title of the plot.
    :param fileName:    Name of the figure to be saved
    :param yticky:      Yticks of the data.
    :return:
    Author: KD
    Details: This code plots the data by matlibplot Library.
    '''
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
    plt.yticks(yticky)
    plt.xticks(xVal)
    for i in range(len(yVal)):
        plt.text(xVal[i], yVal[i], str(round(yVal[i],2)))
    plt.text(index, max/2, text)
    plt.title(title)
    plt.savefig(fileName)
    plt.close()



if __name__ == '__main__':

    # region handle older versions of sklearn
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split
    # endregion

    # region Local Variables
    Accuracy = {}
    F1ScoreMacro = {}
    F1ScoreMicro = {}
    # endregion
    start = time.time()
    print('Welcome to the world of high and low dimensions!')

    # Take File Path as Command Line Arguments
    testfile=sys.argv[2]
    testlabelfile=sys.argv[4]
    DataSetName=str(sys.argv[6])

    if(DataSetName.lower()=='twitter'):
        # If Data-Set Name is Twitter.
        testMatrix = np.genfromtxt(testfile, delimiter=' ')
        testlabelMatrix = np.genfromtxt(testlabelfile, delimiter=' ')
        twitter.twitter(testMatrix,testlabelMatrix)
    else:
        # If Data-Set Name is Dolphin/Pubmed.
        if (DataSetName.lower() == 'dolphin'):
            # region Dolphin Data Set
            inputFilePath='../data/dolphins/'
            inputFileName='dolphins.csv'
            out='dolphins'
            inputLabelFileName = 'dolphins_label.csv'
            dataMatrix = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
            # endregion
        else:
            # region Pubmed Data Set
            inputFilePath = '../data/pubmed/'
            inputFileName = 'pubmed.csv'
            out='dolphins'
            inputLabelFileName = 'pubmed_label.csv'
            dataMatrix = np.genfromtxt(inputFilePath+inputFileName, delimiter=' ')
            # endregion
        testMatrix = np.genfromtxt(testfile, delimiter=' ')
        testlabelMatrix = np.genfromtxt(testlabelfile, delimiter=' ')
        dataMatrix = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
        labelMatrix = np.genfromtxt(inputFilePath+inputLabelFileName, delimiter=' ')

        noOfColumns = len(dataMatrix[0])
        outputFileName=DataSetName+"_stat.txt"
        f = open(outputFileName, "w")

        print("Processing ", str(DataSetName), " DataSet.......")
        f.write("Processing " + str(DataSetName) + " DataSet.......")
        f.write("\n")

        # region Original Dimension Analysis
        trainData=dataMatrix
        trainLabels=labelMatrix
        testData=testMatrix
        testLabels=testlabelMatrix

        # region K-Nearest Neighbour Classifier
        print("Custom K-NN Classifier Statistcs:")
        print("K-Value Taken:5")
        f.write("Custom K-NN Classifier Statistcs:")
        f.write("\n")
        f.write("K-Value Taken:5")
        f.write("\n")
        accuracyNN, f1_score_macroNN, f1_score_microNN = nn.nearestNeighbour(trainData, trainLabels, testData,testLabels)
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
        accuracyBayes, f1_score_macroBayes, f1_score_microBayes = bayes.bayesClassifier(trainData, trainLabels, testData,testLabels)
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


        # region Task-VI:LSH Custom Code

        print("Custom LSH Classifier Statistcs:")
        f.write("Custom LSH Classifier Statistcs:")
        f.write("\n")
        f.write("\n")
        accuracyLSH, f1_score_macroLSH, f1_score_microLSH = lsh.lsh(trainData, trainLabels, testData, testLabels)
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

            Accuracy[K]=0
            F1ScoreMacro[K] =0
            F1ScoreMicro[K] =0
            #Random Projection into K diemsions

            randomDataMatrix=projections.randomProjection(dataMatrix,noOfColumns,K,inputFilePath,out)
            (trainData, testData, trainLabels, testLabels) = train_test_split(randomDataMatrix, labelMatrix, test_size=0.20,random_state=42)
            #accuracyOfMyCode, f1_score_macro, f1_score_micro = nn.nearestNeighbour(trainData, trainLabels, testData,testLabels)
            accuracyOfMyCode, f1_score_macro, f1_score_micro = bayes.bayesClassifier(trainData, trainLabels, testData,testLabels)

            print("Reduced Dimension :", K)
            print("Test Accuracy ",str(accuracyOfMyCode))
            print("Test Macro F1-score ",str(f1_score_macro))
            print("Test Micro F1-score ",str(f1_score_micro))
            f.write("Reduced Dimension :" + str(K))
            f.write("\n")
            f.write("Test Accuracy "+str(accuracyOfMyCode))
            f.write("\n")
            f.write("Test Macro F1-score "+str(f1_score_macro))
            f.write("\n")
            f.write("Test Micro F1-score "+str(f1_score_micro))
            f.write("\n")
            f.write("\n")
            Accuracy[K]=accuracyOfMyCode
            F1ScoreMacro[K]=f1_score_macro
            F1ScoreMicro[K]=f1_score_micro
            K = K * 2
        '''
        # endregion


        # region Plots and Save Data
        '''
        index = []
        accuracyList = []
        for k, accuracy in Accuracy.items():
            index.append(k)
            accuracyList.append(accuracy)
        #title = "Accuracy:Different D values Of Custom K-NN for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Custom-NN_Accuracy_" + str(DataSetName.upper()) + ".png"
        #title = "Accuracy:Different D values Of Sklearn K-NN for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Sklearn-NN_Accuracy_" + str(DataSetName.upper()) + ".png"
        #title = "Accuracy:Different D values Of Custom Bayes for " + str(DataSetName.upper()) + " Data Set."
        #figname="Custom-Bayes_Accuracy_"+str(DataSetName.upper())+".png"
        #title = "Accuracy Different D values Of Sklearn Bayes for " + str(DataSetName.upper()) + " Data Set."
        #figname="Sklearn_Bayes_Accuracy_"+str(DataSetName.upper())+".png"
        plotData(index, accuracyList, 'D-Value', 'Accuracy', title, figname, np.arange(0, 100, step=10))

        index = []
        F1MacroList = []
        for k, f1_score_macro in F1ScoreMacro.items():
            index.append(k)
            F1MacroList.append(f1_score_macro)
        #title = "F1-Score(Macro):Different D values Of Custom K-NN for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Custom_K-NN_F1-Score(Macro)_" + str(DataSetName.upper()) + ".png"
        #title = "F1-Score(Macro):Different D values Of Sklearn K-NN for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Sklearn_K-NN_F1-Score(Macro)_" + str(DataSetName.upper()) + ".png"
        #title = "F1-Score(Macro):Different D values Of Custom Bayes for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Custom_Bayes_F1-Score(Macro)_" + str(DataSetName.upper()) + ".png"
        #title = "F1-Score(Macro):Different D values Of Sklearn Bayes for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Sklearn_Bayes_F1-Score(Macro)_" + str(DataSetName.upper()) + ".png"
        plotData(index, F1MacroList, 'D-Value', 'F1-Score(Macro)', title, figname, np.arange(0, 1, step=0.1))

        index = []
        F1MicroList = []
        for k, f1_score_micro in F1ScoreMicro.items():
            index.append(k)
            F1MicroList.append(f1_score_micro)
        #title = "F1-Score(Micro):Different D values Of Custom K-NN for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Custom_K-NN_F1-Score(Micro)_" + str(DataSetName.upper()) + ".png"
        #title = "F1-Score(Micro):Different D values Of Sklearn K-NN for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Sklearn_K-NN_F1-Score(Micro)_" + str(DataSetName.upper()) + ".png"
        #title = "F1-Score(Micro):Different D values Of Custom Bayes for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Custom_Bayes_F1-Score(Micro)_" + str(DataSetName.upper()) + ".png"
        #title = "F1-Score(Micro):Different D values Of Sklearn Bayes for " + str(DataSetName.upper()) + " Data Set."
        #figname = "Sklearn_Bayes_F1-Score(Micro)_" + str(DataSetName.upper()) + ".png"
        plotData(index, F1MicroList, 'D-Value', 'F1-Score(Micro)', title, figname, np.arange(0, 1, step=0.1))
        '''
        # endregion


        # region Task-VII:PCA Analysis
        '''
        K = 2
        while (K <= int(noOfColumns / 2)):
            pca = PCA(n_components=K).fit(dataMatrix)
            data_reduce = pca.transform(dataMatrix)
            (trainData, testData, trainLabels, testLabels) = train_test_split(data_reduce, labelMatrix, test_size=0.20,random_state=42)
            accuracyOfMyCode, f1_score_macro, f1_score_micro = bayes.bayesClassifier(trainData, trainLabels, testData,testLabels)
            #accuracyOfMyCode, f1_score_macro, f1_score_micro = nn.nearestNeighbour(trainData, trainLabels, testData,testLabels)
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
        title = "Accuracy:Different D values Of PCA_Bayes for " + str(DataSetName.upper()) + " Data Set."
        figname = "PCA_Bayes_Accuracy_" + str(DataSetName.upper()) + ".png"
        plotData(index, accuracyList, 'D-Value', 'Accuracy', title, figname, np.arange(0, 100, step=10))

        index = []
        F1MacroList = []
        for k, f1_score_macro in F1ScoreMacro.items():
            index.append(k)
            F1MacroList.append(f1_score_macro)
        title = "F1-Score(Macro) Different D values Of PCA_Bayes for " + str(DataSetName.upper()) + " Data Set."
        figname = "PCA_Bayes_F1-Score(Macro)_" + str(DataSetName.upper()) + ".png"
        plotData(index, F1MacroList, 'D-Value', 'F1-Score(Macro)', title, figname, np.arange(0, 1, step=0.1))

        index = []
        F1MicroList = []
        for k, f1_score_micro in F1ScoreMicro.items():
            index.append(k)
            F1MicroList.append(f1_score_micro)
        title = "F1-Score(Micro) Different D values Of PCA_Bayes for " + str(DataSetName.upper()) + " Data Set."
        figname = "PCA_Bayes_F1-Score(Micro)_" + str(DataSetName.upper()) + ".png"
        plotData(index, F1MicroList, 'D-Value', 'F1-Score(Micro)', title, figname, np.arange(0, 1, step=0.1))
        '''
        # endregion


        f.close()
    end = time.time()
    print("Time needed to execute :",end-start)



