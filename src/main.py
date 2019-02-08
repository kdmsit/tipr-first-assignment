# region Packages
#This Region contains all imported packages that is used inside the code.
import numpy as np
import math
import sklearn
import nn
import time
import  lsh
import bayes
import twitter
import collections
import projections
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import operator
# endregion

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
    plt.yticks(yticky)
    for i in range(len(yVal)):
        plt.text(xVal[i], yVal[i], str(round(yVal[i],2)))
    plt.text(index, max/2, text)
    # plt.xticks(accuracyList,fontsize=5, rotation=30)
    # plt.xticks(index, accuracyList, fontsize=5, rotation=30)
    plt.title(title)
    plt.savefig(fileName)
    plt.close()

if __name__ == '__main__':
    Accuracy={}
    F1ScoreMacro={}
    F1ScoreMicro={}
    # region handle older versions of sklearn
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split
    # endregion
    print('Welcome to the world of high and low dimensions!')
    # The entire code should be able to run from this file!

    # Take File Path as Command Line Arguments
    '''inputDataFilePath = str(sys.argv[1])
    dataMatrix = np.genfromtxt(inputDataFilePath, delimiter=',')'''
    start = time.time()
    DataSetName='pubmed'
    print("Processing ",str(DataSetName)," DataSet.......")
    if(DataSetName.lower()=='twitter'):
        twitter.twitter()
    else:
        # Hard Coding File Path
        if (DataSetName.lower() == 'dolphin'):
            # region Dolphin Data Set
            inputFilePath='data/dolphins/'
            inputFileName='dolphins.csv'
            out='dolphins'
            inputLabelFileName = 'dolphins_label.csv'
            dataMatrix = np.genfromtxt(inputFilePath + inputFileName, delimiter=' ')
            # endregion
        else:
            # region Pubmed Data Set
            inputFilePath = 'data/pubmed/'
            inputFileName = 'pubmed.csv'
            out='dolphins'
            inputLabelFileName = 'pubmed_label.csv'
            dataMatrix = np.genfromtxt(inputFilePath+inputFileName, delimiter=' ')
            # endregion
        labelMatrix = np.genfromtxt(inputFilePath+inputLabelFileName, delimiter=' ')
        # endregion

        noOfColumns=len(dataMatrix[0])
        K = 2
        while (K <= int(noOfColumns / 2)):
            print("Reduced Dimension :",K)
            Accuracy[K]=0
            F1ScoreMacro[K] =0
            F1ScoreMicro[K] =0
            randomDataMatrix=projections.randomProjection(dataMatrix,noOfColumns,K,inputFilePath,out)

            (trainData, testData, trainLabels, testLabels) = train_test_split(randomDataMatrix, labelMatrix, test_size=0.25,
                                                                          random_state=42)
            '''accuracyOfMyCode, f1_score_macro, f1_score_micro = nn.nearestNeighbour(trainData, trainLabels, testData,
                                                                                   testLabels)'''
            accuracyOfMyCode, f1_score_macro, f1_score_micro = bayes.bayesClassifier(trainData, trainLabels, testData,
                                                                                     testLabels)
            Accuracy[K]=accuracyOfMyCode
            F1ScoreMacro[K]=f1_score_macro
            F1ScoreMicro[K]=f1_score_micro
            K = K * 2
        index=[]
        accuracyList=[]
        for k,accuracy in Accuracy.items():
            index.append(k)
            accuracyList.append(accuracy)
        #title="Accuracy For Different D values Of K-NN for "+ str(DataSetName)+" Data Set."
        #figname="K-NN_Accuracy_"+str(DataSetName)+".png"
        title = "Accuracy For Different D values Of Bayes for " + str(DataSetName.upper()) + " Data Set."
        figname="Bayes_Accuracy_"+str(DataSetName.upper())+".png"
        plotData(index,accuracyList,'D-Value','Accuracy',title,figname,np.arange(0,100,step=10))

        index = []
        F1MacroList = []
        for k, f1_score_macro in F1ScoreMacro.items():
            index.append(k)
            F1MacroList.append(f1_score_macro)
        #title = "F1-Score(Macro) For Different D values Of K-NN for " + str(DataSetName) + " Data Set."
        #figname = "K-NN_F1-Score(Macro)_" + str(DataSetName) + ".png"
        title = "F1-Score(Macro) For Different D values Of Bayes for " + str(DataSetName.upper()) + " Data Set."
        figname = "Bayes_F1-Score(Macro)_" + str(DataSetName.upper()) + ".png"
        plotData(index, F1MacroList, 'D-Value', 'F1-Score(Macro)', title, figname,np.arange(0,1,step=0.1))

        index = []
        F1MicroList = []
        for k, f1_score_micro in F1ScoreMicro.items():
            index.append(k)
            F1MicroList.append(f1_score_micro)
        #title = "F1-Score(Micro) For Different D values Of K-NN for " + str(DataSetName) + " Data Set."
        #figname = "K-NN_F1-Score(Micro)_" + str(DataSetName) + ".png"
        title = "F1-Score(Micro) For Different D values Of Bayes for " + str(DataSetName.upper()) + " Data Set."
        figname = "Bayes_F1-Score(Micro)_" + str(DataSetName.upper()) + ".png"
        plotData(index, F1MicroList, 'D-Value', 'F1-Score(Micro)', title, figname,np.arange(0,1,step=0.1))

        #priorDict=priorDensity(trainLabels)
        #accuracyOfMyCode, f1_score_macro, f1_score_micro=nn.nearestNeighbour(trainData,trainLabels,testData,testLabels)
        #lsh.lsh(trainData, trainLabels, testData, testLabels)
        #bayes.bayesClassifier(dataMatrix,labelMatrix)
    end = time.time()
    print("Time needed to execute :",end-start)



