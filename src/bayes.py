# Implement Bayes Classifier here!
# region Packages
#This Region contains all imported packages that is used inside the code.
import numpy as np
import sklearn
from sklearn.metrics import f1_score
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

def meanOfClass(featureValues):
    '''
    :param featureValues:
    :return:Mean of Feature Values
    Author: KD
    Details: This Code Calculates the mean of Feature Values.
    '''
    return np.mean(featureValues)

def stdevOfClass(featureValues):
    '''
    :param featureValues:
    :return:Standard Deviation of Feature Values
    Author: KD
    Details: This Code Calculates the Standard Deviation of Feature Values.
    '''
    return math.sqrt(np.var(featureValues))

def calMeanVar(featureSet):
    '''
    :param  featureSet:
    :return: meanVarSet
    Author: KD
    Details: This Code Calculates Mean and Variance for each feature of the feature set.
    '''
    meanVarSet = [(meanOfClass(attribute), stdevOfClass(attribute)) for attribute in zip(*featureSet)]
    return meanVarSet

def calculateGaussianDensity(x, mean, stdev):
    '''
    :param x:
    :param mean:
    :param stdev:
    :return:This code returns probability density f(x)~N(mean,stdev) for each x from test data set.
    Author: KD
    '''
    a=math.pow(x - mean, 2)
    b=math.pow(stdev, 2)
    b=2*b
    c=-(a/b)
    d=np.exp(c)
    exponent = np.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    prob=(1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
    return prob

def calculateClassConditionalDensities(meanStdSet, testInputVector):
    '''
    :param meanStdSet:
    :param testInputVector:
    :return:classConditionalProb
    Author: KD
    Details: This code calculates the class conditional densities of input vector for each class.
    '''

    classConditionalProb = {}
    for classlabel, classmeanStd in meanStdSet.items():
        #classConditionalProb[classlabel] = 1
        for i in range(len(classmeanStd)):
            mean, stdev = classmeanStd[i]
            if (stdev != 0):
                x = testInputVector[i]
                if (classlabel not in classConditionalProb):
                    classConditionalProb[classlabel]=calculateGaussianDensity(x, mean, stdev)
                else:
                    classConditionalProb[classlabel] *= calculateGaussianDensity(x, mean, stdev)
            else:
                continue
    return classConditionalProb

def bayesClassifier(trainData, trainLabels, testData,testLabels):
    '''
        :param dataMatrix:
        :param labelMatrix:
        :return: Accuracy of the classifier
        Author: KD
        Details: This is the main BayesClassifier function.
        '''

    # region handle older versions of sklearn
    if int((sklearn.__version__).split(".")[1]) < 18:
        from sklearn.cross_validation import train_test_split
    # otherwise we're using at lease version 0.18
    else:
        from sklearn.model_selection import train_test_split
    # endregion

    # region Sklearn Package Naive Bayes
    '''
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    y_pred = gnb.fit(dataMatrix, labelMatrix).predict(dataMatrix)
    Total = dataMatrix.shape[0]
    predicted = Total - (labelMatrix != y_pred).sum()
    accuracy = (predicted / Total) * 100.0
    print("Bayes:Library Code Runs:",accuracy)
    '''
    # endregion

    classPartitionSet = classPartition(trainData, trainLabels)
    priorProbSet = {}
    for classValue, features in classPartitionSet.items():
        priorProbSet[classValue] = len(features) / len(trainData)

    # region Other Two Data Sets
    meanStdSet = {}
    for classValue, features in classPartitionSet.items():
        meanStdSet[classValue] = calMeanVar(features)

    predictions = []
    accuracy = 0
    for i in range(len(testData)):
        testVector = testData[i]
        probabilities = calculateClassConditionalDensities(meanStdSet, testVector)
        posteriorProb={}
        for key in probabilities:
            posteriorProb[key]=priorProbSet[key]*probabilities[key]
        predictedLabel = None
        PredictionProb = -1
        for classValue, probability in posteriorProb.items():
            if predictedLabel is None or probability > PredictionProb:
                PredictionProb = probability
                predictedLabel = classValue

        predictions.append(predictedLabel)
    # endregion

    for x in range(len(testData)):
        if testLabels[x] == predictions[x]:
            accuracy += 1
    accuracyOfMyCode = (accuracy / len(testData)) * 100.0
    f1_score_macro = f1_score(testLabels, predictions, average='macro')
    f1_score_micro = f1_score(testLabels, predictions, average='micro')
    print("Bayes Classifier:Custom Code Test Accuracy ", accuracyOfMyCode)
    print("Bayes Classifier:Custom Code Test F1-Score(Macro): ", f1_score_macro)
    print("Bayes Classifier:Custom Code Test F1-Score(Micro): ", f1_score_micro)
    return accuracyOfMyCode, f1_score_macro, f1_score_micro
