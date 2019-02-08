# Implement code for random projections here!
import numpy as np
import math


def randomProjection(dataMatrix,noOfColumns,K,inputFilePath,out):
    '''
    :param dataMatrix:
    :param noOfColumns:
    :param inputFilePath:
    :param out:
    :return:
    This Region Of code Does The following
        1.Generate Random Matrix On Gaussian Distribution with mean=0 and Variance=1.
        2.Multiply Data Matrix with Random Matrix and finally multiply with Normalised Value.
        3.We save the output matrix in output file path.
    '''
    normalisedFactor = 1 / (math.sqrt(K))
    randMatrix = np.random.normal(0, 1, (noOfColumns, K))
    noOfRowsRand = len(randMatrix)
    noOfColumnsRand = len(randMatrix[0])
    # print(noOfRows, noOfColumns)
    Emul = np.matmul(dataMatrix, randMatrix)
    Enorm = np.multiply(normalisedFactor, Emul)
    outputFileName =out+'_' + str(K) + '.csv'
    np.savetxt(inputFilePath + outputFileName, Enorm, delimiter=" ")
    return Enorm
