import numpy as np


# Load DataSet
def readNominalData(paraFilename):
    '''
    read data from paraFilename/
    :param paraFilename:dataFile
    :return: resultNames(), resultData
    '''
    resultData  = []
    tempFile = open(paraFilename)
    tempLine = tempFile.readline().replace('\n', '')
    tempNames = np.array(tempLine.split(','))
    resultNames = [tempValue for tempValue in tempNames]

    tempLine = tempFile.readline().replace('\n', '')
    while tempLine != '':
        tempValues = np.array(tempLine.split(','))
        tempArray = [tempValue for tempValue in tempValues]
        resultData.append(tempArray)
        tempLine = tempFile.readline().replace('\n', '')

    tempFile.close()
    return resultNames, resultData

def obtainFeaturesValues(paraDataset):
    '''
    将整个数据集的特征值生成一个矩阵
    :param paraDataset:当前数据集
    :return:生成的矩阵
    '''
    resultMatrix = []
    for i in range(len(paraDataset[0])):
        featureValues = [example[i] for example in paraDataset]  # obtain all values of every feature
        uniqueValues = set(featureValues)
        currentValues = [tempValue for tempValue in uniqueValues]
        resultMatrix.append(currentValues)

    return resultMatrix

def calculateClassCounts(paraData, paraValuesMatrix):
    '''
    统计不同类别的数量
    :param paraData:dataSet
    :param paraValuesMatrix:特征值矩阵
    :return: 统计结果
    '''
    classCount = {}
    tempNumInstances = len(paraData)
    tempNumClasses = len(paraValuesMatrix[-1])

    for i in range(tempNumInstances):
        tempClass = paraData[i][-1]
        if tempClass not in classCount.keys():
            classCount[tempClass] = 0
        classCount[tempClass] += 1

    resultCounts = np.array(classCount)
    return resultCounts

def calculateClassDistributionLaplacian(paraData, paraValuesMatrix):
    '''
    class的概率计算，并进行拉普拉斯变换
    :param paraData: dataSet
    :param paraValuesMatrix: 特征值矩阵
    :return: 不同类别的概率
    '''
    classCount = {}
    tempNumInstances = len(paraData)
    tempNumClasses = len(paraValuesMatrix[-1])

    for i in range(tempNumInstances):
        tempClass = paraData[i][-1]
        if tempClass not in classCount.keys():
            classCount[tempClass] = 0
        classCount[tempClass] += 1

    resultClassDistribution = []
    for tempValue in paraValuesMatrix[-1]:
        resultClassDistribution.append((classCount[tempValue] + 1.0) / (tempNumInstances + tempNumClasses))

    print("tempNumClasses", tempNumClasses)

    return resultClassDistribution

def calculateMappings(paraValuesMatrix):
    '''
    将具体属性名称，取值映射为数值矩阵
    :param paraValuesMatrix: 属性取值矩阵
    :return: 映射后的数值矩阵
    '''
    resultMappings = []
    for i in range(len(paraValuesMatrix)):
        tempMapping = {}
        for j in range(len(paraValuesMatrix[i])):
            tempMapping[paraValuesMatrix[i][j]] = j
        resultMappings.append(tempMapping)

    return resultMappings

def calculateConditionalDistributionLaplacian(paraData, paraValuesMatrix, paraMappings):
    '''
    计算拉普拉斯变换后的条件概率
    :param paraData: dataSet
    :param paraValuesMatrix:属性取值矩阵
    :param paraMappings: 映射后的数值矩阵
    :return: 所有属性取值的条件概率
    '''
    tempNumInstances = len(paraData)
    tempNumConditions = len(paraData[0]) - 1
    tempNumClasses = len(paraValuesMatrix[-1])

    #Step1 Allocate Space
    tempCountCubic = []
    resultDistributionsLaplacianCubic = []
    for i in range(tempNumClasses):
        tempMatrix = []
        tempMatrix2 = []

        #Over all conditions
        for j in range(tempNumConditions):
            #Over all values
            tempNumValues = len(paraValuesMatrix[j])
            tempArray = [0.0] * tempNumValues
            tempArray2 = [0.0] * tempNumValues
            tempMatrix.append(tempArray)
            tempMatrix2.append(tempArray2)

        tempCountCubic.append(tempMatrix)
        resultDistributionsLaplacianCubic.append(tempMatrix2)

    #Step 2. Scan the dataSet
    for i in range(tempNumInstances):
        tempClass = paraData[i][-1]
        tempIntClass = paraMappings[tempNumConditions][tempClass] #get class index
        # 统计不同类别条件下，每种特征不同取值分别有多少个（eg：p的条件下，特征为a的有x个，b有x1个···）
        for j in range(tempNumConditions):
            tempValue = paraData[i][j]
            tempIntValue = paraMappings[j][tempValue] #get a feature's value's correaspondence index
            tempCountCubic[tempIntClass][j][tempIntValue] += 1

    #Calculate the real probability with LapLacian
    tempClassCounts = [0] * tempNumClasses
    for i in range(tempNumInstances):
        tempValue = paraData[i][-1]
        tempIntValue = paraMappings[tempNumConditions][tempValue]
        tempClassCounts[tempIntValue] += 1

    for i in range(tempNumClasses):
        for j in range(tempNumConditions):
            for k in range(len(tempCountCubic[i][j])):
                resultDistributionsLaplacianCubic[i][j][k] = (tempCountCubic[i][j][k] + 1) / (tempClassCounts[i] + tempNumClasses)

    return resultDistributionsLaplacianCubic


def nbClassify(paraTestData, paraValueMatrix, paraClassValues, paraMappings, paraClassDistribution, paraDistributionCubic):
    '''
    分类并返回正确率
    :param paraTestData:
    :param paraValueMatrix:
    :param paraClassValues:
    :param paraMappings:
    :param paraClassDistribution:
    :param paraDistributionCubic:
    :return: 正确率
    '''
    tempCorrect = 0.0
    tempNumInstances = len(paraTestData)
    tempNumConditions = len(paraTestData[0]) - 1
    tempNumClasses = len(paraValueMatrix[-1])
    tempTotal = len(paraTestData)
    tempBiggest = -1000
    tempBest = -1

    for featureVector in paraTestData:
        tempActualLabel = paraMappings[tempNumConditions][featureVector[-1]]
        tempBiggest = -1000
        tempBest = -1
        for i in range(tempNumClasses):
            tempPro = np.log(paraClassDistribution[i])
            for j in range(tempNumConditions):
                tempValue = featureVector[j]
                tempIntValue = paraMappings[j][tempValue]
                tempPro += np.log(paraDistributionCubic[i][j][tempIntValue])

            if tempBiggest < tempPro:
                tempBiggest = tempPro
                tempBest = i
        if tempBest == tempActualLabel:
            tempCorrect += 1
    return tempCorrect/tempNumInstances

def STNBTest(paraFileName):
    featureNames, dataSet = readNominalData(paraFileName)

    print("Feature Names = ", featureNames)

    valuesMatrix = obtainFeaturesValues(dataSet)
    tempMappings = calculateMappings(valuesMatrix)
    classSumValues = calculateClassCounts(dataSet, valuesMatrix)
    classDistribution = calculateClassDistributionLaplacian(dataSet, valuesMatrix)
    print("classDistribution = ", classDistribution)

    conditionalDistributions = calculateConditionalDistributionLaplacian(dataSet, valuesMatrix, tempMappings)

    tempAccuracy = nbClassify(dataSet, valuesMatrix, classSumValues, tempMappings, classDistribution, conditionalDistributions)

    print("The accuracy of NB classifier is {}".format(tempAccuracy))

if __name__ == '__main__':
    STNBTest('mushroom.csv')










