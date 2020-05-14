from _operator import itemgetter

#import numpy as np
#import scipy as sp
import time, math

#Step 1. Load the weather data
def Loaddata():
    '''
    make the weather dataset
    :return: weatherData(原始数据); featureName(天气特征); classValues(分类结果：是否打球)
    '''
    weatherData = [['Sunny','Hot','High','FALSE','N'],
        ['Sunny','Hot','High','TRUE','N'],
        ['Overcast','Hot','High','FALSE','P'],
        ['Rain','Mild','High','FALSE','P'],
        ['Rain','Cool','Normal','FALSE','P'],
        ['Rain','Cool','Normal','TRUE','N'],
        ['Overcast','Cool','Normal','TRUE','P'],
        ['Sunny','Mild','High','FALSE','N'],
        ['Sunny','Cool','Normal','FALSE','P'],
        ['Rain','Mild','Normal','FALSE','P'],
        ['Sunny','Mild','Normal','TRUE','P'],
        ['Overcast','Mild','High','TRUE','P'],
        ['Overcast','Hot','Normal','FALSE','P'],
        ['Rain','Mild','High','TRUE','N']]

    featureName = ['Outlook', 'Temperature', 'Humidity', 'Windy']
    classValues = ['P', 'N']

    return weatherData, featureName, classValues

def calcShannonEnt(paraDataSet):
    '''
    计算给定数据集的香浓熵
    :param paraDataSet: 给定数据集
    :return: shannonEnt
    '''
    numInstances = len(paraDataSet)  # numInstances:当前给定数据集中数据的个数
    labelCounts = {}
    for featureVec in paraDataSet:  # featureVec:数据集中的单个数据
        tempLabel = featureVec[-1]
        if tempLabel not in labelCounts.keys():
            labelCounts[tempLabel] = 0
        labelCounts[tempLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts.keys():
        prob = float(labelCounts[key])/numInstances
        shannonEnt -= prob * math.log2(prob)

    return shannonEnt

def splitDataSet(dataSet, axis, value):
    '''
    划分该出特征下层的数据集
    :param dataSet: 数据集
    :param axis: 第几个特征
    :param value: 该特征的值
    :return: resultDataSet:分类后的数据集
    '''
    resultDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
    # 因为划分当前数据的子集所以去掉当前数据集的分类标准（特征）
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            resultDataSet.append(reducedFeatVec)
    return resultDataSet

def chooseBestFeatureToSplit(dataSet):
    '''
    选择出最好的特征进行划分子数据集
    :param dataSet:数据集
    :return: bestFeature:决策出的划分效果最好（信息增益最大的特征）
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature  = -1
    for i in range(numFeatures):
#把第i个属性的所有取值筛选出来组成一个list
        featList = [data[i] for data in dataSet]
#去除list中的重复值
        uniqueVals = set(featList)
        newEntropy = 0.0
#计算条件信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

def majorityCnt(classList):
    '''
    :param classList:
    :return:投票决定的类别
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=itemgetter(1), reverse=True)
    return sortedClassCount

def creatTree(dataSet, paraFeatureName):
    '''
    建树
    :param dataSet:数据集
    :param paraFeatureName:数据集的属性名称
    :return: 递归创建完成的决策树
    '''
    featureName = paraFeatureName.copy()  # 防止后面原本的数据修改导致的分类出错
    classList = [example[-1] for example in dataSet]

#  如果当前的label只有一种类别则说明该子集已经完善
    if classList.count(classList[0]) == len(classList):
        return classList[0]

#  如果遇到属性一致但是结果不同的冲突情形（无分类属性可用），选择占比大的
    if len(dataSet) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatureName = featureName[bestFeat]
    myTree = {bestFeatureName:{}}
    del(featureName[bestFeat])
    featvalue = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featvalue)
    for value in uniqueVals:
        subfeatureName = featureName[:]
        myTree[bestFeatureName][value] = creatTree(splitDataSet(dataSet, bestFeat, value), subfeatureName)
    return myTree

def id3Classify(paraTree, paraTestingSet, featureNames, classValues):
    '''
    ID3分类器
    :param paraTree: 已生成的决策树
    :param paraTestingSet: 测试集
    :param featureNames: 特征名称
    :param classValues: 分类类型值
    :return: 正确率
    '''
    tempCorrect = 0.0
    tempTotal = len(paraTestingSet)
    tempPrediction = classValues[0]
    for featureVector in paraTestingSet:
        print("Instance: ", featureVector)
        tempTree = paraTree
        while True:
            for feature in featureNames:
                try:
                    tempTree[feature]
                    splitFeature = feature
                    break
                except:
                    i = 1
            attributeValue = featureVector[featureNames.index(splitFeature)]
            print(splitFeature, " = ", attributeValue)

            tempPrediction = tempTree[splitFeature][attributeValue]
            if tempPrediction in classValues:
                break
            else:
                tempTree = tempPrediction
        print("Prediction = ", tempPrediction)
        if featureVector[-1] == tempPrediction:
            tempCorrect += 1
    return tempCorrect/tempTotal

def STID3Test():
    weatherData, featureName, classValues = Loaddata()
    tempTree = creatTree(weatherData, featureName)
    print(tempTree)
    print("Before classification, feature names = ", featureName)
    tempAccuracy = id3Classify(tempTree, weatherData, featureName, classValues)
    print("The accuracy of ID3 classifier is {}%".format(tempAccuracy*100))


if __name__ == '__main__':
    STID3Test()

