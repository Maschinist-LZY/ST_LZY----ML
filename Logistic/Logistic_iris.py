import time
import numpy as np
import matplotlib.pyplot as plt

def LoadDataSet(paraFileName):
    '''
    LoadDataSet
    :param paraFileName:
    :return: 特征&标签
    '''
    dataMat = []
    labelMat = []
    txt = open(paraFileName)
    for line in txt.readlines():
        tempValuesStringArray = np.array(line.replace("\n", "").split(','))
        tempValues = [float(tempValue) for tempValue in tempValuesStringArray]
        tempArray = [1.0] + [tempValue for tempValue in tempValues] # 相当于w0对应的x0
        tempX = tempArray[:-1]
        tempY = tempArray[-1]
        dataMat.append(tempX)
        labelMat.append(tempY)

    return dataMat, labelMat

def sigmoid(paraX):
    '''
    sigmoid函数实现
    :param paraX:参数
    :return: 计算结果
    '''
    return 1.0/(1 + np.exp(-paraX))

def sigmoidPlotTest():
    '''
    画出sigmoid函数的图像
    :return: 图像
    '''
    xValue = np.linspace(-6, 6, 20)
    yValue = sigmoid(xValue)

    x2Value = np.linspace(-60, 60, 120)
    y2Value = sigmoid(x2Value)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(xValue, yValue)
    ax1.set_xlabel('x')
    ax1.set_ylabel('sigmoid(x)')
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x2Value, y2Value)
    ax2.set_xlabel('x')
    ax2.set_ylabel('sigmoid(x)')
    plt.show()

def gradAscent(dataMat, labelMat):
    '''
    用提督上升法求得最优权重
    :param dataMat:
    :param labelMat:
    :return:
    '''
    X = np.mat(dataMat)
    Y = np.mat(labelMat) #transfer the input to matrix
    Y = Y.transpose()
    m, n = np.shape(X)
    alpha = 0.001 # 学习步长
    maxCycles = 1000
    W = np.ones( (n,1))
    for i in range(maxCycles):
        y = sigmoid(X * W)
        error = Y - y
        W = W + alpha * X.transpose() * error

    return W

def STLogisticClassifierTest():
    '''
    Logistic 分类器
    '''
    X, Y = LoadDataSet('iris2condition2class.csv')

    tempStartTime = time.time()
    tempScore = 0
    numInstances = len(Y)
    weights = gradAscent(X, Y)

    tempPredicts = np.zeros((numInstances))

    for i in range(numInstances):
        tempPrediction = X[i] * weights
        if tempPrediction > 0:
            tempPredicts[i] = 1
        else:
            tempPredicts[i] = 0

    tempCorrect = 0
    for i in range(numInstances):
        if tempPredicts[i] == Y[i]:
            tempCorrect += 1

    tempScore = tempCorrect / numInstances
    tempEndTime = time.time()
    tempRunTime = tempEndTime - tempStartTime

    print('STLogistic Score: {}, runtime = {}'.format(tempScore, tempRunTime))

    rowWeights = np.transpose(weights).A[0]
    plotBestFit(rowWeights)


"""
函数：画出决策边界，仅为演示用，且仅支持两个条件属性的数据
"""
def plotBestFit(paraWeights):
    dataMat, labelMat = LoadDataSet('iris2class.txt')
    dataArr = np.array(dataMat)
    m, n = np.shape(dataArr)
    x1 = []  # x1,y1:类别为1的特征
    x2 = []  # x2,y2:类别为2的特征
    y1 = []
    y2 = []
    for i in range(m):
        if (labelMat[i]) == 1:
            x1.append(dataArr[i, 1])
            y1.append(dataArr[i, 2])
        else:
            x2.append(dataArr[i, 1])
            y2.append(dataArr[i, 2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')

    # 画出拟合直线
    x = np.arange(3, 7.0, 0.1)
    y = (-paraWeights[0] - paraWeights[1] * x) / paraWeights[2]  # 直线满足关系：0=w0*1.0+w1*x1+w2*x2
    ax.plot(x, y)

    plt.xlabel('a1')
    plt.ylabel('a2')
    plt.show()

if __name__ == '__main__':
    #sigmoidPlotTest()
    STLogisticClassifierTest()