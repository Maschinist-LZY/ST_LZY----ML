import numpy as np
import time, sklearn
import sklearn.datasets, sklearn.neighbors, sklearn.model_selection

# Step1 load iris data
def Loaddata():
    '''
    :tempDataset: the returned Bunch object
    :X: 150 flowers' data (花朵的特征数据)
    :Y: 150 flowers' label (花朵的种类标签)
    :return: X1(train_set), Y1(train_label), X2(test_set), Y2(test_label)
    '''

    tempDataset = sklearn.datasets.load_iris()
    X = tempDataset.data
    Y = tempDataset.target
    # Step2 split train set & test set
    X1, X2, Y1, Y2 = sklearn.model_selection.train_test_split(X, Y, test_size=0.2)
    return X1, Y1, X2, Y2

def euclideanDistance(x, y):
    '''
    计算欧式距离
    :param x: 某一朵花的数据
    :param y: 另一朵花的数据
    :return: 两个花朵数据之间的欧式距离
    '''
    tempDistance = 0
    m = x.shape[0]
    for i in range(m):
        tempDifference = x[i] - y[i]
        tempDistance += tempDifference * tempDifference
    return tempDistance**0.5

# Step3 Classify
def stKnnClassifierTest(X1, Y1, X2, Y2, K = 5):
    '''
    :param X1: train_set
    :param Y1: train_label
    :param X2: test_set
    :param Y2: test_label
    :param K: 所选的neighbor的数量
    :return: no return
    '''
    tempStartTime = time.time()
    tempScore = 0
    test_Instances = Y2.shape[0]
    train_Instances = Y1.shape[0]
    print('the num of testInstances = {}'.format(test_Instances))
    print('the num of trainInstances = {}'.format(train_Instances))
    tempPredicts = np.zeros((test_Instances))

    for i in range(test_Instances):
        # tempDistacnes = np.zeros((test_Instances))

        # Find K neighbors
        tempNeighbors = np.zeros(K + 2)
        tempDistances = np.zeros(K + 2)

        for j in range(K + 2):
            tempDistances[j] = 1000
        tempDistances[0] = -1

        for j in range(train_Instances):
            tempdis = euclideanDistance(X2[i], X1[j])
            tempIndex = K
            while True:
                if tempdis < tempDistances[tempIndex]:
            # prepare move forward
                    tempDistances[tempIndex + 1] = tempDistances[tempIndex]
                    tempNeighbors[tempIndex + 1] = tempNeighbors[tempIndex]
                    tempIndex -= 1
            #insert
                else:
                    tempDistances[tempIndex + 1] = tempdis
                    tempNeighbors[tempIndex + 1] = j
                    break

        # Vote
        tempLabels = []
        for j in range(K):
            tempIndex = int(tempNeighbors[j + 1])
            tempLabels.append(int(Y1[tempIndex]))

        tempCounts = []
        for label in tempLabels:
            tempCounts.append(int(tempLabels.count(label)))
        tempPredicts[i] = tempLabels[np.argmax(tempCounts)]

    # the rate of correct classify
    tempCorrect = 0
    for i in range(test_Instances):
        if tempPredicts[i] == Y2[i]:
            tempCorrect += 1

    tempScore = tempCorrect / test_Instances

    tempEndTime = time.time()
    tempRunTime = tempEndTime - tempStartTime

    print('ST KNN score: {}%, runtime = {}'.format(tempScore*100, tempRunTime))

                    
if __name__ == '__main__':
    X1, Y1, X2, Y2 = Loaddata()
    stKnnClassifierTest(X1, Y1, X2, Y2)

