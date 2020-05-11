import numpy, time, sklearn
import sklearn.datasets, sklearn.neighbors

def sklearnKnnClassifierTest():
    #Step 1. Load the dataset
    tempDataset = sklearn.datasets.load_iris()
    x = tempDataset.data
    y = tempDataset.target

    #Step 2. Classify
    tempClassifier = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
    tempStartTime = time.time()
    tempClassifier.fit(x, y)
    tempScore = tempClassifier.score(x, y)
    tempEndTime = time.time()
    tempRuntime = tempEndTime - tempStartTime

    #Step 3. Output
    print('sklearn socre: {}, runtime = {}'.format(tempScore, tempRuntime))

if __name__ == '__main__':
    sklearnKnnClassifierTest()