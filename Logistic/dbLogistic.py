import time, sklearn
import sklearn.datasets, sklearn.neighbors, sklearn.linear_model

"""
This is using sklearn to test sklearn
"""
def sklearnLogisticTest():

    #Step1 Load the dataset
    tempDataset = sklearn.datasets.load_iris()
    x = tempDataset.data
    y = tempDataset.target

    #Step2 Classify
    tempClassifier = sklearn.linear_model.LogisticRegression()
    tempStartTime = time.time()
    tempClassifier.fit(x, y)
    tempScore = tempClassifier.score(x, y)
    tempEndTime = time.time()

    #Step3 Output
    print('sklearnLogistic score:{}, runtime= {}'.format(tempScore, tempEndTime-tempStartTime))

if __name__ == '__main__':
    sklearnLogisticTest()

