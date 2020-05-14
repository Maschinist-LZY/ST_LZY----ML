import sklearn
from sklearn.model_selection import train_test_split
import sklearn.datasets, sklearn.neighbors, sklearn.tree, sklearn.metrics



def sklearnDecisionTreeTest():
    # Step 1. Load the dataset
    tempDataset = sklearn.datasets.load_breast_cancer()
    x = tempDataset.data
    y = tempDataset.target

    # Split for training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Step 2. Build classifier
    tempClassifier = sklearn.tree.DecisionTreeClassifier(criterion='entropy')
    tempClassifier.fit(x_train, y_train)

    # Step 3. Test
    # precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, tempClassifier.predict(x_test))
    tempAccuracy = sklearn.metrics.accuracy_score(y_test, tempClassifier.predict(x_test))
    tempRecall = sklearn.metrics.recall_score(y_test, tempClassifier.predict(x_test))

    # Step 4. Output
    print("precision = {}, recall = {}".format(tempAccuracy, tempRecall))

if __name__ == '__main__':
    sklearnDecisionTreeTest()