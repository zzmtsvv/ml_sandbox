import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def euclidean(point, data):
    return np.sqrt(np.sum(np.square(point - data), axis=1))

'''
def manhattan(point, data):
    return sum(abs(i - j) for i, j in zip(point, data))
'''

def most_common(array):
    return max(set(array), key=lambda x: array.count(x))


class KNNClassifier:
    def __init__(self, k=1, distance_metric=euclidean):
        self.k = k
        self.distance_metric = distance_metric
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        neighbours = []

        for x in x_test:
            distances = self.distance_metric(x, self.x_train)
            sorted_labels = [y for _, y in sorted(zip(distances, self.y_train), key=lambda z: z[0])]
            neighbours.append(sorted_labels[:self.k])
        
        return list(map(most_common, neighbours))
    
    def evaluate(self, x_test, y_test):
        y_pred = np.array(self.predict(x_test))
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


class KNNRegressor(KNNClassifier):
    def __init__(self, k=1, distance_metric=euclidean):
        super(KNNRegressor, self).__init__(k, distance_metric)
    
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self, x_test):
        neighbours = []

        for x in x_test:
            distances = self.distance_metric(x, self.x_train)
            sorted_labels = [y for _, y in sorted(zip(distances, self.y_train), key=lambda z: z[0])]
            neighbours.append(sorted_labels[:self.k])
        
        return np.mean(neighbours, axis=1)
    
    def evaluate(self, x_test, y_test):
        y_pred = self.predict(X_test)
        ssre = np.square(y_pred - y_test).sum()
        return ssre

'''
iris = datasets.load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

accuracies = []
ks = range(1, 101)
for k in ks:
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)

fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()
'''

housing = datasets.fetch_california_housing()
X = housing['data'][:500]
y = housing['target'][:500]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

ss = StandardScaler().fit(X_train)
X_train, X_test = ss.transform(X_train), ss.transform(X_test)

accuracies = []
ks = range(1, 101)
for k in ks:
    knn = KNNRegressor(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.evaluate(X_test, y_test)
    accuracies.append(accuracy)

fig, ax = plt.subplots()
ax.plot(ks, accuracies)
ax.set(xlabel="k",
       ylabel="SSRE",
       title="Performance of knn")
plt.show()
