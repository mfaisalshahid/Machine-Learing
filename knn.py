import matplotlib.pyplot as plt
import operator
from operator import itemgetter
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

names = [
  'sepal_length','sepal_width','petal_length','petal_width','class',
]
#change the directory file to your own iris dataset
df = pd.read_csv('/Users/muhammadshahid/Desktop/ML_Ass5/data/iris.data', header=None, names=names)

X = np.array(df.iloc[:, 0:4])
y = np.array(df['class']) 
accuracies = {}
for k in range(1,50):
    accuracy = 0
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i+2)
        # print(X_train, X_test)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        # print(pred)
        accuracy = accuracy + accuracy_score(y_test, pred)
        # print(accuracy)
    accuracies[k] = accuracy/5


print("K: {}".format(max(accuracies.items(), key=operator.itemgetter(1))[0]))
print("Accuracy: {}".format(max(accuracies.values())))
plt.plot(list(accuracies.keys()), list(accuracies.values()))
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.show()