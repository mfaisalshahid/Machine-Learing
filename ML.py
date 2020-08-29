import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
Import scikit-learn metrics module for accuracy calculation
from sklearn.externals.six import StringIO  
from IPython.display import Image 
import numpy as np
import pydotplus

pd.set_option('display.max_rows', 10)
task5 = pd.read_csv('/Users/muhammadshahid/Downloads/task5_train.csv')

task5 = task5.apply(lambda x: x.astype(str).str.lower())
task5 = task5.replace('win', 1)
task5 = task5.replace('lose', 0)
task5 = task5.replace('home', 1)
task5 = task5.replace('away', 0)
task5 = task5.replace('out', 1)
task5 = task5.replace('in', 0)

df_with_dummies = pd.get_dummies(task5, columns=['Media'])
task5 = pd.get_dummies(task5['Media'])
print(df_with_dummies)
print('**********')
print(media_Categorical)
print(task5)+ task5['Is_Home_or_Away']  + task5['Is_Opponent_in_AP25_Preseason'] +
f_columns = ['Is_Home_or_Away','Is_Opponent_in_AP25_Preseason','Media_abc','Media_cbs','Media_espn','Media_fox','Media_nbc']
xVar = df_with_dummies[f_columns]
yvar = task5.Label
X_train, X_test, y_train, y_test = train_test_split(xVar, yvar, test_size=0.3, random_state=1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(clf)
print(y_pred)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,filled=True, rounded=True, special_characters=True,feature_names = f_columns,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())