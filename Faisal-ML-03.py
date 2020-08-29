import pandas as pd
from IPython.display import display
from chefboost import Chefboost as chef
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#substitue the paths for the file according to your computer, they are currently set to my directory

# Task 4
df = pd.read_csv("/Users/muhammadshahid/Downloads/task4-1.csv")
target = df.Label
data =df[["HomeOrAway", "InOrOut", "Media"]]
data = pd.get_dummies(data)

# Entropy
decision_tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
decision_tree = decision_tree.fit(data, target)
plot_tree(decision_tree)

#%%
# Gini
decision_tree = DecisionTreeClassifier(random_state=0, criterion='gini')
decision_tree = decision_tree.fit(data, target)
plot_tree(decision_tree)

# C4.5
df = pd.read_csv("/Users/muhammadshahid/Downloads/task4-1.csv")[["HomeOrAway", "InOrOut", "Media", "Label"]]
df = df.rename(columns={"Label": "Decision"})
test = pd.read_csv("/Users/muhammadshahid/Downloads/task4-1.csv")
config_c45 = {'algorithm': 'C4.5'}
model_c45 = chef.fit(df.copy(), config_c45)

for index, instance in test.iterrows():
    prediction = chef.predict(model_c45, instance)
    print(index, prediction)


# Task 4-2
df = pd.read_csv("/Users/muhammadshahid/Downloads/task4-2.csv")
test = pd.read_csv("/Users/muhammadshahid/Downloads/task4-2-test.csv")

target = df.Label
data = df[["Outlook","Temperature","Humidity","Windy"]]
data = pd.get_dummies(data)

test = pd.get_dummies(test)
test = data.iloc[0:0].combine_first(test).fillna("0")
test

#Entropy
fig = plt.figure(figsize=[10,10])

decision_tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
decision_tree = decision_tree.fit(data, target)
plot_tree(
    decision_tree,
    filled=True,
    rounded=True,
    class_names=["Yes", "No"]
)

print(f'Prediction: {decision_tree.predict(test)}')

#Gini
fig = plt.figure(figsize=[14,6])

decision_tree = DecisionTreeClassifier(random_state=0, criterion='gini')
decision_tree = decision_tree.fit(data, target)
plot_tree(
    decision_tree,
    filled=True,
    rounded=True,
    class_names=["Yes", "No"]
)
print(f'Prediction: {decision_tree.predict(test)}')

#C4.5
df = pd.read_csv("/Users/muhammadshahid/Downloads/task4-2.csv")[["Outlook","Temperature","Humidity","Windy","Label"]]
df = df.rename(columns={"Label": "Decision"})
test = pd.read_csv("/Users/muhammadshahid/Downloads/task4-2-test.csv")

config_c45 = {'algorithm': 'C4.5'}
model_c45 = chef.fit(df.copy(), config_c45)

for index, instance in test.iterrows():
    prediction = chef.predict(model_c45, instance)
    print(index, prediction)

#Task 5

df = pd.read_csv("/Users/muhammadshahid/Downloads/task5_train.csv")
test = pd.read_csv("/Users/muhammadshahid/Downloads/task5_test.csv")
test_target = test["Label"].replace("Win", 1).replace("Lose", 0).to_numpy()

target = df.Label.replace("Win", 1).replace("Lose", 0)

data = df[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","Media"]]
data = pd.get_dummies(data)

test = test[["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","Media"]]
test = pd.get_dummies(test)
test = data.iloc[0:0].combine_first(test).fillna("0")

def print_stats(y_true, y_pred):
    print(f'Precision: {precision_score(y_true, y_pred)}')
    print(f'Recall: {round(recall_score(y_true, y_pred), 2)}')
    print(f'F1 Score: {round(f1_score(y_true, y_pred), 2)}')
    print(f'Accuracy Score: {round(accuracy_score(y_true, y_pred),2)}')

#Entropy
fig = plt.figure(figsize=[10,12])

decision_tree = DecisionTreeClassifier(random_state=0, criterion='entropy')
decision_tree = decision_tree.fit(data, target)
plot_tree(
    decision_tree,
    # feature_names=[list(data.columns)],
    filled=True,
    rounded=True,
    class_names=["Yes", "No"]
)
pred_target = decision_tree.predict(test)
print_stats(test_target, pred_target)

#Gini
fig = plt.figure(figsize=[14,6])

decision_tree = DecisionTreeClassifier(random_state=0, criterion='gini')
decision_tree = decision_tree.fit(data, target)
plot_tree(
    decision_tree,
    filled=True,
    rounded=True,
    class_names=["Yes", "No"]
)
pred_target = decision_tree.predict(test)
print_stats(test_target, pred_target)

#C4.5
train_cols = ["Is_Home_or_Away","Is_Opponent_in_AP25_Preseason","Media", "Label"]
df = pd.read_csv("/Users/muhammadshahid/Downloads/task5_train.csv")[train_cols]
df = df.rename(columns={"Label": "Decision"})

config_c45 = {'algorithm': 'C4.5'}
model_c45 = chef.fit(df.copy(), config_c45)

test = pd.read_csv("/Users/muhammadshahid/Downloads/task5_test.csv")
test_target = test["Label"].replace({"Win": 1, "Lose": 0})
test = test[train_cols]

pred_target = []

for index, instance in test.iterrows():
    prediction = chef.predict(model_c45, instance)
    pred_target.append(1 if prediction == "Win" else 0)

print_stats(test_target, pred_target)
