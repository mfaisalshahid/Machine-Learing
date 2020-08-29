#%%
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np

#%%

def print_stats(y_true, y_pred):
    print(f'Precision: {precision_score(y_true, y_pred)}')
    print(f'Recall: {round(recall_score(y_true, y_pred), 2)}')
    print(f'F1 Score: {round(f1_score(y_true, y_pred), 2)}')
    print(f'Accuracy Score: {round(accuracy_score(y_true, y_pred),2)}')

def prepare_dataframe(df):
    df["Is_Home_or_Away"] = df["Is_Home_or_Away"].replace({"Away": 0, "Home": 1})
    df["Is_Opponent_in_AP25_Preseason"] = df["Is_Opponent_in_AP25_Preseason"].replace({"In": 1, "Out": 0})
    media_dummy = pd.get_dummies(df["Media"])
    df = pd.concat([df, media_dummy], axis=1)
    df = df.drop(["Media", "Label", "ID"], axis=1)
    return df

enc = OrdinalEncoder()
def encoder(df):
    if hasattr(enc, 'categories_'):
        return enc.transform(df)
    enc.fit(df)
    return enc.transform(df)

def get_encoded(labels):
    return labels.replace({"Win": 1, "Lose": 0})

#%%
gnb = CategoricalNB()
feature_cols = ["Is_Home_or_Away", "Is_Opponent_in_AP25_Preseason", "Media"]

train = pd.read_csv("./train.csv")
train_label = get_encoded(train["Label"])
train = encoder(train[feature_cols])

test = pd.read_csv("./test.csv")
test_labels = get_encoded(test["Label"])
test = encoder(test[feature_cols])
#%%

y_pred = gnb.fit(train, train_label).predict(test)
pred_labels = np.where(y_pred==0, "Lose", "Win")

#%%

output = pd.read_csv("./test.csv")
output["Predicted"] = pred_labels
output = output[["Label", "Predicted"]]
output


#%%
print_stats(test_labels, y_pred)
