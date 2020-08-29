import pandas as pd
import math
import numpy as np

df = pd.read_csv(
    "/Users/muhammadshahid/Desktop/train.csv",
    engine="c",
    low_memory=False
)

df1 = pd.read_csv(
    "/Users/muhammadshahid/Desktop/test.csv",
    engine="c",
    low_memory=False
)

combined = df.append(df1)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


print (combined[['Age', 'Fare', 'Parch', 'SibSp']].describe())
combined.Pclass = combined.Pclass.astype('category')
combined.Sex = combined.Sex.astype('category')
combined.Embarked = combined.Embarked.astype('category')
combined.Survived = combined.Survived.astype('category')
print(combined.describe(include=['category']))
