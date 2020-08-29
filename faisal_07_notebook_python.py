import pandas as pd
import numpy as np
import seaborn as sns
import joblib
import os
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import NMF
from surprise import Dataset
from surprise import accuracy
from surprise import  Reader
from surprise import KNNBasic

# some code taken from the assignments PDF

file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
A7_dataset = Dataset.load_from_file(file_path, reader=reader)
fold_var = KFold(n_splits=3,random_state=0,shuffle=True)
train_data=[]
test_data=[]
for train_temp, test_temp in fold_var.split(A7_dataset):
    train_data.append(train_temp)
    test_data.append(test_temp)



def averageFunc(train_data, test_data, algorithm):
    rmsetotal_var = 0
    maetotal_var = 0
    for i in range(len(train_data)):
        algorithm.fit(train_temp)
        predict_var = algorithm.test(test_temp)

        rmsetotal_var+=accuracy.rmse(predict_var, verbose=True)
        maetotal_var+=accuracy.mae(predict_var, verbose=True)
    print("\nAverage Mean of RMSE is ={}".format(rmsetotal_var/len(train_data)))
    print("\nAverage Mean of MAE is ={}".format(maetotal_var/len(test_data)))
    
    return rmsetotal_var/len(train_data),maetotal_var/len(train_data)



def algoFunc(train_data,test_data):
    SVD_var=SVD()
    print("Singular Value Decomposition :\n")
    SVD_var.fit(train_data)
    predict_var = SVD_var.test(test_data)
    SVD_RMSE_var= accuracy.rmse(predict_var, verbose=True)
    SVD_MAE_var= accuracy.mae(predict_var, verbose=True)

    print("\nProbabilistic Matrix Factorization :\n")
    PMF_var=SVD(biased=False)
    PMF_var.fit(train_data)
    predict_var = PMF_var.test(test_data)
    PMF_RMSE_var= accuracy.rmse(predict_var, verbose=True)
    PMF_MAE_var= accuracy.mae(predict_var, verbose=True)

    print("\nNon-negative Matrix Factorization :\n")
    NMF_var=NMF()
    NMF_var.fit(train_data)
    predict_var = NMF_var.test(test_data)
    NMF_RMSE_var = accuracy.rmse(predict_var, verbose=True)
    NMF_MAE_var = accuracy.mae(predict_var, verbose=True)

    print("\nUser based Collaborative Filtering algorithm :\n")
    UB_var=KNNBasic(sim_options = {'user_based': True })
    UB_var.fit(train_data)
    predict_var = UB_var.test(test_data)
    user_RMSE_var = accuracy.rmse(predict_var, verbose=True)
    user_MAE_var=accuracy.mae(predict_var, verbose=True)

    print("\nItem based Collaborative Filtering algorithm :\n")
    IB_var=KNNBasic(sim_options = {'user_based': False })
    IB_var.fit(train_data)
    predict_var = IB_var.test(test_data)
    item_RMSE_var=accuracy.rmse(predict_var, verbose=True)
    item_MAE_var=accuracy.mae(predict_var, verbose=True)
    print("\n")
    
    return SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var



SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var = algoFunc(train_data[0],test_data[0])
temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])



SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var = algoFunc(train_data[1],test_data[1])
temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])


SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var = algoFunc(train_data[2],test_data[2])
temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])


print("SVD\n")
SVD_AVG_RMSE_var,SVD_AVG_MAE_var = averageFunc(train_data,test_data,SVD())

print("\nPMF\n")
PMF_AVG_RMSE_var,PMF_AVG_MAE_var = averageFunc(train_data,test_data,SVD(biased=False))

print("\nNMF\n")
NMF_AVG_RMSE_var,NMF_AVG_MAE_var = averageFunc(train_data,test_data,NMF())

print("\nUser-based Collaborative Filtering algorithm\n")
UB_AVG_RMSE_var,UB_AVG_MAE_var = averageFunc(train_data,test_data,KNNBasic(sim_options = {'user_based': True }))

print("\nItem based Collaborative Filtering algorithm\n")
IB_AVG_RMSE_var,IB_AVG_MAE_var = averageFunc(train_data,test_data,KNNBasic(sim_options = {'user_based': False }))

temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_AVG_RMSE_var,SVD_AVG_MAE_var,PMF_AVG_RMSE_var,PMF_AVG_MAE_var,NMF_AVG_RMSE_var,NMF_AVG_MAE_var,UB_AVG_RMSE_var,UB_AVG_MAE_var,IB_AVG_RMSE_var,IB_AVG_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])



print("\nUser based Collaborative Filtering - MSD Similarity\n")
userMSDRMSE,userMSDMAE = averageFunc(train_data,test_data,KNNBasic(sim_options = {'user_based': True }))

print("\nUser based Collaborative Filtering - Cosine Similarity\n")
userCosineRMSE,userCosineMAE = averageFunc(train_data,test_data,KNNBasic(sim_options = {'name':'cosine','user_based': True }))

print("\nUser based Collaborative Filtering - Pearson Similarity\n")
userPearsonRMSE,userPearsonMAE = averageFunc(train_data,test_data,KNNBasic(sim_options = {'name':'pearson','user_based': True }))

print("\nItem based Collaborative Filtering - MSD Similarity\n")
itemMSDRMSE,itemMSDMAE = averageFunc(train_data,test_data,KNNBasic(sim_options = {'user_based': False }))

print("\nItem based Collaborative Filtering - Cosine Similarity\n")
itemCosineRMSE,itemCosineMAE = averageFunc(train_data,test_data,KNNBasic(sim_options = {'name':'cosine','user_based': False }))

print("\nItem based Collaborative Filtering - Pearson Similarity\n")
itemPearsonRMSE,itemPearsonMAE = averageFunc(train_data,test_data,KNNBasic(sim_options = {'name':'pearson','user_based': False }))




temp_df = {
    'Algorithm': ('User MSD','User MSD','User Cosine', 'User Cosine','User Pearson','User Pearson'), 
    'RMSE and MAE': [userMSDRMSE,userMSDMAE,userCosineRMSE,userCosineMAE,userPearsonRMSE,userPearsonMAE],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.5,1.3])



temp_df = {
    'Algorithm': ('Item MSD','Item MSD','Item Cosine', 'Item Cosine','Item Pearson','Item Pearson'), 
    'RMSE and MAE': [itemMSDRMSE,itemMSDMAE,itemCosineRMSE,itemCosineMAE,itemPearsonRMSE,itemPearsonMAE],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.5,1.3])



KNN_RMSE=[]
KNN_MAE=[]
value_var=20
for i in range(1,value_var):
    RMSE_var,MAE_var = averageFunc(train_data,test_data,KNNBasic(k=i, sim_options = {'user_based': True }))
    KNN_RMSE.append(RMSE_var)
    KNN_MAE.append(MAE_var)



temp_df = {
    'KNN K Value': [y for y in range(1,value_var)], 
    'RMSE': KNN_RMSE
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="RMSE",data=plot_df,color="LightGreen")
graph.set_ylim([.9,1.35])



temp_df = {
    'KNN K Value': [y for y in range(1,value_var)], 
    'MAE': KNN_MAE
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="MAE",data=plot_df,color="LightBlue")
graph.set_ylim([.75,1.0])



best_k_var = KNN_RMSE.index(min(KNN_RMSE))+1
print("Best Value of K = {}".format(best_k_var))
print("RMSE = {}".format(min(KNN_RMSE)))



KNN_RMSE1=[]
KNN_MAE1=[]
value_var1=20
for i in range(1,value_var1):
    RMSE_var1,MAE_var1 = averageFunc(train_data,test_data,KNNBasic(k=i, sim_options = {'user_based': False }))
    KNN_RMSE1.append(RMSE_var1)
    KNN_MAE1.append(MAE_var1)


temp_df = {
    'KNN K Value': [y for y in range(1,value_var1)], 
    'RMSE': KNN_RMSE1
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="RMSE",data=plot_df,color="LightGreen")
graph.set_ylim([.9,1.5])


temp_df = {
    'KNN K Value': [y for y in range(1,value_var1)], 
    'MAE': KNN_MAE1
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="MAE",data=plot_df,color="LightBlue")
graph.set_ylim([.75,1.1])


best_K_var1 = KNN_RMSE1.index(min(KNN_RMSE1))+1
print("Best Value of K = {}".format(best_K_var1))
print("RMSE = {}".format(min(KNN_RMSE1)))

