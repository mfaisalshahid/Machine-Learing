#!/usr/bin/env python
# coding: utf-8

# In[21]:


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


# In[22]:


file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
A7_dataset = Dataset.load_from_file(file_path, reader=reader)
fold_var = KFold(n_splits=3,random_state=0,shuffle=True)
train_data=[]
test_data=[]
for train_temp, test_temp in fold_var.split(A7_dataset):
    train_data.append(train_temp)
    test_data.append(test_temp)


# In[23]:


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


# In[24]:


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


# Question 10. Compare the performances of User-based collaborative filtering, item-based collaborative filtering, SVD, PMF, NMF on fold-1 with respect to RMSE and MAE. Since data.split(n_folds=3)randomly split the data into 3 folds, please make sure you test the five algorithms on the same fold-1 so the results are comparable.
# 
# Answer - On comparing the performance observations below we can infer that SVD has lowest RMSE and MAE.

# In[25]:


SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var = algoFunc(train_data[0],test_data[0])
temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])


# Question 11. Compare the performances of User-based collaborative filtering, item-based collaborative filtering, SVD, PMF, NMF on fold-2 with respect to RMSE and MAE. Please make sure you test the five algorithms on the same fold-2 so the results are comparable.
# 
# Answer - On comparing the performance observations below we can infer that SVD has lowest RMSE and MAE.

# In[26]:


SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var = algoFunc(train_data[1],test_data[1])
temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])


# Question 12. Compare the performances of User-based collaborative filtering, item-based collaborative filtering, SVD, PMF, NMF on fold-3 with respect to RMSE and MAE. Please make sure you test the five algorithms on the same fold-3 so the results are comparable.
# 
# Answer - On comparing the performance observations below we can infer that SVD has lowest RMSE and MAE.

# In[40]:


SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var = algoFunc(train_data[2],test_data[2])
temp_df = {
    'Algorithm': ('SVD','SVD','PMF', 'PMF','NMF','NMF','User Based','User Based','Item Based','Item Based'), 
    'RMSE and MAE': [SVD_RMSE_var,SVD_MAE_var,PMF_RMSE_var,PMF_MAE_var,NMF_RMSE_var,NMF_MAE_var,user_RMSE_var,user_MAE_var,item_RMSE_var,item_MAE_var],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.6,1.1])


# Question 13. Compare the average (mean) performances of User-based collaborative filtering, itembased collaborative filtering, SVD, PMF, NMF with respect to RMSE and MAE. Please make sure you test the five algorithms on the same 3-fold data split plan so the results are comparable.
# 
# Answer - On comparing the  average (mean) performance observations below we can infer that SVD has lowest RMSE and MAE.

# In[61]:


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


# Question 14. Examine how the cosine, MSD (Mean Squared Difference), and Pearson similarities impact the performances of User based Collaborative Filtering and Item based Collaborative Filtering. Finally, is the impact of the three metrics on User based Collaborative Filtering consistent with the impact of the three metrics on Item based Collaborative Filtering? Plot your results.
# 
# Answer - On comparing the observations below we can infer that MSD similarity has lowest RMSE and MAE for both User-Based Collaborative Filtering and Item Based Collaborative Filtering.

# In[13]:


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


# In[14]:


temp_df = {
    'Algorithm': ('User MSD','User MSD','User Cosine', 'User Cosine','User Pearson','User Pearson'), 
    'RMSE and MAE': [userMSDRMSE,userMSDMAE,userCosineRMSE,userCosineMAE,userPearsonRMSE,userPearsonMAE],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.5,1.3])


# In[15]:


temp_df = {
    'Algorithm': ('Item MSD','Item MSD','Item Cosine', 'Item Cosine','Item Pearson','Item Pearson'), 
    'RMSE and MAE': [itemMSDRMSE,itemMSDMAE,itemCosineRMSE,itemCosineMAE,itemPearsonRMSE,itemPearsonMAE],
    'Metric':['RMSE','MAE','RMSE','MAE','RMSE','MAE']
}
plot_df = pd.DataFrame(temp_df)
x = sns.barplot(x="Algorithm",y="RMSE and MAE",data=plot_df,hue="Metric",palette="Set2")
x.set_ylim([.5,1.3])


# Question 15. Examine how the number of neighbors impacts the performances of User based Collaborative Filtering or Item based Collaborative Filtering? Plot your results. Identify the best K for User/Item based collaborative filtering in terms of RMSE. Is the the best K of User based collaborative filtering the same with the best K of Item based collaborative filtering?
# 
# Answer - 
# 1. Best value of k for User based Collaborative Filtering is 19.
# 2. Best value of k for Item based Collaborative Filtering is 19.

# In[101]:


KNN_RMSE=[]
KNN_MAE=[]
value_var=20
for i in range(1,value_var):
    RMSE_var,MAE_var = averageFunc(train_data,test_data,KNNBasic(k=i, sim_options = {'user_based': True }))
    KNN_RMSE.append(RMSE_var)
    KNN_MAE.append(MAE_var)


# In[102]:


temp_df = {
    'KNN K Value': [y for y in range(1,value_var)], 
    'RMSE': KNN_RMSE
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="RMSE",data=plot_df,color="LightGreen")
graph.set_ylim([.9,1.35])


# In[103]:


temp_df = {
    'KNN K Value': [y for y in range(1,value_var)], 
    'MAE': KNN_MAE
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="MAE",data=plot_df,color="LightBlue")
graph.set_ylim([.75,1.0])


# In[104]:


best_k_var = KNN_RMSE.index(min(KNN_RMSE))+1
print("Best Value of K = {}".format(best_k_var))
print("RMSE = {}".format(min(KNN_RMSE)))


# In[105]:


KNN_RMSE1=[]
KNN_MAE1=[]
value_var1=20
for i in range(1,value_var1):
    RMSE_var1,MAE_var1 = averageFunc(train_data,test_data,KNNBasic(k=i, sim_options = {'user_based': False }))
    KNN_RMSE1.append(RMSE_var1)
    KNN_MAE1.append(MAE_var1)


# In[106]:


temp_df = {
    'KNN K Value': [y for y in range(1,value_var1)], 
    'RMSE': KNN_RMSE1
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="RMSE",data=plot_df,color="LightGreen")
graph.set_ylim([.9,1.5])


# In[107]:


temp_df = {
    'KNN K Value': [y for y in range(1,value_var1)], 
    'MAE': KNN_MAE1
}
plot_df=pd.DataFrame(temp_df)
graph = sns.barplot(x="KNN K Value",y="MAE",data=plot_df,color="LightBlue")
graph.set_ylim([.75,1.1])


# In[108]:


best_K_var1 = KNN_RMSE1.index(min(KNN_RMSE1))+1
print("Best Value of K = {}".format(best_K_var1))
print("RMSE = {}".format(min(KNN_RMSE1)))

