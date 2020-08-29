import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant

# df = pd.read_csv("/Users/muhammadshahid/Desktop/HousingPrice.csv")
# # print(df['ExterQual'])
# freq = df['ExterQual'].value_counts()
# print (freq)

# # In[1]:

import os
# import pandas as pd


# In[2]:
os.getcwd()


df = pd.read_csv("/Users/muhammadshahid/Desktop/HousingPrice.csv")


# In[3]:


df.columns


# In[4]:


freq_table=pd.value_counts(df.ExterQual).to_frame().reset_index()

freq_table.columns=['Category_Value','Count']
freq_table


# In[5]:


leng= len(df.ExterQual)
leng


# In[6]:


freq_per= (freq_table.Count/leng)*100

freq_table['freq_percent']= freq_per
freq_table


# In[7]:


dummy= pd.get_dummies(df['ExterQual'])
dummy


# In[8]:


x = df.SalePrice
x


# In[9]:


y= dummy.TA
y


# In[10]:


#import statsmodels as sm

# import numpy as np


# In[25]:


test_obs = pd.DataFrame(np.array([200000, 0]))
test_obs


# In[26]:


test_obs = sm.add_constant(test_obs)


# In[27]:


test_obs


# In[28]:


x = sm.add_constant(x)
print(x)
test_obs1 = sm.add_constant(test_obs)
print(test_obs1)


# In[29]:


model = sm.Logit(y, x)
 
result = model.fit()
#logreg.fit(x,y)
print(result.summary())


# In[30]:


pred= result.predict(test_obs)
len(pred)


# In[31]:


pred


# In[18]:


x_train2 = df[['SalePrice','YearBuilt']]
x_train2


# In[19]:


x_train= sm.add_constant(x_train2)
x_train


# In[20]:


model2 = sm.Logit(y, x_train)
 
result2 = model2.fit()
#logreg.fit(x,y)
print(result2.summary())