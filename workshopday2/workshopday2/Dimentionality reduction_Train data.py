
# coding: utf-8

# In[2]:


#Imputation of data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


# Reading the csv file
train=pd.read_csv("Train_UWu5bXk.csv")
train.head()


# In[4]:


train.isnull().sum()


# In[5]:


# Imputation of the values
train["Outlet_Size"]=train["Outlet_Size"].fillna("Medium")
train["Outlet_Establishment_Year"]=train["Outlet_Establishment_Year"].fillna(1985)
mean1=round(train["Item_Weight"].mean(),1)
train["Item_Weight"]=train["Item_Weight"].fillna(mean1)
train


# ## Backward feature elimination

# In[7]:


# drop the target variable
df=train.drop(["Item_Outlet_Sales"],1)


# In[10]:


df=train.drop(['Item_Identifier', 'Outlet_Identifier'], axis=1)
df.head()


# In[11]:


# converting string values to numeric
df=pd.get_dummies(df)


# In[12]:


df.head()


# In[13]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import datasets
lreg = LinearRegression()
rfe = RFE(lreg,4)
rfe = rfe.fit_transform(df, train.Item_Outlet_Sales)
# (RFE=Recursive feature elimination)


# In[14]:


rfe


# ## Forward Feature Selection

# In[16]:


from sklearn.feature_selection import f_regression
ffs = f_regression(df,train.Item_Outlet_Sales )


# In[17]:


# This returns an array containing the F-values of the variables and the p-values corresponding to each F value. 
ffs


# In[18]:


# selecting the variables having F-value greater than 10:
variable = [ ]
for i in range(0,len(df.columns)-1):
    if ffs[0][i] >=10:
       variable.append(df.columns[i])


# In[19]:


# top most variables based on the forward feature selection algorithm.
variable


# ## Factor Analysis

# In[20]:


from factor_analyzer import FactorAnalyzer
fa = FactorAnalyzer()


# In[28]:


fa.analyze(train, 3, rotation=None)


# In[29]:


fa.loadings


# In[30]:


fa.get_uniqueness()
# we have to select which feature have the heighest uniqueness value that feature is the first importent variable


# ## Principle component Analysis

# In[31]:


from sklearn.decomposition import PCA
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(df)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3','principal component 4'])


# In[32]:


principalDf


# In[33]:


finalDf = pd.concat([principalDf, df[['Item_Outlet_Sales']]], axis = 1)


# In[35]:


finalDf.head()


# In[36]:


pca.explained_variance_ratio_


# In[37]:


plt.plot(range(4), pca.explained_variance_ratio_)
plt.plot(range(4), np.cumsum(pca.explained_variance_ratio_))
plt.title("Component-wise and Cumulative Explained Variance")
# in the output:
# blue line represents component-wise explained variance
# orange line represents the cumulative explained variance
# We are able to explain around 99% variance in the dataset using 4 components

