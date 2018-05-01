
# coding: utf-8

# 考虑以下增加特征：
# 
# 1、用户的兴趣id数目，kw数目和topic数目，
# 
# 2、同一条广告向相同用户推送的次数，#此条不需考虑，用户的重复出现次数比较少
# 
# 3、同一个用户收到的推送次数
# 
# 广告特征：
# 
# 1、广告的出现次数,被点击次数，转化率
# 
# 2、广告商家所

# In[246]:


import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb
import lightgbm as lgb
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[247]:


data = pd.read_csv('./datasets/train_data.csv')
target = pd.read_csv('./datasets/train_target.csv', names=['label'], header=None)

data.drop(data.columns[[0]], axis=1, inplace=True)  #删除某列函数
target = target.reset_index(drop=True)


# In[248]:


# data.drop("interest3", axis=1, inplace=True)
# data.drop("interest4", axis=1,inplace=True)
# data.drop("kw3", axis=1,inplace=True)
# data.drop("topic3", axis=1,inplace=True)
# data.drop('appIdInstall', axis=1,inplace=True)
# data.drop('appIdAction', axis=1,inplace=True)

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']

for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])


# In[249]:


interests = data[['interest1', 'interest2', 'interest3', 'interest4', 'interest5']]
array = np.array(interests)
num_interest = []
for i in range(array.shape[0]):
    num = 0
    inter = array[i]
    for j in inter:
        inter_lis = j.split(' ')
        if inter_lis[0] == '-1':
            continue
        num += len(inter_lis)
    num_interest.append(num)


# In[250]:


num_interests = pd.DataFrame(num_interest, columns=['num_interests'])
num_interests.head()


# In[251]:


kws = data[['kw1', 'kw2', 'kw3']]
array = np.array(kws)
num_kw = []
for i in range(array.shape[0]):
    num = 0
    kw = array[i]
    for j in kw:
        kw_lis = j.split(' ')
        if kw_lis[0] == '-1':
            continue
        num += len(kw_lis)
    num_kw.append(num)


# In[252]:


num_kws = pd.DataFrame(num_kw, columns=['num_kws'])
num_kws[:6]


# In[253]:


topics = data[['topic1','topic2','topic3']]
array = np.array(topics)
num_topic = []
for i in range(array.shape[0]):
    num = 0
    topic = array[i]
    for j in topic:
        topic_lis = j.split(' ')
        if topic_lis[0] == '-1':
            continue
        num += len(topic_lis)
    num_topic.append(num)


# In[254]:


num_topics = pd.DataFrame(num_topic, columns=['num_topics'])
num_topics.head()


# In[255]:


num_ad = data['aid'].value_counts()


# In[256]:


aid = num_ad.index.values
values = num_ad.values


# In[257]:


num_ads = pd.DataFrame({
    'aid' : aid,
    'num_ads' : values
})


# In[258]:


data_with_num = pd.concat([data, num_interests, num_kws, num_topics], axis=1)


# In[259]:


data_with_num = pd.merge(data_with_num, num_ads, on=['aid'], how='left')


# In[260]:


data_with_num.head()


# In[175]:


data_with_num['num_kws'].value_counts()


# In[176]:


data_with_num['num_topics'].value_counts()


# In[265]:


data_combined = pd.concat([data_with_num, target], axis=1, join_axes=[data_with_num.index])
data_clicked = data_combined[data_combined['label']==1]


# In[285]:


num_ads2 = num_ads.sort_index(by='aid')
sort_values = num_ads2.values


# In[283]:


num_ad_clicked = num_ad = data_clicked['aid'].value_counts()
num_ad_clicked2 = num_ad_clicked.sort_index()


# In[298]:


click_values = num_ad_clicked2.values
ratio = click_values / sort_values[:, 1]


# In[304]:


ratio_clicked = pd.DataFrame({
    'aid': num_ad_clicked2.index,
    'ratio_clicked' : ratio
})


# In[306]:


data_with_num = pd.merge(data_with_num, ratio_clicked, on=['aid'], how='left')


# In[308]:


data_combined = pd.concat([data_with_num, target], axis=1, join_axes=[data_with_num.index])
data_clicked = data_combined[data_combined['label']==1]


# In[339]:


num_user_getad = data['uid'].value_counts()


# In[342]:


num_user_clicked = data_clicked['uid'].value_counts()

