# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
 训练数据中剔除了缺失率过高的特征: interest3, interest4, kw3, appIdInstall, appIdAction, topic3, 同时将除了ct(上网类型)之外的
 特征进行相同的处理，额外增加了四个组合特征
"""

import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import xgboost as xgb
import lightgbm as lgb

# 下面根据前面的特征分析，剔除掉缺失率过高的特征: interest3, interest4, kw3, appIdInstall, topic3
ad_feature=pd.read_csv('./datasets/adFeature.csv')
user_feature = pd.read_csv('./datasets/userFeature.csv')

one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']

train = pd.read_csv('./datasets/train.csv')
predict = pd.read_csv('./datasets/test1.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data = pd.concat([train,predict])
data = pd.merge(data,ad_feature,on='aid',how='left')
data = pd.merge(data,user_feature,on='uid',how='left')
data = data.fillna('-1')

print('start!')
for feature in one_hot_feature:
    try:
        data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
    except:
        data[feature] = LabelEncoder().fit_transform(data[feature])

train = data[data.label != -1]
data_clicked = train[train['label'] == 1]


# 增加广告点击率特征

print('开始加入广告点击率特征')

num_ad = train['aid'].value_counts().sort_index()
num_ad_clicked = data_clicked['aid'].value_counts().sort_index()

ratio = num_ad_clicked / num_ad

ratio_clicked = pd.DataFrame({
    'aid': ratio.index,
    'ratio_clicked' : ratio.values
})
data = pd.merge(data, ratio_clicked, on=['aid'], how='left')

# 增加每个广告推送给不同的用户数

print('开始加入广告推送给不同用户的数特征')

num_advertise_touser = train.groupby('aid').uid.nunique()
num_advertise_touser = pd.DataFrame({
    'aid': num_advertise_touser.index,
    'num_advertise_touser' : num_advertise_touser.values
})
data = pd.merge(data, num_advertise_touser, on=['aid'], how='left')

# 加入推广计划转化率

print('开始加入推广计划转化率特征')

num_campaign = train['campaignId'].value_counts().sort_index()
num_campaign_clicked = data_clicked['campaignId'].value_counts().sort_index()
ratio_num_campaign = num_campaign_clicked / num_campaign
ratio_num_campaign = pd.DataFrame({
    'campaignId': ratio_num_campaign.index,
    'ratio_num_campaign' : ratio_num_campaign.values
})

data = pd.merge(data, ratio_num_campaign, on=['campaignId'], how='left')


# 加入学历所对应转化率

print('开始加入学历所对应转化率特征')

num_education = train['education'].value_counts().sort_index()
num_education_clicked = data_clicked['education'].value_counts().sort_index()
ration_num_education = num_education_clicked / num_education
ration_num_education = pd.DataFrame({
    'education': ration_num_education.index,
    'ration_num_education' : ration_num_education.values
})
data = pd.merge(data, ration_num_education, on=['education'], how='left')

# 分离测试集
train = data[data.label != -1]
test = data[data.label == -1]
res = test[['aid','uid']]
test = test.drop('label', axis=1)
train_y = train.pop('label')

# 处理联网类型特征
ct_train = train['ct'].values
ct_train = [m.split(' ') for m in ct_train]
ct_trains = []
for i in ct_train:
    index = [0, 0, 0, 0, 0]
    for j in i:
        index[int(j)] = 1
    ct_trains.append(index)

ct_test = test['ct'].values
ct_test = [m.split(' ') for m in ct_test]
ct_tests = []
for i in ct_test:
    index = [0, 0, 0, 0, 0]
    for j in i:
        index[int(j)] = 1
    ct_tests.append(index)


# 将上面新加入的特征进行归一化

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train[['ratio_clicked', 'num_advertise_touser', 'ratio_num_campaign',
                          'ration_num_education']].values)
train_x = scaler.transform(train[['ratio_clicked', 'num_advertise_touser', 'ratio_num_campaign',
                                                  'ration_num_education']].values)

test_x = scaler.transform(test[['ratio_clicked', 'num_advertise_touser', 'ratio_num_campaign',
                                                  'ration_num_education']].values)
train_x = np.hstack((train_x, ct_trains))
test_x = np.hstack((test_x, ct_tests))


# 特征进行onehot处理
enc = OneHotEncoder()

oc_encoder = OneHotEncoder()
for feature in one_hot_feature:
    oc_encoder.fit(data[feature].values.reshape(-1, 1))
    train_a=oc_encoder.transform(train[feature].values.reshape(-1, 1))
    test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

# 处理count特征向量

ct_encoder = CountVectorizer(min_df=0.001)
for feature in vector_feature:
    ct_encoder.fit(data[feature])
    train_a = ct_encoder.transform(train[feature])
    test_a = ct_encoder.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')
# print('ths shape of train data:', test_x.shape)

sparse.save_npz('./datasets/model_fea_add1_train.npz', train_x)
sparse.save_npz('./datasets/model_fea_add1_test.npz', test_x)


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=5000, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./datasets/submission_model_fea_add1.csv', index=False)
    return clf


model = LGB_predict(train_x,train_y,test_x,res)
joblib.dump(model, './datasets/model_fea_add1.model')

