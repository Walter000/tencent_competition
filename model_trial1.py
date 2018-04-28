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

# 下面根据前面的特征分析，剔除掉缺失率过高的特征: interest3, interest4, kw3, appIdInstall, topic3,同时保留原始ConsumeAbility和education的值
ad_feature=pd.read_csv('./datasets/adFeature.csv')
user_feature = pd.read_csv('./datasets/userFeature.csv')

one_hot_feature=['LBS','age','carrier','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
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

train=data[data.label!=-1]
train_y=train.pop('label')
test=data[data.label==-1]
res=test[['aid','uid']]
test=test.drop('label',axis=1)
enc = OneHotEncoder()

train_x = train[['consumptionAbility', 'education']]
test_x = test[['consumptionAbility', 'education']]

scaler = StandardScaler()
scaler.fit(train['creativeSize'].reshape(-1, 1))
scaler_train = scaler.transform(train['creativeSize'].reshape(-1, 1))
scaler_test =  scaler.transform(test['creativeSize'].reshape(-1, 1))

train_x = np.array(train_x)
test_x = np.array(test_x)
train_x = np.hstack((train_x, scaler_train))
test_x = np.hstack((test_x, scaler_test))

oc_encoder = OneHotEncoder()
for feature in one_hot_feature:
    oc_encoder.fit(train[feature].reshape(-1, 1))
    train_a=oc_encoder.transform(train[feature].values.reshape(-1, 1))
    test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('one-hot prepared !')

# 处理count特征向量

ct_encoder = CountVectorizer(min_df=0.01)
for feature in vector_feature:
    ct_encoder.fit(train[feature])
    train_a = ct_encoder.transform(train[feature])
    test_a = ct_encoder.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')
print('ths shape of train data:', test_x.shape)


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=1200, objective='binary',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./datasets/submission2.csv', index=False)
    return clf

model = LGB_predict(train_x,train_y,test_x,res)

joblib.dump(model, './datasets/model_trial1.model')

