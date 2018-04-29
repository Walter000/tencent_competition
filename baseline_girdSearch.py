# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
 在作者Baseline代码基础上增加了网格搜索，同时直接读取sparse格式的训练数据，最后保存预测的结果和最好的模型best_estimator,
 同时将feature_importance写入文件方面后期分析
"""
import pandas as pd
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.externals import joblib
from scipy import sparse
from sklearn.model_selection import RandomizedSearchCV
import os

# ad_feature=pd.read_csv('./datasets/adFeature.csv')
# if os.path.exists('./datasets/userFeature.csv'):
#     user_feature = pd.read_csv('./datasets/userFeature.csv')
# else:
#     userFeature_data = []
#     with open('./datasets/userFeature.data', 'r') as f:
#         for i, line in enumerate(f):
#             line = line.strip().split('|')
#             userFeature_dict = {}
#             for each in line:
#                 each_list = each.split(' ')
#                 userFeature_dict[each_list[0]] = ' '.join(each_list[1:])
#             userFeature_data.append(userFeature_dict)
#             if i % 100000 == 0:
#                 print(i)
#         user_feature = pd.DataFrame(userFeature_data)
#         user_feature.to_csv('./datasets/userFeature.csv', index=False)
#
# print('data preprocessing finish')
train=pd.read_csv('./datasets/train.csv')
predict=pd.read_csv('./datasets/test1.csv')
train.loc[train['label']==-1,'label']=0
# predict['label']=-1
# data=pd.concat([train,predict])
# data=pd.merge(data,ad_feature,on='aid',how='left')
# print('merge ad_feature')
# print(data[:5])
# data=pd.merge(data,user_feature,on='uid',how='left')
# print('merge user_feature')

# data=data.fillna('-1')
# one_hot_feature=['LBS','age','carrier','consumptionAbility','education','gender','house','os','ct','marriageStatus','advertiserId','campaignId', 'creativeId',
#        'adCategoryId', 'productId', 'productType']
# vector_feature=['appIdAction','appIdInstall','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3']
# for feature in one_hot_feature:
#     try:
#         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
#     except:
#         data[feature] = LabelEncoder().fit_transform(data[feature])
#
# train=data[data.label!=-1]
train_y = train.pop('label')
# # train, test, train_y, test_y = train_test_split(train,train_y,test_size=0.2, random_state=2018)
# test=data[data.label==-1]
res = predict[['aid', 'uid']]
# test=test.drop('label',axis=1)
# enc = OneHotEncoder()
# train_x=train[['creativeSize']]
# test_x=test[['creativeSize']]

# 将原始数据集进行层次划分
# from sklearn.model_selection import StratifiedShuffleSplit
#
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
# for train_index, test_index in split.split(train, train_y):
#     train_data = train.loc[test_index]
#     train_target = train_y[test_index]
#     train_data.to_csv('./datasets/train_data.csv')
#     train_target.to_csv('./datasets/train_target.csv')

# print('start!')
# for feature in one_hot_feature:
#     enc.fit(data[feature].values.reshape(-1, 1))
#     train_a=enc.transform(train[feature].values.reshape(-1, 1))
#     test_a = enc.transform(test[feature].values.reshape(-1, 1))
#     train_x= sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('one-hot prepared !')
#
# cv = CountVectorizer()
# for feature in vector_feature:
#     cv.fit(data[feature])
#     train_a = cv.transform(train[feature])
#     test_a = cv.transform(test[feature])
#     train_x = sparse.hstack((train_x, train_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('cv prepared !')
#
# sparse.save_npz('./datasets/train.npz', train_x)
# sparse.save_npz('./datasets/test.npz', test_x)
print('start')
train_x = sparse.load_npz('./datasets/train.npz')
test_x = sparse.load_npz('./datasets/test.npz')

clf = lgb.LGBMClassifier(
        boosting_type='gbdt', max_depth=-1, objective='binary',
        subsample=0.7, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.05, random_state=2018, n_jobs=-1
    )

params = {
    'n_estimators' : np.arange(1000, 8000, 1000),
    'num_leaves' : np.arange(30, 45, 2),
    'min_child_weight' : np.arange(40, 65, 5),
    'reg_alpha' : [0.2, 0.4, 0.8, 1],
    'reg_lambda' : [0.2, 0.4, 0.8, 1],
}

rand_search = RandomizedSearchCV(clf, params, cv=5, n_iter=10, random_state=2018, verbose=2)
rand_search.fit(train_x, train_y)
best = rand_search.best_estimator_
print(rand_search.best_params_)
res['score'] = best.predict_proba(test_x)[:, 1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./datasets/submission4.csv', index=False)
fea_imp = best.feature_importances_
fea_imp = pd.Series(fea_imp)
fea_imp.to_csv('./datasets/fea_imp.csv', index=False)
# def LGB_predict(train_x,train_y,test_x,res):
#     print("LGB test")
#     clf = lgb.LGBMClassifier(
#         boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=0.9,
#         max_depth=-1, n_estimators=2000, objective='binary',
#         subsample=0.7, colsample_bytree=0.8, subsample_freq=1,
#         learning_rate=0.05, min_child_weight=60, random_state=2018, n_jobs=-1
#     )
#     clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
#
#     feat_imp = clf.feature_importances_
#     print(feat_imp)
#     res['score'] = clf.predict_proba(test_x)[:,1]
#     res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
#     res.to_csv('./datasets/submission3.csv', index=False)
#     # os.system('zip ./datasets/baseline.zip ./datasets/submission.csv')
#     return clf

# model = LGB_predict(train_x,train_y,test_x,res)
joblib.dump(best, './datasets/random_search_best.model')
