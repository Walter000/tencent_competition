# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
  add the follow features:
  num_advertise_touser, ratio_num_ad_toage, ratio_num_ad_toconsume
  ratio_num_ad_tohouse, ratio_clicked
  other features stay the same with model_fea_add_3
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
# ad_feature=pd.read_csv('./datasets/adFeature.csv')
# user_feature = pd.read_csv('./datasets/userFeature.csv')
#
# one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
#        'adCategoryId', 'productId', 'productType']
# vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']
#
# train = pd.read_csv('./datasets/train.csv')
# predict = pd.read_csv('./datasets/test1.csv')
# train.loc[train['label']==-1,'label']=0
# predict['label']=-1
# data = pd.concat([train,predict])
# data = pd.merge(data,ad_feature,on='aid',how='left')
# data = pd.merge(data,user_feature,on='uid',how='left')
# data = data.fillna('-1')
#
# print('start!')
# for feature in one_hot_feature:
#     try:
#         data[feature] = LabelEncoder().fit_transform(data[feature].apply(int))
#     except:
#         data[feature] = LabelEncoder().fit_transform(data[feature])
#
# train = data[data.label != -1]
# data_clicked = train[train['label'] == 1]
#
#
# # 增加每个广告推送给不同的用户数
#
# print('开始加入广告推送给不同用户的数特征')
#
# num_advertise_touser = train.groupby('aid').uid.nunique()
# num_advertise_touser = pd.DataFrame({
#     'aid': num_advertise_touser.index,
#     'num_advertise_touser' : num_advertise_touser.values
# })
# data = pd.merge(data, num_advertise_touser, on=['aid'], how='left')
#
# # 增加广告点击率特征
#
# print('开始加入广告点击率特征')
#
# num_ad = train['aid'].value_counts().sort_index()
# num_ad_clicked = data_clicked['aid'].value_counts().sort_index()
#
# ratio = num_ad_clicked / num_ad
#
# ratio_clicked = pd.DataFrame({
#     'aid': ratio.index,
#     'ratio_clicked' : ratio.values
# })
# data = pd.merge(data, ratio_clicked, on=['aid'], how='left')
#
# # 增加推广计划点击率特征
#
# print('开始加入推广计划点击率特征')
# num_campaign = train['campaignId'].value_counts().sort_index()
# num_campaign_clicked = data_clicked['campaignId'].value_counts().sort_index()
#
# ratio_num_campaign = num_campaign_clicked / num_campaign
#
# ratio_num_campaign = pd.DataFrame({
#     'campaignId': ratio_num_campaign.index,
#     'ratio_num_campaign' : ratio_num_campaign.values
# })
#
# data = pd.merge(data, ratio_num_campaign, on=['campaignId'], how='left')
#
# # 增加各种兴趣ID，和主题ID的比例值
#
# def get_common_interest(type_name, ratio):
#     num_adid = data_clicked['aid'].value_counts().sort_index().index
#     num_aid_clicked = dict(data_clicked['aid'].value_counts().sort_index())
#     num_user_clicksameAd_interest = data_clicked.groupby('aid')[type_name].value_counts()
#     dict_interest = {}
#     for adid in num_adid:
#         dict_buf = {}
#         for interest in num_user_clicksameAd_interest.items():
#             index = interest[0]
#             if index[0] == adid:
#                 number = interest[1]
#                 detail = index[1]
#                 detail = detail.split(' ')
#                 for det in detail:
#                     if det not in dict_buf:
#                         dict_buf[det] = number
#                     else:
#                         dict_buf[det] += number
#         dict_interest[adid] = dict_buf
#     dict_common_interest = []
#     for adid, dict_inter in dict_interest.items():
#         dict_common_buf = {}
#         dict_common_buf['aid'] = adid
#         common_inter = []
#         ad_total = num_aid_clicked[adid] - dict_inter.get('-1', 0)
#         if '-1' in dict_inter:
#             dict_inter.pop('-1')
#         for id_inter, num in dict_inter.items():
#             if num >= ad_total*ratio:
#                 common_inter.append(id_inter)
#         str_name = 'common_'+type_name
#         dict_common_buf[str_name] = common_inter
#         dict_common_interest.append(dict_common_buf)
#     return dict_common_interest
#
#
# # 获取相同的兴趣ID2
# print('开始加入兴趣ID2')
# dict_common_interest2 = get_common_interest('interest2', 0.25)
# df_common_interest2 = pd.DataFrame(dict_common_interest2)
# data = pd.merge(data, df_common_interest2, on=['aid'], how='left')
# data['num_common_interest2'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest2', 'common_interest2']].values]
#
# # 获取相同的兴趣ID1
# print('开始加入兴趣ID1')
# dict_common_interest1 = get_common_interest('interest1', 0.25)
# df_common_interest1 = pd.DataFrame(dict_common_interest1)
# data = pd.merge(data, df_common_interest1, on=['aid'], how='left')
# data['num_common_interest1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest1', 'common_interest1']].values]
#
# # 获取相同的兴趣ID5
# print('开始加入兴趣ID5')
# dict_common_interest5 = get_common_interest('interest5', 0.25)
# df_common_interest5 = pd.DataFrame(dict_common_interest5)
# data = pd.merge(data, df_common_interest5, on=['aid'], how='left')
# data['num_common_interest5'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest5', 'common_interest5']].values]
#
# # 获取相同的主题1
# print('开始加入主题1')
# dict_common_topic1 = get_common_interest('topic1', 0.1)
# df_common_topic1 = pd.DataFrame(dict_common_topic1)
# data = pd.merge(data, df_common_topic1, on=['aid'], how='left')
# data['num_common_topic1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['topic1', 'common_topic1']].values]
#
#
# # 增加广告对应的年龄分布，消费能力分布，是否有房分布
#
# def get_ad_toother(typename):
#     num_ad_totype = train.groupby('aid')[typename].value_counts()
#     num_ad_totype_clicked = data_clicked.groupby('aid')[typename].value_counts()
#     ratio_num_ad_totype = num_ad_totype_clicked / num_ad_totype
#     list_num_ad_totype = []
#     num_adid = train['aid'].value_counts().sort_index().index
#     for aid_out in num_adid:
#         dict_buf = {}
#         dict_num_ad_totype = {}
#         dict_num_ad_totype['aid'] = aid_out
#         for i, j in ratio_num_ad_totype.items():
#             aid = i[0]
#             feature = i[1]
#             if(aid == aid_out):
#                 dict_buf[feature] = float("%.5f" % j)
#         fea_name = 'num_ad_to'+typename
#         dict_num_ad_totype[fea_name] = dict_buf
#         list_num_ad_totype.append(dict_num_ad_totype)
#     return list_num_ad_totype
#
#
# print('开始加入年龄分布!')
# list_num_ad_toage = get_ad_toother('age')
# list_num_ad_toage = pd.DataFrame(list_num_ad_toage)
# data = pd.merge(data, list_num_ad_toage, on=['aid'], how='left')
# data['ratio_num_ad_toage'] = [j.get(i, 0) for i, j in data[['age', 'num_ad_toage']].values]
#
# print('开始加入消费能力分布!')
# list_num_ad_toconsume = get_ad_toother('consumptionAbility')
# list_num_ad_toconsume = pd.DataFrame(list_num_ad_toconsume)
# data = pd.merge(data, list_num_ad_toconsume, on=['aid'], how='left')
# data['ratio_num_ad_toconsume'] = [j.get(i, 0) for i, j in data[['consumptionAbility', 'num_ad_toconsumptionAbility']].values]
#
# print('开始加入是否有房分布!')
# list_num_ad_tohouse = get_ad_toother('house')
# list_num_ad_tohouse = pd.DataFrame(list_num_ad_tohouse)
# data = pd.merge(data, list_num_ad_tohouse, on=['aid'], how='left')
# data['ratio_num_ad_tohouse'] = [j.get(i, 0) for i, j in data[['house', 'num_ad_tohouse']].values]
#
# # 分离测试集
# data = data.fillna(0)
# train = data[data.label != -1]
# test = data[data.label == -1]
# res = test[['aid','uid']]
#
#
# from sklearn.model_selection import StratifiedShuffleSplit
#
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2018)
# x_train_y = train.pop('label').values
# for train_index, test_index in split.split(train, x_train_y) :
#     X_train = train.iloc[train_index]
#     X_eval = train.iloc[test_index]
#     y_train = x_train_y[train_index]
#     y_eval = x_train_y[test_index]
#
# test = test.drop('label', axis=1)
#
#
#
# # 处理联网类型特征
# ct_train = X_train['ct'].values
# ct_train = [m.split(' ') for m in ct_train]
# ct_trains = []
# for i in ct_train:
#     index = [0, 0, 0, 0, 0]
#     for j in i:
#         index[int(j)] = 1
#     ct_trains.append(index)
#
# ct_eval = X_eval['ct'].values
# ct_eval = [m.split(' ') for m in ct_eval]
# ct_evals = []
# for i in ct_eval:
#     index = [0, 0, 0, 0, 0]
#     for j in i:
#         index[int(j)] = 1
#     ct_evals.append(index)
#
# ct_test = test['ct'].values
# ct_test = [m.split(' ') for m in ct_test]
# ct_tests = []
# for i in ct_test:
#     index = [0, 0, 0, 0, 0]
#     for j in i:
#         index[int(j)] = 1
#     ct_tests.append(index)
#
#
# # 将上面新加入的特征进行归一化
# print('归一化...')
# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
#
# scaler.fit(data[['num_advertise_touser', 'ratio_clicked','num_common_interest2', 'num_common_interest1', 'num_common_interest5', 'num_common_topic1',
#                       'ratio_num_ad_toage', 'ratio_num_ad_toconsume', 'ratio_num_ad_tohouse', 'ratio_num_campaign']].values)
# train_x = scaler.transform(X_train[['num_advertise_touser', 'ratio_clicked', 'num_common_interest2', 'num_common_interest1', 'num_common_interest5',
#                       'num_common_topic1', 'ratio_num_ad_toage', 'ratio_num_ad_toconsume', 'ratio_num_ad_tohouse', 'ratio_num_campaign']].values)
#
# eval_x = scaler.transform(X_eval[['num_advertise_touser', 'ratio_clicked','num_common_interest2', 'num_common_interest1', 'num_common_interest5',
#                         'num_common_topic1', 'ratio_num_ad_toage', 'ratio_num_ad_toconsume', 'ratio_num_ad_tohouse', 'ratio_num_campaign']].values)
#
#
# test_x = scaler.transform(test[['num_advertise_touser', 'ratio_clicked','num_common_interest2', 'num_common_interest1', 'num_common_interest5',
#                         'num_common_topic1', 'ratio_num_ad_toage', 'ratio_num_ad_toconsume', 'ratio_num_ad_tohouse', 'ratio_num_campaign']].values)
#
# train_x = np.hstack((train_x, ct_trains))
# eval_x = np.hstack((eval_x, ct_evals))
# test_x = np.hstack((test_x, ct_tests))
#
#
# # 特征进行onehot处理
# enc = OneHotEncoder()
#
# oc_encoder = OneHotEncoder()
# for feature in one_hot_feature:
#     oc_encoder.fit(data[feature].values.reshape(-1, 1))
#     train_a=oc_encoder.transform(X_train[feature].values.reshape(-1, 1))
#     eval_a = oc_encoder.transform(X_eval[feature].values.reshape(-1, 1))
#     test_a = oc_encoder.transform(test[feature].values.reshape(-1, 1))
#     train_x = sparse.hstack((train_x, train_a))
#     eval_x = sparse.hstack((eval_x, eval_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('one-hot prepared !')
#
# # 处理count特征向量
#
# ct_encoder = CountVectorizer(min_df=0.0009)
# for feature in vector_feature:
#     ct_encoder.fit(data[feature])
#     train_a = ct_encoder.transform(X_train[feature])
#     eval_a = ct_encoder.transform(X_eval[feature])
#     test_a = ct_encoder.transform(test[feature])
#     train_x = sparse.hstack((train_x, train_a))
#     eval_x = sparse.hstack((eval_x, eval_a))
#     test_x = sparse.hstack((test_x, test_a))
# print('cv prepared !')
# # print('ths shape of train data:', test_x.shape)
#
# sparse.save_npz('./datasets/params_tune1_train.npz', train_x)
# sparse.save_npz('./datasets/params_tune1_test.npz', test_x)
# sparse.save_npz('./datasets/params_tune1_eval.npz', eval_x)
# train_y_csv = pd.DataFrame(y_train)
# eval_y_csv = pd.DataFrame(y_eval)
# train_y_csv.to_csv('./datasets/params_tune1_trainy.csv', index=False)
# eval_y_csv.to_csv('./datasets/params_tune1_evaly.csv', index=False)

# def LGB_predict(train_x, train_y, test_x, res):
#     print("LGB test")
#     clf = lgb.LGBMClassifier(
#         boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
#         max_depth=-1, n_estimators=8000, objective='binary',
#         subsample=0.7, colsample_bytree=0.8, subsample_freq=1,
#         learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
#     )
#     clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
#     res['score'] = clf.predict_proba(test_x)[:, 1]
#     res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
#     res.to_csv('./datasets/submission_model_fea_add4.csv', index=False)
#     return clf


# model = LGB_predict(train_x, train_y, test_x, res)

X_train = sparse.load_npz('./datasets/params_tune1_train.npz')
X_eval = sparse.load_npz('./datasets/params_tune1_eval.npz')
y_train = pd.read_csv('./datasets/params_tune1_trainy.csv')
y_eval = pd.read_csv('./datasets/params_tune1_evaly.csv')

from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK

epoch = 0


def objective(argsDict):
    global epoch

    epoch += 1
    num_leaves = argsDict["num_leaves"] * 10 + 10
    n_estimators = argsDict['n_estimators'] * 100 + 100
    learning_rate = argsDict["learning_rate"] * 0.01 + 0.05
    subsample = argsDict["subsample"] * 0.1 + 0.7
    colsample_bytree = argsDict["colsample_bytree"] * 0.1 + 0.7
    scale_pos_weight = argsDict["scale_pos_weight"] + 1
    min_child_weight = argsDict["min_child_weight"] * 5 + 5
    print('epoch: ', epoch)
    print("num_leaves:" + str(num_leaves))
    print("n_estimator:" + str(n_estimators))
    print("learning_rate:" + str(learning_rate))
    print("subsample:" + str(subsample))
    print("colsample_bytree:" + str(colsample_bytree))
    print("scale_pos_weight:" + str(scale_pos_weight))
    print("min_child_weight:" + str(min_child_weight))

    gbm = lgb.LGBMClassifier(boosting_type='gbdt',
                             objective='binary',
                             learning_rate=learning_rate,
                             num_leaves=num_leaves,
                             reg_alpha=0.0,
                             reg_lambda=1,
                             max_depth=-1,
                             n_estimators=n_estimators,
                             subsample=subsample,
                             colsample_bytree=colsample_bytree,
                             subsample_freq=1,
                             min_child_weight=min_child_weight,
                             scale_pos_weight=scale_pos_weight,
                             random_state=2018, n_jobs=-1)
    gbm.fit(X_train, y_train.values.squeeze(), eval_set=[(X_eval, y_eval.values.squeeze())], eval_metric='auc',
            early_stopping_rounds=80)
    metric = gbm.best_score_['valid_0']['auc']
    #     metric = cross_val_score(gbm, X_test_encoded2, y_train, cv=5,scoring="roc_auc").mean()
    print(metric)
    return -metric


space = {"num_leaves":hp.randint("num_leaves",10),
         "n_estimators":hp.randint("n_estimators", 10),
         "subsample":hp.randint("subsample",4),
         "learning_rate":hp.randint("learning_rate",6),
         "colsample_bytree":hp.randint("colsample_bytree",4),
         "scale_pos_weight": hp.randint("scale_pos_weight", 9),
         "min_child_weight": hp.randint("min_child_weight",15),
        }

algo = partial(tpe.suggest,n_startup_jobs=8)
best = fmin(objective,space,algo=algo,max_evals=100)

print(best)
print(objective(best))

import pickle

pickle.dump(best, open("./datasets/params_tune1_best.model", "wb"))

