# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
 相比model_fea_add2，只保留了增加的广告出现次数特征，同时进一步提取了用户兴趣和主题的特征，其余无变化，主要
 提取了interest1, 2, 5, topic1，其余未提取因为大部分比值为零而且本地测试效果下降
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


# 增加每个广告推送给不同的用户数

print('开始加入广告推送给不同用户的数特征')

num_advertise_touser = train.groupby('aid').uid.nunique()
num_advertise_touser = pd.DataFrame({
    'aid': num_advertise_touser.index,
    'num_advertise_touser' : num_advertise_touser.values
})
data = pd.merge(data, num_advertise_touser, on=['aid'], how='left')

# 增加各种兴趣ID，和主题ID的比例值


def get_common_interest(type_name, ratio):
    num_adid = data_clicked['aid'].value_counts().sort_index().index
    num_aid_clicked = dict(data_clicked['aid'].value_counts().sort_index())
    num_user_clicksameAd_interest = data_clicked.groupby('aid')[type_name].value_counts()
    dict_interest = {}
    for adid in num_adid:
        dict_buf = {}
        for interest in num_user_clicksameAd_interest.items():
            index = interest[0]
            if index[0] == adid:
                number = interest[1]
                detail = index[1]
                detail = detail.split(' ')
                for det in detail:
                    if det not in dict_buf:
                        dict_buf[det] = number
                    else:
                        dict_buf[det] += number
        dict_interest[adid] = dict_buf
    dict_common_interest = []
    for adid, dict_inter in dict_interest.items():
        dict_common_buf = {}
        dict_common_buf['aid'] = adid
        common_inter = []
        ad_total = num_aid_clicked[adid] - dict_inter.get('-1', 0)
        if '-1' in dict_inter:
            dict_inter.pop('-1')
        for id_inter, num in dict_inter.items():
            if num >= ad_total*ratio:
                common_inter.append(id_inter)
        str_name = 'common_'+type_name
        dict_common_buf[str_name] = common_inter
        dict_common_interest.append(dict_common_buf)
    return dict_common_interest


# 获取相同的兴趣ID2
print('开始加入兴趣ID2')
dict_common_interest2 = get_common_interest('interest2', 0.25)
df_common_interest2 = pd.DataFrame(dict_common_interest2)
data = pd.merge(data, df_common_interest2, on=['aid'], how='left')
data['num_common_interest2'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest2', 'common_interest2']].values]

# 获取相同的兴趣ID1
print('开始加入兴趣ID1')
dict_common_interest1 = get_common_interest('interest1', 0.25)
df_common_interest1 = pd.DataFrame(dict_common_interest1)
data = pd.merge(data, dict_common_interest1, on=['aid'], how='left')
data['num_common_interest1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest1', 'common_interest1']].values]

# 获取相同的兴趣ID5
print('开始加入兴趣ID5')
dict_common_interest5 = get_common_interest('interest5', 0.25)
df_common_interest5 = pd.DataFrame(dict_common_interest5)
data = pd.merge(data, dict_common_interest5, on=['aid'], how='left')
data['num_common_interest5'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['interest5', 'common_interest5']].values]

# 获取相同的主题1
print('开始加入主题1')
dict_common_topic1 = get_common_interest('topic1', 0.1)
df_common_topic1 = pd.DataFrame(dict_common_topic1)
data = pd.merge(data, df_common_topic1, on=['aid'], how='left')
data['num_common_topic1'] = [len(set(i.split(' ')).intersection(set(j))) / (len(j)+1) for i, j in data[['topic1', 'common_topic1']].values]


# 分离测试集
data = data.fillna(0)
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
print('归一化...')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data[['num_advertise_touser', 'num_common_interest1', 'num_common_interest2',
                 'num_common_interest5', 'num_common_topic1']].values)
train_x = scaler.transform(train[['num_advertise_touser', 'num_common_interest1', 'num_common_interest2',
                                  'num_common_interest5', 'num_common_topic1']].values)

test_x = scaler.transform(test[['num_advertise_touser', 'num_common_interest1', 'num_common_interest2',
                                'num_common_interest5', 'num_common_topic1']].values)
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

ct_encoder = CountVectorizer(min_df=0.0009)
for feature in vector_feature:
    ct_encoder.fit(data[feature])
    train_a = ct_encoder.transform(train[feature])
    test_a = ct_encoder.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a))
    test_x = sparse.hstack((test_x, test_a))
print('cv prepared !')
# print('ths shape of train data:', test_x.shape)

sparse.save_npz('./datasets/model_fea_add3_train.npz', train_x)
sparse.save_npz('./datasets/model_fea_add3_test.npz', test_x)


def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=5000, objective='binary',
        subsample=0.7, colsample_bytree=0.8, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./datasets/submission_model_fea_add3.csv', index=False)
    return clf


model = LGB_predict(train_x,train_y,test_x,res)
joblib.dump(model, './datasets/model_fea_add3.model')

