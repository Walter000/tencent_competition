# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
  将统计特征单独保存为字典后读入，然后读入原始切分好的训练，验证和测试的sparse文件，在代码里进行组合，并结合训练
  好的embedding文件，放到一起训练
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

print('read raw npz data...')
one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']

train_x = sparse.load_npz('./datasets/model_test2_raw_split_train.npz')
valid_x = sparse.load_npz('./datasets/model_test2_raw_split_valid.npz')
test_x = sparse.load_npz('./datasets/model_test2_raw_split_test.npz')
all_y = pd.read_csv('./datasets/train_label.csv', names=['label'])  # 直接读入label文件，省去了读取train源文件时间
all_y[all_y == -1] = 0
predict = pd.read_csv('./datasets/test2.csv')
res = predict[['aid', 'uid']]

print('read raw csv data...')
train_raw_data = pd.read_csv('./datasets/train_raw_data.csv')
test_raw_data = pd.read_csv('./datasets/test2_raw_data.csv')

import pickle
print('load new feature...')

train_size = round(0.75 * len(all_y))

valid_y = all_y.iloc[train_size:]
train_y = all_y.iloc[:train_size]
new_feature_train = pickle.load(open('./datasets/new_feature_train', 'rb'))
new_feature_test = pickle.load(open('./datasets/new_feature_test', 'rb'))

print('add new feature to train and valid...')

for k, v in new_feature_train.items():
    print('process: ', k)
    train_x = sparse.hstack((train_x, np.array(v)[:train_size].reshape(-1, 1)))
    valid_x = sparse.hstack((valid_x, np.array(v)[train_size:].reshape(-1, 1)))

print('add new feature to test...')

for k, v in new_feature_test.items():
    print('process: ', k)
    test_x = sparse.hstack((test_x, np.array(v).reshape(-1, 1)))

import gc

del new_feature_test
del new_feature_train
gc.collect()

print('add embedding feature...')

def get_vector(type_name):
    vector = {}
    path = './datasets/word_vector/' + 'vectors_'+type_name+'_10.txt'
    with open(path) as f:
        for line in f:
            line = line.strip().split(' ')
            vector[line[0]] = [float(m) for m in line[1:]]
    return vector

vector_interest1 = get_vector('interest1')
vector_interest2 = get_vector('interest2')
vector_interest5 = get_vector('interest5')
vector_kw1 = get_vector('kw1')
vector_kw2 = get_vector('kw2')
vector_topic1 = get_vector('topic1')
vector_topic2 = get_vector('topic2')

all_vectors = dict()
all_vectors['interest1'] = vector_interest1
all_vectors['interest2'] = vector_interest2
all_vectors['interest5'] = vector_interest5
all_vectors['kw1'] = vector_kw1
all_vectors['kw2'] = vector_kw2
all_vectors['topic1'] = vector_topic1
all_vectors['topic2'] = vector_topic2

print('start transforming...')
train_interest_embed = {}
test_interest_embed = {}
for count in vector_feature:
    print('processing feature: ', count)
    vector_dict = all_vectors[count]
    train_vector = []
    test_vector = []
    for inter in train_raw_data[count].values:
        inter = inter.split(' ')
        vector_sum = np.zeros(shape=(10,))
        for i in inter:
            vector_sum += np.array(vector_dict.get(i, 0))
        train_vector.append(vector_sum)

    for inter in test_raw_data[count].values:
        inter = inter.split(' ')
        vector_sum = np.zeros(shape=(10,))
        for i in inter:
            vector_sum += np.array(vector_dict.get(i, 0))
        test_vector.append(vector_sum)
    train_interest_embed[count] = train_vector
    test_interest_embed[count] = test_vector

print('start combine the embedding feature...')

for count in vector_feature:
    train_x = sparse.hstack((train_x, train_interest_embed[count][:train_size]))
    valid_x = sparse.hstack((valid_x, train_interest_embed[count][train_size:]))
    test_x = sparse.hstack((test_x, test_interest_embed[count]))

print('the shape of test is: ', test_x.shape)

del train_raw_data
del test_raw_data
gc.collect()

def LGB_predict(train_x, train_y, test_x, res):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=64, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=10000, objective='binary',
        subsample=0.79, colsample_bytree=0.79, subsample_freq=1,
        learning_rate=0.05, min_child_weight=50, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(valid_x, valid_y)], eval_metric='auc', early_stopping_rounds=200)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./submissions/submission_model_fea_add6_with_embedding_.csv', index=False)
    return clf


model = LGB_predict(train_x, train_y, test_x, res)
joblib.dump(model, './model/model_fea_add_6_with_embedding.model')

