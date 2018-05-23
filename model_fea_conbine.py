# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
 模型训练结束后，将lightgbm的叶子结点onehot处理，再结合原来的特征重新训练得到最终结果
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


predict = pd.read_csv('./datasets/test2.csv')
res = predict[['aid', 'uid']]

train_x = sparse.load_npz('./datasets/train.npz')
test_x = sparse.load_npz('./datasets/test.npz')
train_y = pd.read_csv('./datasets/train_label.csv', names=['label'])  # 直接读入label文件，省去了读取train源文件时间
train_y[train_y == -1] = 0

def LGB_train(train_x, train_y):
    print("LGB test")
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=10, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=30, objective='binary', scale_pos_weight=3,
        subsample=0.79, colsample_bytree=0.79, subsample_freq=1,
        learning_rate=0.07, min_child_weight=70, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc',early_stopping_rounds=100)
    return clf


def LGB_retrain(train_x, train_y, test_x, res):
    clf = lgb.LGBMClassifier(
        boosting_type='gbdt', num_leaves=90, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=5000, objective='binary', scale_pos_weight=3,
        subsample=0.79, colsample_bytree=0.79, subsample_freq=1,
        learning_rate=0.07, min_child_weight=70, random_state=2018, n_jobs=-1
    )
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y)], eval_metric='auc', early_stopping_rounds=60)
    res['score'] = clf.predict_proba(test_x)[:, 1]
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('./submissions/submission_model_add5_combine_leaf_5000_test2.csv', index=False)
    return clf

clf = LGB_train(train_x, train_y)

# get new feature and onehot

print("get new feature")
train_new = clf.apply(train_x)
test_new = clf.apply(test_x)
one = OneHotEncoder()
one.fit(train_new)
train_new = one.transform(train_new)
test_new = one.transform(test_new)

# combine with origin featurees

print("combine new feature")
# train_new = sparse.load_npz('./datasets/model_crossfea_origin_train_fea10_tree30.npz')
# test_new = sparse.load_npz('./datasets/model_crossfea_origin_test_fea10_tree30.npz')
X_train_new = sparse.hstack((train_x, train_new))
X_test_new = sparse.hstack((test_x, test_new))

print("saving new feature")
sparse.save_npz('./datasets/model_fea_add5_fea10_tree30_train_test2.npz', train_new)
sparse.save_npz('./datasets/model_fea_add5_fea10_tree30_test_test2.npz', test_new)
del train_new
del test_new
del train_x
del test_x
import gc
gc.collect()

# retrain and predict

# from sklearn.model_selection import train_test_split

# train_x_real, valid_x_real, train_y_real, valid_y_real = train_test_split(X_train_new, train_y, random_state=2018, test_size=0.3)

print('retrain the model with new feature...')

# X_train_new = sparse.load_npz('./datasets/model_crossfea_train.npz')
# X_test_new = sparse.load_npz('./datasets/model_crossfea_test.npz')
model = LGB_retrain(X_train_new, train_y, X_test_new, res)
joblib.dump(model, './model/model_add5_combine_leaf_5000_test2.model')

model = joblib.load('./datasets/model_combine_leaf_add5_5000.model')

res['score'] = model.predict_proba(X_test_new)[:, 1]
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./datasets/submission_model_combine_leaf_add5_5000_test2.csv', index=False)