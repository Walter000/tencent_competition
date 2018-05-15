import numpy as np
import pandas as pd
import os
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle



one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility', 'ct', 'education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']


def get_raw_data(data_path, target_path):
    data = pd.read_csv(data_path)
    target = pd.read_csv(target_path, names=['label'], header=None)
    target = target.values.squeeze()
    data.drop(data.columns[[0]], axis=1, inplace=True)  # 删除某列函数
    return data, target


def get_train_data(raw_data, label):
    result = {}
    ffm = pd.DataFrame()
    feature_size = []
    for col in one_hot_feature:
        idx = 0
        col_value = raw_data[col].unique()
        feature_size.append(len(col_value))
        feat_dict = dict(zip(col_value, range(idx, len(col_value))))
        se = raw_data[col].apply(lambda x: feat_dict[x])
        ffm = pd.concat([ffm, se], axis=1)
    result['index'] = ffm.values.tolist()
    result['label'] = label
    result['feature_size'] = feature_size
    # pickle.dump(ffm.values.tolist(), open('./data/all_data_index', 'wb'))
    # pickle.dump(feature_size, open('./data/feature_size', 'wb'))
    pickle.dump(result, open('./data/train_data', 'wb'))


def get_train_interest(raw_data):
    feature_size_interest = []
    dict_all_interest = {}
    ct_encoder = CountVectorizer(min_df=0.001)
    for index, col in enumerate(vector_feature):
        ct_encoder.fit(raw_data[col])
        col_value = ct_encoder.vocabulary_
        feature_size_interest.append(len(col_value)+1)
        all_se = []
        for m in raw_data[col]:
            ses = set()
            for n in m.split(' '):
                if n in col_value:
                    se = col_value[n]
                else:
                    se = len(col_value)
                ses.add(se)
            all_se.append(list(ses))
        dict_all_interest[index] = all_se
    dict_all_interest['feature_size_interest'] = feature_size_interest
    pickle.dump(dict_all_interest, open('./data/train_data_interest', 'wb'))


print('read the raw data...')
data, target = get_raw_data('./data/train_data.csv', './data/train_target.csv')

print('get the feature index...')
get_train_data(data, target)

print('get the interest index...')
get_train_interest(data)

