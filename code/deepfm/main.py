# -*- coding:utf-8 -*-

import DeepFm_padded
import torch
import torch.nn as nn
from torch.autograd import Variable

# result_dict = data_preprocess.read_criteo_data('./data/tiny_train_input.csv', './data/category_emb.csv')
# test_dict = data_preprocess.read_criteo_data('./data/tiny_test_input.csv', './data/category_emb.csv')
#
# deepfm = AFM.AFM(39, result_dict['feature_sizes'], verbose=True, use_cuda=False, weight_decay=0.0001, use_fm=True,
#                        use_ffm=False)
# deepfm.fit(result_dict['index'], result_dict['value'], result_dict['label'],
#                test_dict['index'], test_dict['value'], test_dict['label'],  ealry_stopping=True, refit=True)

import pickle

# 获取数据

print('get data...')
data_interest = pickle.load(open('./data/train_data_interest', 'rb'))
train_data = pickle.load(open('./data/train_data', 'rb'))

print('split data...')
label = train_data['label']
# print(label[:100])
feature_size = train_data['feature_size']
all_data_index = train_data['index']
feature_size_interest = data_interest['feature_size_interest']
values = []
for i in range(len(label)):
    value = [1 for i in range(len(feature_size))]
    values.append(value)
#
# # 分离兴趣
interests = {}
for i in range(7):
    interests[i] = data_interest[i]

train_interest = {}
test_interest = {}

train_size = round(len(label) * 0.7)
train_index = all_data_index[:train_size]
train_values = values[:train_size]
train_label = label[:train_size]
test_index = all_data_index[train_size:]
test_values = values[train_size:]
test_label = label[train_size:]
for i in range(7):
    train_interest[i] = data_interest[i][:train_size]
    test_interest[i] = data_interest[i][train_size:]

print('first data index: ', train_index[4])
print('first data interest1: ', train_interest[0][28])


print('start train...')
deepfm = DeepFm_padded.DeepFM(len(feature_size), feature_size, feature_size_interest, embedding_size=8, verbose=True, use_cuda=False, weight_decay=0.0001, use_fm=True,
                       batch_size=1024, n_epochs=10, use_ffm=False, use_deep=True)
deepfm.fit(train_index, train_values, train_interest, train_label,
           test_index, test_values, test_interest, test_label,  ealry_stopping=True, refit=True, save_path='./data')

import pandas as pd
res = pd.DataFrame()
results = deepfm.predict_proba(test_index, test_values, test_interest)
res['score'] = results
res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
res.to_csv('./data/submission_model_deepfm.csv', index=False)


# v = torch.LongTensor([[0, 1], [2, 3], [1, 3]])
# input_lengths = max([len(s) for s in v])
# print(input_lengths)
# interest_embed = nn.Embedding(4, 1)
# print(interest_embed(v))
# print(torch.sum(interest_embed(v), 1))
# all_embel = []
# for inter in v:
#     buf = torch.sum(interest_embed(torch.LongTensor(inter)), 0).unsqueeze(0)
#     # buf = interest_embed(torch.LongTensor(inter))
#     print(buf)
#     all_embel.append(buf)


# print(torch.cat(all_embel, 0))




