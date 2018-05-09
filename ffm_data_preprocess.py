# coding=utf-8
# @author:walter000
# github: https://github.com/Walter000/tencent_competition

"""
 训练ffm模型的数据预处理，格式为 label field:feature:value
 处理原始数据集，同时去除缺失率过高的特征，训练框架为xlearn
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

ad_feature=pd.read_csv('../datasets/adFeature.csv')
user_feature=pd.read_csv('../datasets/userFeature.csv')

train=pd.read_csv('../datasets/train.csv')
predict=pd.read_csv('../datasets/test1.csv')
train.loc[train['label']==-1,'label']=0
predict['label']=-1
data=pd.concat([train,predict])
data=pd.merge(data,ad_feature,on='aid',how='left')
data=pd.merge(data,user_feature,on='uid',how='left')
data=data.fillna('-1')
one_hot_feature=['creativeSize', 'LBS','age','carrier','consumptionAbility','education','gender','house','os','marriageStatus','advertiserId','campaignId', 'creativeId',
       'adCategoryId', 'productId', 'productType']
vector_feature=['interest1','interest2','interest5','kw1','kw2','topic1','topic2']

print('merging finished!')
# processing one hot feature
idx = 0
field_dict = dict(zip(one_hot_feature,range(len(one_hot_feature))))
ffm = pd.DataFrame()
for col in one_hot_feature:
    col_value = data[col].unique()
    feat_dict = dict(zip(col_value, range(idx,idx+len(col_value))))
    se = data[col].apply(lambda x: "{0}:{1}:{2}".format(field_dict[col]+1,feat_dict[x], 1))
    ffm = pd.concat([ffm,se],axis=1)
    idx += len(col_value)

print('processing one-hot feature finished!')

# processing count vector feature

field_index = len(one_hot_feature)
field_dict2 = dict(zip(vector_feature,range(len(vector_feature))))
ct_encoder = CountVectorizer(min_df=0.0009)
for col in vector_feature:
    ct_encoder.fit(data[col])
    col_value =  ct_encoder.vocabulary_
    feat_dict = {}
    for k, v in col_value.items():
        feat_dict[k] = v+idx
    all_se = []
    for m in data[col]:
        ses = str()
        buf = {}
        for n in m.split(' '):
            if n in feat_dict:
                se = "{0}:{1}:{2}".format(field_dict2[col]+field_index+1,feat_dict[n], 1)
            ses = ses + se + ' '
        buf[col] = ses
        all_se.append(buf)
    final = pd.DataFrame(all_se)
    ffm = pd.concat([ffm,final],axis=1)
    idx += len(col_value)

print('processing vector feature finished!')


# processing ct feature

ct_train = data['ct'].values
ct_train = [m.split(' ') for m in ct_train]
dict_ct = {'0':0, '1':1, '2':2, '3':3, '4':4}
all_ct = []
for i in ct_train:
    buf_dict = {}
    ses = str()
    for j in i:
        se = "{0}:{1}:{2}".format(len(field_dict)+len(field_dict2)+1,dict_ct[j]+idx, 1)
        ses = ses + se + ' '
    buf_dict['ct'] = ses
    all_ct.append(buf_dict)

all_ct = pd.DataFrame(all_ct)
ffm = pd.concat([ffm,all_ct], axis=1)

# add label to ffm data

ffm.insert(0, 'label', data['label'])
ffm.to_csv('./outputs/ffm_all.csv')
train=ffm[ffm.label!=-1]

test=ffm[ffm.label==-1]

# split train,val datasets

len_train = np.ceil(len(train) * 0.7).astype(int)
X_train = train[:len_train]
X_eval = train[len_train:]


def save_data_txt(path, raw_data):
    with open(path, 'w') as f:
        for i, data in enumerate(raw_data.values):
            for index, item in enumerate(data):
                if index <= 16:
                    f.write(str(item) + ' ')
                else:
                    f.write(item)
            if i < len(raw_data) - 1:
                f.write('\n')


print('saving file')

save_data_txt('./outputs/ffm_train.txt', X_train)
save_data_txt('./outputs/ffm_eval.txt', X_eval)
save_data_txt('./outputs/ffm_test.txt', test)