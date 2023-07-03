import numpy as np 
import pandas as pd 
import os 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import lightgbm as lgb

train_data_root = './raw_data/sharechat_recsys2023_data/train'
train_data_files = os.listdir(train_data_root)
test_data_file= './raw_data/sharechat_recsys2023_data/test/000000000000.csv'

train_data = [pd.read_csv(os.path.join(train_data_root,train_data_files[i]),sep='\t') 
              for i in range(len(train_data_files))]
train_data = pd.concat(train_data,axis=0)
test_data = pd.read_csv(test_data_file,sep='\t')

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

print("Data Loaded")
# with open('./raw_train.pkl','rb') as f:
#     train_data = pickle.load(f).reset_index(drop=True)
# with open('./raw_test.pkl','rb') as f:
#     test_data = pickle.load(f).reset_index(drop=True)

test_data['is_clicked'] = -1
test_data['is_installed'] = -1

train_data['x_1'] = train_data['f_1'] % 7
test_data['x_1'] = test_data['f_1'] % 7

train_data_full = train_data.copy()
valid_data = train_data[train_data['f_1']==66]
train_data = train_data[train_data['f_1']<66]

y_valid = pd.DataFrame(columns=['f_30','f_31'],index=valid_data.index)
y_test = pd.DataFrame(columns=['f_30','f_31'],index=test_data.index)

train_na = train_data[train_data['f_30'].isna()]
valid_na = valid_data[valid_data['f_30'].isna()]
test_na = test_data[test_data['f_30'].isna()]

train_not_na = train_data[~train_data['f_30'].isna()]
X_train = train_not_na.drop(['is_clicked','is_installed','f_30','f_31'],axis=1)
y_train = train_not_na[['f_30','f_31']]

X_train_na = train_na.drop(['is_clicked','is_installed','f_30','f_31'],axis=1)
X_valid_na = valid_na.drop(['is_clicked','is_installed','f_30','f_31'],axis=1)
X_test_na = test_na.drop(['is_clicked','is_installed','f_30','f_31'],axis=1)

gbm1 = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          random_state=42,
                          learning_rate=0.05,
                          max_depth=3,
                          num_leaves=7,verbose=3)
gbm1.fit(X_train, y_train.f_30)
X_train_na['f_30'] = gbm1.predict(X_train_na)
X_valid_na['f_30'] = gbm1.predict(X_valid_na)
X_test_na['f_30'] = gbm1.predict(X_test_na)

gbm2 = lgb.LGBMClassifier(objective='binary',
                          metric='auc',
                          random_state=42,
                          learning_rate=0.05,
                          max_depth=3,
                          num_leaves=7,verbose=3)
gbm2.fit(X_train, y_train.f_31)

X_train_na['f_31'] = gbm2.predict(X_train_na.drop(['f_30'],axis=1))
X_valid_na['f_31'] = gbm2.predict(X_valid_na.drop(['f_30'],axis=1))
X_test_na['f_31'] = gbm2.predict(X_test_na.drop(['f_30'],axis=1))

cnt_na = np.sum(train_data.isna())
cols = cnt_na[cnt_na!=0].index
fillna_dict = dict()
for c in cols:
    if c in ['f_30','f_31']:
        continue
    else:
        fillna_dict[c] = np.mean(train_data[c])

fill_train = X_train_na[['f_30','f_31']]
fill_valid = X_valid_na[['f_30','f_31']]
fill_test = X_test_na[['f_30','f_31']]

test_data = test_data.fillna(fill_test)
print(valid_data.columns)
print('f_30' in valid_data)
print('f_31' in valid_data)
valid_data = valid_data.fillna(fill_valid)
train_data = train_data.fillna(fill_train)

print(np.sum(train_data.isna())[['f_30','f_31']])

train_data = train_data.fillna(fillna_dict)
valid_data = valid_data.fillna(fillna_dict)
test_data = test_data.fillna(fillna_dict)

mms = MinMaxScaler(feature_range=(0, 1))# 
dense_feature = []
for i in range(42,80):
    dense_feature.append('f_{}'.format(i))
train_data[dense_feature] = mms.fit_transform(train_data[dense_feature])
valid_data[dense_feature] = mms.transform(valid_data[dense_feature])
test_data[dense_feature] = mms.transform(test_data[dense_feature])

train_data_full = pd.concat([train_data,valid_data],axis=0).reset_index(drop=True)

# save_data 
dataset_name = 'final'
if not os.path.exists("{}".format(dataset_name)):
    os.mkdir("{}".format(dataset_name))
train_data.to_csv('./{}/train_data_{}.csv'.format(dataset_name,dataset_name),index=None)
test_data.to_csv('./{}/test_data_{}.csv'.format(dataset_name,dataset_name),index=None)
valid_data.to_csv('./{}/valid_data_{}.csv'.format(dataset_name,dataset_name),index=None)
train_data_full.to_csv('./{}/train_data_{}_full.csv'.format(dataset_name,dataset_name),index=None)