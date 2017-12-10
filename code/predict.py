import pandas as pd
from sklearn import  preprocessing
import xgboost as xgb

feature_all = pd.read_csv('..\\feature\\feature_all_one_hot_multi_record.csv',index_col=0)
train_neg = feature_all[(feature_all.is_train == 1) & (feature_all.is_risk == 0)]
train_pos = feature_all[(feature_all.is_train == 1) & (feature_all.is_risk == 1)]
train = pd.concat([train_neg,train_pos])
label = train['is_risk']
test = feature_all[feature_all.is_train == 0]
columns = train.columns
feature = [x for x in columns if x not in ['rowkey','is_risk','is_train','label','login_time','log_from_1','buy_time_box',
                                           'login_time_box','log_result_1','log_type_1','cnt_total']]
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['is_risk'].values))
train['label'] = lbl.transform(list(train['is_risk'].values))
num_class = 2
params = {
    'objective': 'multi:softmax',
    'eta': 0.03,
    'max_depth': 9,
    'eval_metric': 'merror',
    'scale_pos_weight':10,
    'learning_rate': 0.08,
    'seed': 0,
    'silent': 1,
    'num_class':2
}
xgbtrain = xgb.DMatrix(train[feature], train['label'])
xgbtest = xgb.DMatrix(test[feature])
watchlist = [(xgbtrain, 'train'), (xgbtrain, 'test')]
num_rounds = 2000
model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
test_label = model.predict(xgbtest)
test['label'] = (list(test_label))
test[['rowkey','label']].to_csv('re_feature_all_one_hot_multi_record_1210.csv')
