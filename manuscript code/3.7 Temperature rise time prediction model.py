#烟叶烘烤升温时间的预测
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

features = ['B', 'G', 'R', 'H', 'S', 'a*', 'b*', 'dry_up', 'wet_up', 'baketime',
       'driage', 'waterloss_rate', 'yel_per', 'contrast', 'homogeneity',
       'energy']
target = ['holdingtime']
use6_stages = ['label']
all_features = features + use6_stages + target
dataset = pd.read_csv(r"F:\pythoncode\Tobacco_thesis\Data_set\All_Data\tobacco.csv")

# 1阶段的升温时间模型训练
data_1 = dataset[dataset['label'] == 1]
# 将特征数据和标签分开
Xtrain_1, Xtest_1, Ytrain_1, Ytest_1 = train_test_split(data_1[features], data_1[target], test_size=0.2, random_state=233)
Ytest_1 = Ytest_1.values.ravel()
Ytrain_1 = Ytrain_1.values.ravel()
xgb_1 = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_jobs=4,random_state=567, seed=567,nthread=None,
                   max_depth=4,
                   learning_rate=0.02,
                   n_estimators=200,
                   gamma=0,
                   min_child_weight=2,
                   subsample=0.6,
                   reg_alpha=0.5,
                   reg_lambda=1,
                   colsample_bytree=0.9,
                   colsample_bylevel=0.6,
                  )
xgb_1.fit(Xtrain_1,Ytrain_1)
pred_1 = xgb_1.predict(Xtest_1)
rmse_1 = mean_squared_error(Ytest_1, pred_1, squared=False)
mse_1 = mean_squared_error(Ytest_1, pred_1)
mae_1 = mean_absolute_error(Ytest_1, pred_1)
r2_1 = r2_score(Ytest_1, pred_1)
print(f"stage:1,MAE:{mae_1},R2:{r2_1},MSE:{mse_1},RMSE:{rmse_1}")
joblib.dump(xgb_1, r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_1.pkl')

# 2阶段的升温时间模型训练
data_2 = dataset[dataset['label'] == 2]
# 将特征数据和标签分开
Xtrain_2, Xtest_2, Ytrain_2, Ytest_2 = train_test_split(data_2[features], data_2[target], test_size=0.2, random_state=233)
Ytest_2 = Ytest_2.values.ravel()
Ytrain_2 = Ytrain_2.values.ravel()
xgb_2 = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_jobs=4,random_state=567, seed=567,nthread=None,
                   max_depth=5,
                   learning_rate=0.05,
                   n_estimators=180,
                   gamma=0,
                   min_child_weight=0,
                   subsample=0.5,
                   reg_alpha=0.2,
                   reg_lambda=0.3,
                   colsample_bytree=1,
                   colsample_bylevel=1,
                  )
xgb_2.fit(Xtrain_2,Ytrain_2)
pred_2 = xgb_2.predict(Xtest_2)
rmse_2 = mean_squared_error(Ytest_2, pred_2, squared=False)
mse_2 = mean_squared_error(Ytest_2, pred_2)
mae_2 = mean_absolute_error(Ytest_2, pred_2)
r2_2 = r2_score(Ytest_2, pred_2)
print(f"stage:2,MAE:{mae_2},R2:{r2_2},MSE:{mse_2},RMSE:{rmse_2}")
joblib.dump(xgb_2, r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_2.pkl')

# 3阶段的升温时间模型训练
data_3 = dataset[dataset['label'] == 3]
# 将特征数据和标签分开
Xtrain_3, Xtest_3, Ytrain_3, Ytest_3 = train_test_split(data_3[features], data_3[target], test_size=0.2, random_state=233)
Ytest_3 = Ytest_3.values.ravel()
Ytrain_3 = Ytrain_3.values.ravel()
xgb_3 = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_jobs=4,random_state=567, seed=567,nthread=None,
                   max_depth=6,
                   learning_rate=0.02,
                   n_estimators=220,
                   gamma=0,
                   min_child_weight=0,
                   subsample=0.9,
                   reg_alpha=0.4,
                   reg_lambda=0.3,
                   colsample_bytree=1,
                   colsample_bylevel=0.9,
                  )
xgb_3.fit(Xtrain_3,Ytrain_3)
pred_3 = xgb_3.predict(Xtest_3)
rmse_3 = mean_squared_error(Ytest_3, pred_3, squared=False)
mse_3 = mean_squared_error(Ytest_3, pred_3)
mae_3 = mean_absolute_error(Ytest_3, pred_3)
r2_3 = r2_score(Ytest_3, pred_3)
print(f"stage:3,MAE:{mae_3},R2:{r2_3},MSE:{mse_3},RMSE:{rmse_3}")
joblib.dump(xgb_3, r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_3.pkl')

# 4阶段的升温时间模型训练
data_4 = dataset[dataset['label'] == 4]
# 将特征数据和标签分开
Xtrain_4, Xtest_4, Ytrain_4, Ytest_4 = train_test_split(data_4[features], data_4[target], test_size=0.2, random_state=233)
Ytest_4 = Ytest_4.values.ravel()
Ytrain_4 = Ytrain_4.values.ravel()
xgb_4 = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_jobs=4,random_state=567, seed=567,nthread=None,
                   max_depth=6,
                   learning_rate=0.1,
                   n_estimators=250,
                   gamma=0,
                   min_child_weight=2,
                   subsample=1,
                   reg_alpha=0.4,
                   reg_lambda=0.2,
                   colsample_bytree=1,
                   colsample_bylevel=1,
                  )
xgb_4.fit(Xtrain_4,Ytrain_4)
pred_4 = xgb_4.predict(Xtest_4)
rmse_4 = mean_squared_error(Ytest_4, pred_4, squared=False)
mse_4 = mean_squared_error(Ytest_4, pred_4)
mae_4 = mean_absolute_error(Ytest_4, pred_4)
r2_4 = r2_score(Ytest_4, pred_4)
print(f"stage:4,MAE:{mae_4},R2:{r2_4},MSE:{mse_4},RMSE:{rmse_4}")
joblib.dump(xgb_4, r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_4.pkl')

# 5阶段的升温时间模型训练
data_5 = dataset[dataset['label'] == 5]
# 将特征数据和标签分开
Xtrain_5, Xtest_5, Ytrain_5, Ytest_5 = train_test_split(data_5[features], data_5[target], test_size=0.2, random_state=233)
Ytest_5 = Ytest_5.values.ravel()
Ytrain_5 = Ytrain_5.values.ravel()
xgb_5 = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_jobs=4,random_state=567, seed=567,nthread=None,
                   max_depth=3,
                   learning_rate=0.03,
                   n_estimators=200,
                   gamma=0,
                   min_child_weight=2,
                   subsample=0.7,
                   reg_alpha=1e-5,
                   reg_lambda=0.001,
                   colsample_bytree=0.8,
                   colsample_bylevel=1,
                  )
xgb_5.fit(Xtrain_5,Ytrain_5)
pred_5 = xgb_5.predict(Xtest_5)
rmse_5 = mean_squared_error(Ytest_5, pred_5, squared=False)
mse_5 = mean_squared_error(Ytest_5, pred_5)
mae_5 = mean_absolute_error(Ytest_5, pred_5)
r2_5 = r2_score(Ytest_5, pred_5)
print(f"stage:5,MAE:{mae_5},R2:{r2_5},MSE:{mse_5},RMSE:{rmse_5}")
joblib.dump(xgb_5, r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_5.pkl')

# 6阶段的升温时间模型训练
data_6 = dataset[dataset['label'] == 6]
# 将特征数据和标签分开
Xtrain_6, Xtest_6, Ytrain_6, Ytest_6 = train_test_split(data_6[features], data_6[target], test_size=0.2, random_state=233)
Ytest_6 = Ytest_6.values.ravel()
Ytrain_6 = Ytrain_6.values.ravel()
xgb_6 = XGBRegressor(objective='reg:squarederror', booster='gbtree', n_jobs=4,random_state=567, seed=567,nthread=None,
                   max_depth=6,
                   learning_rate=0.14,
                   n_estimators=190,
                   gamma=0,
                   min_child_weight=0,
                   subsample=1,
                   reg_alpha=0.1,
                   reg_lambda=1,
                   colsample_bytree=1,
                   colsample_bylevel=1,
                  )
xgb_6.fit(Xtrain_6,Ytrain_6)
pred_6 = xgb_6.predict(Xtest_6)
rmse_6 = mean_squared_error(Ytest_6, pred_6, squared=False)
mse_6 = mean_squared_error(Ytest_6, pred_6)
mae_6 = mean_absolute_error(Ytest_6, pred_6)
r2_6 = r2_score(Ytest_6, pred_6)
print(f"stage:6,MAE:{mae_6},R2:{r2_6},MSE:{mse_6},RMSE:{rmse_6}")
joblib.dump(xgb_6, r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_6.pkl')