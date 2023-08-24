# 常用工具库
import pandas as pd
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
begin = time.time()
dataset = pd.read_csv(r"F:\pythoncode\Tobacco_thesis\Data_set\All_Data.csv")
features = ['B', 'G', 'R', 'H', 'S', 'a*', 'b*', 'dry_up', 'wet_up', 'baketime',
       'driage', 'waterloss_rate', 'yel_per', 'contrast', 'homogeneity',
       'energy']
# 将特征数据和标签分开
X,y = dataset.drop(columns=["label","holdingtime"]),dataset.label
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.20, random_state=666)
print(Ytest.shape)
print(Xtest.shape)
Xtrain

# 第一层训练
xgb_1 = XGBClassifier(objective='multi:softmax', booster='gbtree',
                    random_state=567, seed=567,eval_metric='mlogloss',
                      n_estimators=500
                     ,learning_rate = 0.19
                     ,max_depth = 3
                     ,gamma = 0
                     ,subsample = 0.7
                     ,colsample_bytree = 0.6
                     ,colsample_bylevel = 1
                     ,reg_lambda = 0.0001
                     ,reg_alpha = 1
                     ,min_child_weight = 1
                     )
xgb_1 = xgb_1.fit(Xtrain,Ytrain)
print(xgb_1.score(Xtest,Ytest))
first_pred = xgb_1.predict(Xtest)
first_pred
# 模型存储
joblib.dump(xgb_1, r'F:\pythoncode\Tobacco_thesis\Model\status_1.pkl')
#建立表格，存储子模型的预测结果
result = pd.DataFrame()

# 开始第二次训练，根据不同的数据构建不同的模型
# 颜色特征
color_features = ["B", "G", "R","H", "S", "a*", "b*","yel_per"]
# 纹理特征
texture_features = ['contrast', 'homogeneity', 'energy']
# 含水量特征
water_features = ['driage','waterloss_rate']
# 烤房环境特征
environment_features = ['dry_up', 'wet_up', 'baketime']
# 标签
labels = ["label"]
def modeling(data, features):
    Xtest, Ytest = data[features].values, data[labels].values
    return Xtest, Ytest

# 训练不同的特征模型

# 颜色特征模型
xgb_2_color = XGBClassifier(objective='multi:softmax', booster='gbtree',
                    random_state=567, seed=567,eval_metric='mlogloss',
                      n_estimators=600
                     ,learning_rate = 0.11
                     ,max_depth = 6
                     ,gamma = 0
                     ,subsample = 0.6
                     ,colsample_bytree = 0.9
                     ,colsample_bylevel = 0.9
                     ,reg_lambda = 0.0
                     ,reg_alpha = 1
                     ,min_child_weight = 1
                     )
# 构建数据集进行训练和预测
color_Xtrain,color_Ytrain = Xtrain[color_features], Ytrain  #取出训练集中的颜色特征数据
xgb_2_color = xgb_2_color.fit(color_Xtrain, color_Ytrain) #将颜色特征数据放入学习器中进行训练
color_Ypred = xgb_2_color.predict(Xtest[color_features]) #用训练好的学习器预测测试集的数据
print(xgb_2_color.score(Xtest[color_features],Ytest))
result['true_label'] = Ytest
result['color_Ypred'] = color_Ypred
# 模型存储
joblib.dump(xgb_2_color, r'F:\pythoncode\Tobacco_thesis\Model\status_2_color.pkl')

# 纹理特征模型
xgb_2_texture = XGBClassifier(objective='multi:softmax', booster='gbtree',
                    random_state=567, seed=567,eval_metric='mlogloss',
                      n_estimators=600
                     ,learning_rate = 0.2
                     ,max_depth = 10
                     ,gamma = 0
                     ,subsample = 0.9
                     ,colsample_bytree = 1
                     ,colsample_bylevel = 1
                     ,reg_lambda = 0.5
                     ,reg_alpha = 0.1
                     ,min_child_weight = 1
                     )
# 构建数据集进行训练和预测
texture_Xtrain,texture_Ytrain = Xtrain[texture_features], Ytrain  #取出训练集中的纹理特征数据
xgb_2_texture = xgb_2_texture.fit(texture_Xtrain, texture_Ytrain) #将纹理特征数据放入学习器中进行训练
texture_Ypred = xgb_2_texture.predict(Xtest[texture_features]) #用训练好的学习器预测测试集的数据
result['texture_Ypred'] = texture_Ypred
print(xgb_2_texture.score(Xtest[texture_features],Ytest))
# 模型存储
joblib.dump(xgb_2_texture, r'F:\pythoncode\Tobacco_thesis\Model\status_2_texture.pkl')

# 含水量特征模型
xgb_2_water = XGBClassifier(objective='multi:softmax', booster='gbtree',
                    random_state=567, seed=567,eval_metric='mlogloss',
                      n_estimators=500
                     ,learning_rate = 0.1
                     ,max_depth = 9
                     ,gamma = 0.9
                     ,subsample = 0.8
                     ,colsample_bytree = 1
                     ,colsample_bylevel = 0.5
                     ,reg_lambda = 0.1
                     ,reg_alpha = 0.9
                     ,min_child_weight = 0
                     )
# 构建数据集进行训练和预测
water_Xtrain,water_Ytrain = Xtrain[water_features], Ytrain  #取出训练集中的含水量特征数据
xgb_2_water = xgb_2_water.fit(water_Xtrain, water_Ytrain) #将含水量特征数据放入学习器中进行训练
water_Ypred = xgb_2_water.predict(Xtest[water_features]) #用训练好的学习器预测测试集的数据
result['water_Ypred'] = water_Ypred
print(xgb_2_water.score(Xtest[water_features],Ytest))
# 模型存储
joblib.dump(xgb_2_water, r'F:\pythoncode\Tobacco_thesis\Model\status_2_water.pkl')

# 烤房环境特征模型
xgb_2_environment = XGBClassifier(objective='multi:softmax', booster='gbtree',
                    random_state=567, seed=567,eval_metric='mlogloss',
                      n_estimators=400
                     ,learning_rate = 0.15
                     ,max_depth = 6
                     ,gamma = 0
                     ,subsample = 1
                     ,colsample_bytree = 1
                     ,colsample_bylevel = 1
                     ,reg_lambda = 0
                     ,reg_alpha = 1
                     )
# 构建数据集进行训练和预测
environment_Xtrain,environment_Ytrain = Xtrain[environment_features], Ytrain  #取出训练集中的烤房环境特征数据
environment_Xtest,environment_Ytest = Xtest[environment_features], Ytest  #取出训练集中的烤房环境特征数据
xgb_2_environment = xgb_2_environment.fit(environment_Xtrain, environment_Ytrain) #将烤房环境特征数据放入学习器中进行训练
environment_Ypred = xgb_2_environment.predict(Xtest[environment_features]) #用训练好的学习器预测测试集的数据
result['environment_Ypred'] = environment_Ypred
print(xgb_2_environment.score(Xtest[environment_features],Ytest))
# 模型存储
joblib.dump(xgb_2_environment, r'F:\pythoncode\Tobacco_thesis\Model\status_2_environment.pkl')


total = np.zeros((1634, 6)) #建立新表存放预测的数据
total
# 对每行赋予不同的权重进行预测，预测结果收集起来
for i in range(Xtest.shape[0]):
    series_np = np.array(Xtest.iloc[i]).reshape(1, -1)  # 将每一行取出来，转为可以导入模型的数据格式
    pred_proba_solo = xgb_1.predict_proba(series_np)  # 提取该样本第一层预测的概率矩阵
    series = Xtest.iloc[i]
    # 颜色模型预测该样本
    color_series_solo = series[color_features]  # 将每一行的颜色数据取出来
    color_series_solo_np = np.array(color_series_solo).reshape(1, -1)  # 转为可以导入模型的数据格式
    color_pred_proba_solo = xgb_2_color.predict_proba(color_series_solo_np)  # 导入颜色模型，得到预测概率矩阵
    # 纹理模型预测该样本
    texture_series_solo = series[texture_features]
    texture_series_solo_np = np.array(texture_series_solo).reshape(1, -1)
    texture_pred_proba_solo = xgb_2_texture.predict_proba(texture_series_solo_np)
    # 含水量模型预测该样本
    water_series_solo = series[water_features]
    water_series_solo_np = np.array(water_series_solo).reshape(1, -1)
    water_pred_proba_solo = xgb_2_water.predict_proba(water_series_solo_np)
    # 烤房环境模型预测该样本
    environment_series_solo = series[environment_features]
    environment_series_solo_np = np.array(environment_series_solo).reshape(1, -1)
    environment_pred_proba_solo = xgb_2_environment.predict_proba(environment_series_solo_np)

    # 将第一层的预测结果转化为权重比值输入第二层
    pred_solo = xgb_1.predict(series_np)
    if pred_solo == 1:
        W_color = 1.5
        W_texture = 1
        W_water = 1
        W_environment = 3
    elif pred_solo == 2:
        W_color = 1.5
        W_texture = 1.2
        W_water = 0.5
        W_environment = 2
    elif pred_solo == 3:
        W_color = 1
        W_texture = 1.5
        W_water = 2
        W_environment = 2
    elif pred_solo == 4:
        W_color = 1
        W_texture = 1
        W_water = 1
        W_environment = 3
    elif pred_solo == 5:
        W_color = 1
        W_texture = 1
        W_water = 0.5
        W_environment = 3
    else:
        W_color = 1
        W_texture = 1
        W_water = 0.5
        W_environment = 3
    for col in range(len(pred_proba_solo[0])):
        total[i, col] = color_pred_proba_solo[0, col] * W_color + texture_pred_proba_solo[0, col] * W_texture + \
                        water_pred_proba_solo[0, col] * W_water + environment_pred_proba_solo[0, col] * W_environment