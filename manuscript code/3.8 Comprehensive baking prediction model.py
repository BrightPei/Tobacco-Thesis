# 常用工具库
import pandas as pd
import numpy as np
import warnings
import joblib
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

features = ['B', 'G', 'R', 'H', 'S', 'a*', 'b*', 'dry_up', 'wet_up', 'baketime',
       'driage', 'waterloss_rate', 'yel_per', 'contrast', 'homogeneity',
       'energy']
# 颜色特征
color_features = ["B", "G", "R","H", "S", "a*", "b*","yel_per"]
# 纹理特征
texture_features = ['contrast', 'homogeneity', 'energy']
# 含水量特征
water_features = ['driage','waterloss_rate']
# 烤房环境特征
environment_features = ['dry_up', 'wet_up', 'baketime']
target = ['holdingtime']
use6_stages = ['label']
all_features = features + use6_stages + target
dataset = pd.read_csv(r"F:\pythoncode\Tobacco_thesis\Data_set\All_Data.csv")
#构建测试数据集
X,y = dataset.drop(columns=["label","holdingtime"]),dataset[["label","holdingtime"]]
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.20, random_state=666)

total = np.zeros((1634, 6)) #建立新表存放预测的数据

# 使用阶段模型对烘烤阶段进行预测
for i in range(Xtest.shape[0]):
    series_np=np.array(Xtest.iloc[i]).reshape(1,-1) #将每一行取出来，转为可以导入模型的数据格式

    # 第一层的阶段模型预测
    status_1 = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\status_1.pkl')# 模型加载
    pred_solo = status_1.predict(series_np)# 模型预测

    # 将第一层的预测结果转化为权重比值输入第二层
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

    # 第二层的阶段模型预测：
    series = Xtest.iloc[i]#取出每行的数据
    #颜色模型预测该样本
    status_2_color = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\status_2_color.pkl')# 颜色模型加载
    color_series_solo = series[color_features] #将每一行的颜色数据取出来
    color_series_solo_np = np.array(color_series_solo).reshape(1,-1) #转为可以导入模型的数据格式
    color_pred_proba_solo = status_2_color.predict_proba(color_series_solo_np) #导入颜色模型，得到预测概率矩阵
    color_pred_solo = status_2_color.predict(color_series_solo_np)# 模型预测
    #纹理模型预测该样本
    status_2_texture = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\status_2_texture.pkl')# 纹理模型加载
    texture_series_solo = series[texture_features] #将每一行的纹理数据取出来
    texture_series_solo_np = np.array(texture_series_solo).reshape(1,-1) #转为可以导入模型的数据格式
    texture_pred_proba_solo = status_2_texture.predict_proba(texture_series_solo_np) #导入纹理模型，得到预测概率矩阵
    texture_pred_solo = status_2_texture.predict(texture_series_solo_np)# 模型预测
    #含水量模型预测该样本
    status_2_water = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\status_2_water.pkl')# 含水量模型加载
    water_series_solo = series[water_features] #将每一行的含水量数据取出来
    water_series_solo_np = np.array(water_series_solo).reshape(1,-1) #转为可以导入模型的数据格式
    water_pred_proba_solo = status_2_water.predict_proba(water_series_solo_np) #导入含水量模型，得到预测概率矩阵
    water_pred_solo = status_2_water.predict(water_series_solo_np)# 模型预测
    #烤房环境模型预测该样本
    status_2_environment = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\status_2_environment.pkl')# 烤房环境模型加载
    environment_series_solo = series[environment_features] #将每一行的烤房环境数据取出来
    environment_series_solo_np = np.array(environment_series_solo).reshape(1,-1) #转为可以导入模型的数据格式
    environment_pred_proba_solo = status_2_environment.predict_proba(environment_series_solo_np) #导入烤房环境模型，得到预测概率矩阵
    environment_pred_solo = status_2_environment.predict(environment_series_solo_np)# 模型预测
    for col in range(len(environment_pred_proba_solo[0])):
        total[i, col] = color_pred_proba_solo[0, col] *W_color + texture_pred_proba_solo[0, col] * W_texture + water_pred_proba_solo[0, col] * W_water+ environment_pred_proba_solo[0, col] * W_environment

PointTime = []

# 根据第一层的预测结果进行升温时间的预测
for i in range(total.shape[0]):
    final_status = total.iloc[i]['final_status']#提取该行数据的阶段预测值
    #根据阶段预测值调用对应的升温时间模型
    if final_status == 1:
        PointTime_model = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_1.pkl')# 1阶段模型加载
    elif final_status == 2:
        PointTime_model = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_2.pkl')# 2阶段模型加载
    elif final_status == 3:
        PointTime_model = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_3.pkl')# 3阶段模型加载
    elif final_status == 4:
        PointTime_model = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_4.pkl')# 4阶段模型加载
    elif final_status == 5:
        PointTime_model = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_5.pkl')# 5阶段模型加载
    else:
        PointTime_model = joblib.load(r'F:\pythoncode\Tobacco_thesis\Model\RiseTime_6.pkl')# 6阶段模型加载
    # 预测升温时间
    series_np=np.array(Xtest.iloc[i]).reshape(1,-1) #将每一行取出来，转为可以导入模型的数据格式
    pred_PointTime = PointTime_model.predict(series_np)
    PointTime.append(pred_PointTime)

total['pred_PointTime'] = PointTime