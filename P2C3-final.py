# -*- coding: utf-8 -*-

# 性能-成分模型 （对数变换，测试集范围限制）
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import prettytable as pt

# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")

# 读取数据
df = pd.read_excel('Data.xlsx', sheet_name='数据处理1')
data = df.values

# 选择特征与标签(x是铜合金的元素含量，y是材料的抗拉强度及导电率)
x = data[0:408, 1:16]
y = data[0:408, 16:18]
# 获取特征和标签的维度
x_shape = x.shape[1]
y_shape = y.shape[1]

# 归一化函数
def Normalization(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    min = min_max_scaler.data_min_
    max = min_max_scaler.data_max_
    return data, min, max

# 反归一化函数
def Anti_Normalization(data, min, max):
    data = data * (max - min) + min
    return data

# 对数化函数
def Logarithmic(data):
    data[:, 5] = np.log(data[:, 5] + 1)
    return data

# 反对数化函数
def Anti_Logarithmic(data):
    data[:, 5] = np.exp(data[:, 5]) - 1
    return data

# minmax归一化处理
x, minx, maxx = Normalization(x)
y, miny, maxy = Normalization(y)
x = Logarithmic(x)

# 划分90%训练集，10%测试集
x_train_, x_test_, y_train_, y_test_ = train_test_split(x, y, test_size=0.1, random_state=0)

# 测试集反对数化，反归一化
x_test_ = Anti_Logarithmic(x_test_)
x_test_ = Anti_Normalization(x_test_, minx, maxx)

# 限制测试集范围
test_shape = 0
for k in range(0, x_test_.shape[0]):
    flag=False
    for g in range(0, x_test_.shape[1]):
        if x_test_[k, g] > 10:
            flag=True
    if flag==False:
        if test_shape == 0:
            x_test = x_test_[k, :]
            y_test = y_test_[k, :]
        else:
            x_test = np.vstack((x_test, x_test_[k, :]))
            y_test = np.vstack((y_test, y_test_[k, :]))
        test_shape = test_shape + 1
print('测试集大小:',x_test_.shape[0],'限制范围后测试集大小:',test_shape)

fe_Evaluation = np.zeros((3, 4))
p_Evaluation = np.zeros((3, 4))
ni_Evaluation = np.zeros((3, 4))
si_Evaluation = np.zeros((3, 4))
mg_Evaluation = np.zeros((3, 4))
zn_Evaluation = np.zeros((3, 4))
sn_Evaluation = np.zeros((3, 4))
cr_Evaluation = np.zeros((3, 4))
zr_Evaluation = np.zeros((3, 4))
re_Evaluation = np.zeros((3, 4))
al_Evaluation = np.zeros((3, 4))
pb_Evaluation = np.zeros((3, 4))
mn_Evaluation = np.zeros((3, 4))
sb_Evaluation = np.zeros((3, 4))
s_Evaluation = np.zeros((3, 4))

ALL_Evaluation = np.zeros((5, 4))

# 全集的反对数化和反归一化
x = Anti_Logarithmic(x)
x = Anti_Normalization(x, minx, maxx)

for i in range(0, 5):
    x_train, x_validation, y_train, y_validation = train_test_split(x_train_, y_train_, test_size=0.1, random_state=0)

    # 建立Fe模型
    # 梯度提升树
    fe_model = GradientBoostingRegressor(n_estimators=50, max_depth=7)
    # 训练模型并代入验证集
    fe_model_train = fe_model.fit(y_train, x_train[:, 0])
    fe_pre_train = fe_model.predict(y_train)
    fe_pre_validation = fe_model.predict(y_validation)
    fe_pre_test = fe_model.predict(y_test)
    fe_pre_all = fe_model.predict(y)

    # 建立P模型
    # 随机森林
    p_model = RandomForestRegressor(max_depth=9)
    # 训练模型并代入验证集
    p_model_train = p_model.fit(y_train, x_train[:, 1])
    p_pre_train = p_model.predict(y_train)
    p_pre_validation = p_model.predict(y_validation)
    p_pre_test = p_model.predict(y_test)
    p_pre_all = p_model.predict(y)

    # 建立Ni模型
    # 支持向量机
    ni_model = SVR(kernel='rbf', gamma=7.5)
    # 训练模型并代入验证集
    ni_model_train = ni_model.fit(y_train, x_train[:, 2])
    ni_pre_train = ni_model.predict(y_train)
    ni_pre_validation = ni_model.predict(y_validation)
    ni_pre_test = ni_model.predict(y_test)
    ni_pre_all = ni_model.predict(y)

    # 建立Si模型
    # XGBoost
    si_model = XGBRegressor(n_estimators=50, max_depth=9)
    # 训练模型并代入验证集
    si_model_train = si_model.fit(y_train, x_train[:, 3])
    si_pre_train = si_model.predict(y_train)
    si_pre_validation = si_model.predict(y_validation)
    si_pre_test = si_model.predict(y_test)
    si_pre_all = si_model.predict(y)

    # 建立Mg模型
    # GBDT
    mg_model = GradientBoostingRegressor(n_estimators=30, max_depth=3)
    # 训练模型并代入验证集
    mg_model_train = mg_model.fit(y_train, x_train[:, 4])
    mg_pre_train = mg_model.predict(y_train)
    mg_pre_validation = mg_model.predict(y_validation)
    mg_pre_test = mg_model.predict(y_test)
    mg_pre_all = mg_model.predict(y)

    # 建立Zn模型
    # BP神经网络
    zn_model = MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=17, max_iter=600)
    # 训练模型并代入验证集
    zn_model_train = zn_model.fit(y_train, x_train[:, 5])
    zn_pre_train = zn_model.predict(y_train)
    zn_pre_validation = zn_model.predict(y_validation)
    zn_pre_test = zn_model.predict(y_test)
    zn_pre_all = zn_model.predict(y)

    # 建立Sn模型
    # 随机森林
    sn_model = RandomForestRegressor(max_depth=6)
    # 训练模型并代入验证集
    sn_model_train = sn_model.fit(y_train, x_train[:, 6])
    sn_pre_train = sn_model.predict(y_train)
    sn_pre_validation = sn_model.predict(y_validation)
    sn_pre_test = sn_model.predict(y_test)
    sn_pre_all = sn_model.predict(y)

    # 建立Cr模型
    # BP神经网络
    cr_model = MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=19, max_iter=1600)
    # 训练模型并代入验证集
    cr_model_train = cr_model.fit(y_train, x_train[:, 7])
    cr_pre_train = cr_model.predict(y_train)
    cr_pre_validation = cr_model.predict(y_validation)
    cr_pre_test = cr_model.predict(y_test)
    cr_pre_all = cr_model.predict(y)

    # 建立Zr模型
    # 随机森林
    zr_model = RandomForestRegressor(max_depth=1)
    # 训练模型并代入验证集
    zr_model_train = zr_model.fit(y_train, x_train[:, 8])
    zr_pre_train = zr_model.predict(y_train)
    zr_pre_validation = zr_model.predict(y_validation)
    zr_pre_test = zr_model.predict(y_test)
    zr_pre_all = zr_model.predict(y)

    # 建立RE模型
    # 随机森林
    re_model = RandomForestRegressor(max_depth=6)
    # 训练模型并代入验证集
    re_model_train = re_model.fit(y_train, x_train[:, 9])
    re_pre_train = re_model.predict(y_train)
    re_pre_validation = re_model.predict(y_validation)
    re_pre_test = re_model.predict(y_test)
    re_pre_all = re_model.predict(y)

    # 建立Al模型
    # 梯度提升树
    al_model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    # 训练模型并代入验证集
    al_model_train = al_model.fit(y_train, x_train[:, 10])
    al_pre_train = al_model.predict(y_train)
    al_pre_validation = al_model.predict(y_validation)
    al_pre_test = al_model.predict(y_test)
    al_pre_all = al_model.predict(y)

    # 建立Pb模型
    # XGBoost
    pb_model = XGBRegressor(n_estimators=50, max_depth=5)
    # 训练模型并代入验证集
    pb_model_train = pb_model.fit(y_train, x_train[:, 11])
    pb_pre_train = pb_model.predict(y_train)
    pb_pre_validation = pb_model.predict(y_validation)
    pb_pre_test = pb_model.predict(y_test)
    pb_pre_all = pb_model.predict(y)

    # 建立Mn模型
    # 随机森林
    mn_model = RandomForestRegressor(max_depth=1)
    # 训练模型并代入验证集
    mn_model_train = mn_model.fit(y_train, x_train[:, 12])
    mn_pre_train = mn_model.predict(y_train)
    mn_pre_validation = mn_model.predict(y_validation)
    mn_pre_test = mn_model.predict(y_test)
    mn_pre_all = mn_model.predict(y)

    # 建立Sb模型
    # 梯度提升树
    sb_model = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    # 训练模型并代入验证集
    sb_model_train = sb_model.fit(y_train, x_train[:, 13])
    sb_pre_train = sb_model.predict(y_train)
    sb_pre_validation = sb_model.predict(y_validation)
    sb_pre_test = sb_model.predict(y_test)
    sb_pre_all = sb_model.predict(y)

    # 建立S模型
    # 支持向量机
    s_model = SVR(kernel='rbf', gamma=1.5)
    # 训练模型并代入验证集
    s_model_train = s_model.fit(y_train, x_train[:, 14])
    s_pre_train = s_model.predict(y_train)
    s_pre_validation = s_model.predict(y_validation)
    s_pre_test = s_model.predict(y_test)
    s_pre_all = s_model.predict(y)

    x_pre_train = np.vstack((fe_pre_train, p_pre_train, ni_pre_train, si_pre_train, mg_pre_train, zn_pre_train,
                             sn_pre_train, cr_pre_train, zr_pre_train, re_pre_train, al_pre_train, pb_pre_train,
                             mn_pre_train, sb_pre_train, s_pre_train)).T
    x_pre_validation = np.vstack((fe_pre_validation, p_pre_validation, ni_pre_validation, si_pre_validation,
                                  mg_pre_validation, zn_pre_validation, sn_pre_validation, cr_pre_validation,
                                  zr_pre_validation, re_pre_validation, al_pre_validation, pb_pre_validation,
                                  mn_pre_validation, sb_pre_validation, s_pre_validation)).T
    x_pre_test = np.vstack((fe_pre_test, p_pre_test, ni_pre_test, si_pre_test, mg_pre_test, zn_pre_test, sn_pre_test,
                            cr_pre_test, zr_pre_test, re_pre_test, al_pre_test, pb_pre_test, mn_pre_test, sb_pre_test,
                            s_pre_test)).T
    x_pre_all = np.vstack((fe_pre_all, p_pre_all, ni_pre_all, si_pre_all, mg_pre_all, zn_pre_all, sn_pre_all, cr_pre_all
                           , zr_pre_all, re_pre_all, al_pre_all, pb_pre_all, mn_pre_all, sb_pre_all, s_pre_all)).T

    # 训练集/验证集/测试集/全集 反对数变换
    x_train = Anti_Logarithmic(x_train)
    x_validation = Anti_Logarithmic(x_validation)

    x_pre_train = Anti_Logarithmic(x_pre_train)
    x_pre_validation = Anti_Logarithmic(x_pre_validation)
    x_pre_test = Anti_Logarithmic(x_pre_test)
    x_pre_all = Anti_Logarithmic(x_pre_all)

    # 训练集/验证集/测试集/全集 反归一化
    x_train = Anti_Normalization(x_train, minx, maxx)
    x_validation = Anti_Normalization(x_validation, minx, maxx)

    x_pre_train = Anti_Normalization(x_pre_train, minx, maxx)
    x_pre_validation = Anti_Normalization(x_pre_validation, minx, maxx)
    x_pre_test = Anti_Normalization(x_pre_test, minx, maxx)
    x_pre_all = Anti_Normalization(x_pre_all, minx, maxx)

    # 计算Fe的MAE
    fe_Evaluation[0, 0] = mean_absolute_error(x_train[:, 0], x_pre_train[:, 0])  # 训练集
    fe_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 0], x_pre_validation[:, 0])  # 验证集
    fe_Evaluation[0, 2] = mean_absolute_error(x_test[:, 0], x_pre_test[:, 0])  # 测试集
    fe_Evaluation[0, 3] = mean_absolute_error(x[:, 0], x_pre_all[:, 0])  # 全集
    # 计算Fe的MSE
    fe_Evaluation[1, 0] = mean_squared_error(x_train[:, 0], x_pre_train[:, 0])  # 训练集
    fe_Evaluation[1, 1] = mean_squared_error(x_validation[:, 0], x_pre_validation[:, 0])  # 验证集
    fe_Evaluation[1, 2] = mean_squared_error(x_test[:, 0], x_pre_test[:, 0])  # 测试集
    fe_Evaluation[1, 3] = mean_squared_error(x[:, 0], x_pre_all[:, 0])  # 全集
    # 计算Fe的R2
    fe_Evaluation[2, 0] = r2_score(x_train[:, 0], x_pre_train[:, 0])  # 训练集
    fe_Evaluation[2, 1] = r2_score(x_validation[:, 0], x_pre_validation[:, 0])  # 验证集
    fe_Evaluation[2, 2] = r2_score(x_test[:, 0], x_pre_test[:, 0])  # 测试集
    fe_Evaluation[2, 3] = r2_score(x[:, 0], x_pre_all[:, 0])  # 全集

    # 计算P的MAE
    p_Evaluation[0, 0] = mean_absolute_error(x_train[:, 1], x_pre_train[:, 1])  # 训练集
    p_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 1], x_pre_validation[:, 1])  # 验证集
    p_Evaluation[0, 2] = mean_absolute_error(x_test[:, 1], x_pre_test[:, 1])  # 测试集
    p_Evaluation[0, 3] = mean_absolute_error(x[:, 1], x_pre_all[:, 1])  # 全集
    # 计算P的MSE
    p_Evaluation[1, 0] = mean_squared_error(x_train[:, 1], x_pre_train[:, 1])  # 训练集
    p_Evaluation[1, 1] = mean_squared_error(x_validation[:, 1], x_pre_validation[:, 1])  # 验证集
    p_Evaluation[1, 2] = mean_squared_error(x_test[:, 1], x_pre_test[:, 1])  # 测试集
    p_Evaluation[1, 3] = mean_squared_error(x[:, 1], x_pre_all[:, 1])  # 全集
    # 计算P的R2
    p_Evaluation[2, 0] = r2_score(x_train[:, 1], x_pre_train[:, 1])  # 训练集
    p_Evaluation[2, 1] = r2_score(x_validation[:, 1], x_pre_validation[:, 1])  # 验证集
    p_Evaluation[2, 2] = r2_score(x_test[:, 1], x_pre_test[:, 1])  # 测试集
    p_Evaluation[2, 3] = r2_score(x[:, 1], x_pre_all[:, 1])  # 全集

    # 计算Ni的MAE
    ni_Evaluation[0, 0] = mean_absolute_error(x_train[:, 2], x_pre_train[:, 2])  # 训练集
    ni_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 2], x_pre_validation[:, 2])  # 验证集
    ni_Evaluation[0, 2] = mean_absolute_error(x_test[:, 2], x_pre_test[:, 2])  # 测试集
    ni_Evaluation[0, 3] = mean_absolute_error(x[:, 2], x_pre_all[:, 2])  # 全集
    # 计算Ni的MSE
    ni_Evaluation[1, 0] = mean_squared_error(x_train[:, 2], x_pre_train[:, 2])  # 训练集
    ni_Evaluation[1, 1] = mean_squared_error(x_validation[:, 2], x_pre_validation[:, 2])  # 验证集
    ni_Evaluation[1, 2] = mean_squared_error(x_test[:, 2], x_pre_test[:, 2])  # 测试集
    ni_Evaluation[1, 3] = mean_squared_error(x[:, 2], x_pre_all[:, 2])  # 全集
    # 计算Ni的R2
    ni_Evaluation[2, 0] = r2_score(x_train[:, 2], x_pre_train[:, 2])  # 训练集
    ni_Evaluation[2, 1] = r2_score(x_validation[:, 2], x_pre_validation[:, 2])  # 验证集
    ni_Evaluation[2, 2] = r2_score(x_test[:, 2], x_pre_test[:, 2])  # 测试集
    ni_Evaluation[2, 3] = r2_score(x[:, 2], x_pre_all[:, 2])  # 全集

    # 计算Si的MAE
    si_Evaluation[0, 0] = mean_absolute_error(x_train[:, 3], x_pre_train[:, 3])  # 训练集
    si_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 3], x_pre_validation[:, 3])  # 验证集
    si_Evaluation[0, 2] = mean_absolute_error(x_test[:, 3], x_pre_test[:, 3])  # 测试集
    si_Evaluation[0, 3] = mean_absolute_error(x[:, 3], x_pre_all[:, 3])  # 全集
    # 计算Si的MSE
    si_Evaluation[1, 0] = mean_squared_error(x_train[:, 3], x_pre_train[:, 3])  # 训练集
    si_Evaluation[1, 1] = mean_squared_error(x_validation[:, 3], x_pre_validation[:, 3])  # 验证集
    si_Evaluation[1, 2] = mean_squared_error(x_test[:, 3], x_pre_test[:, 3])  # 测试集
    si_Evaluation[1, 3] = mean_squared_error(x[:, 3], x_pre_all[:, 3])  # 全集
    # 计算Si的R2
    si_Evaluation[2, 0] = r2_score(x_train[:, 3], x_pre_train[:, 3])  # 训练集
    si_Evaluation[2, 1] = r2_score(x_validation[:, 3], x_pre_validation[:, 3])  # 验证集
    si_Evaluation[2, 2] = r2_score(x_test[:, 3], x_pre_test[:, 3])  # 测试集
    si_Evaluation[2, 3] = r2_score(x[:, 3], x_pre_all[:, 3])  # 全集

    # 计算Mg的MAE
    mg_Evaluation[0, 0] = mean_absolute_error(x_train[:, 4], x_pre_train[:, 4])  # 训练集
    mg_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 4], x_pre_validation[:, 4])  # 验证集
    mg_Evaluation[0, 2] = mean_absolute_error(x_test[:, 4], x_pre_test[:, 4])  # 测试集
    mg_Evaluation[0, 3] = mean_absolute_error(x[:, 4], x_pre_all[:, 4])  # 全集
    # 计算Mg的MSE
    mg_Evaluation[1, 0] = mean_squared_error(x_train[:, 4], x_pre_train[:, 4])  # 训练集
    mg_Evaluation[1, 1] = mean_squared_error(x_validation[:, 4], x_pre_validation[:, 4])  # 验证集
    mg_Evaluation[1, 2] = mean_squared_error(x_test[:, 4], x_pre_test[:, 4])  # 测试集
    mg_Evaluation[1, 3] = mean_squared_error(x[:, 4], x_pre_all[:, 4])  # 全集
    # 计算Mg的R2
    mg_Evaluation[2, 0] = r2_score(x_train[:, 4], x_pre_train[:, 4])  # 训练集
    mg_Evaluation[2, 1] = r2_score(x_validation[:, 4], x_pre_validation[:, 4])  # 验证集
    mg_Evaluation[2, 2] = r2_score(x_test[:, 4], x_pre_test[:, 4])  # 测试集
    mg_Evaluation[2, 3] = r2_score(x[:, 4], x_pre_all[:, 4])  # 全集

    # 计算Zn的MAE
    zn_Evaluation[0, 0] = mean_absolute_error(x_train[:, 5], x_pre_train[:, 5])  # 训练集
    zn_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 5], x_pre_validation[:, 5])  # 验证集
    zn_Evaluation[0, 2] = mean_absolute_error(x_test[:, 5], x_pre_test[:, 5])  # 测试集
    zn_Evaluation[0, 3] = mean_absolute_error(x[:, 5], x_pre_all[:, 5])  # 全集
    # 计算Zn的MSE
    zn_Evaluation[1, 0] = mean_squared_error(x_train[:, 5], x_pre_train[:, 5])  # 训练集
    zn_Evaluation[1, 1] = mean_squared_error(x_validation[:, 5], x_pre_validation[:, 5])  # 验证集
    zn_Evaluation[1, 2] = mean_squared_error(x_test[:, 5], x_pre_test[:, 5])  # 测试集
    zn_Evaluation[1, 3] = mean_squared_error(x[:, 5], x_pre_all[:, 5])  # 全集
    # 计算Zn的R2
    zn_Evaluation[2, 0] = r2_score(x_train[:, 5], x_pre_train[:, 5])  # 训练集
    zn_Evaluation[2, 1] = r2_score(x_validation[:, 5], x_pre_validation[:, 5])  # 验证集
    zn_Evaluation[2, 2] = r2_score(x_test[:, 5], x_pre_test[:, 5])  # 测试集
    zn_Evaluation[2, 3] = r2_score(x[:, 5], x_pre_all[:, 5])  # 全集

    # 计算Sn的MAE
    sn_Evaluation[0, 0] = mean_absolute_error(x_train[:, 6], x_pre_train[:, 6])  # 训练集
    sn_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 6], x_pre_validation[:, 6])  # 验证集
    sn_Evaluation[0, 2] = mean_absolute_error(x_test[:, 6], x_pre_test[:, 6])  # 测试集
    sn_Evaluation[0, 3] = mean_absolute_error(x[:, 6], x_pre_all[:, 6])  # 全集
    # 计算Sn的MSE
    sn_Evaluation[1, 0] = mean_squared_error(x_train[:, 6], x_pre_train[:, 6])  # 训练集
    sn_Evaluation[1, 1] = mean_squared_error(x_validation[:, 6], x_pre_validation[:, 6])  # 验证集
    sn_Evaluation[1, 2] = mean_squared_error(x_test[:, 6], x_pre_test[:, 6])  # 测试集
    sn_Evaluation[1, 3] = mean_squared_error(x[:, 6], x_pre_all[:, 6])  # 全集
    # 计算Sn的R2
    sn_Evaluation[2, 0] = r2_score(x_train[:, 6], x_pre_train[:, 6])  # 训练集
    sn_Evaluation[2, 1] = r2_score(x_validation[:, 6], x_pre_validation[:, 6])  # 验证集
    sn_Evaluation[2, 2] = r2_score(x_test[:, 6], x_pre_test[:, 6])  # 测试集
    sn_Evaluation[2, 3] = r2_score(x[:, 6], x_pre_all[:, 6])  # 全集

    # 计算Cr的MAE
    cr_Evaluation[0, 0] = mean_absolute_error(x_train[:, 7], x_pre_train[:, 7])  # 训练集
    cr_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 7], x_pre_validation[:, 7])  # 验证集
    cr_Evaluation[0, 2] = mean_absolute_error(x_test[:, 7], x_pre_test[:, 7])  # 测试集
    cr_Evaluation[0, 3] = mean_absolute_error(x[:, 7], x_pre_all[:, 7])  # 全集
    # 计算Cr的MSE
    cr_Evaluation[1, 0] = mean_squared_error(x_train[:, 7], x_pre_train[:, 7])  # 训练集
    cr_Evaluation[1, 1] = mean_squared_error(x_validation[:, 7], x_pre_validation[:, 7])  # 验证集
    cr_Evaluation[1, 2] = mean_squared_error(x_test[:, 7], x_pre_test[:, 7])  # 测试集
    cr_Evaluation[1, 3] = mean_squared_error(x[:, 7], x_pre_all[:, 7])  # 全集
    # 计算Cr的R2
    cr_Evaluation[2, 0] = r2_score(x_train[:, 7], x_pre_train[:, 7])  # 训练集
    cr_Evaluation[2, 1] = r2_score(x_validation[:, 7], x_pre_validation[:, 7])  # 验证集
    cr_Evaluation[2, 2] = r2_score(x_test[:, 7], x_pre_test[:, 7])  # 测试集
    cr_Evaluation[2, 3] = r2_score(x[:, 7], x_pre_all[:, 7])  # 全集

    # 计算Zr的MAE
    zr_Evaluation[0, 0] = mean_absolute_error(x_train[:, 8], x_pre_train[:, 8])  # 训练集
    zr_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 8], x_pre_validation[:, 8])  # 验证集
    zr_Evaluation[0, 2] = mean_absolute_error(x_test[:, 8], x_pre_test[:, 8])  # 测试集
    zr_Evaluation[0, 3] = mean_absolute_error(x[:, 8], x_pre_all[:, 8])  # 全集
    # 计算Zr的MSE
    zr_Evaluation[1, 0] = mean_squared_error(x_train[:, 8], x_pre_train[:, 8])  # 训练集
    zr_Evaluation[1, 1] = mean_squared_error(x_validation[:, 8], x_pre_validation[:, 8])  # 验证集
    zr_Evaluation[1, 2] = mean_squared_error(x_test[:, 8], x_pre_test[:, 8])  # 测试集
    zr_Evaluation[1, 3] = mean_squared_error(x[:, 8], x_pre_all[:, 8])  # 全集
    # 计算Zr的R2
    zr_Evaluation[2, 0] = r2_score(x_train[:, 8], x_pre_train[:, 8])  # 训练集
    zr_Evaluation[2, 1] = r2_score(x_validation[:, 8], x_pre_validation[:, 8])  # 验证集
    zr_Evaluation[2, 2] = r2_score(x_test[:, 8], x_pre_test[:, 8])  # 测试集
    zr_Evaluation[2, 3] = r2_score(x[:, 8], x_pre_all[:, 8])  # 全集

    # 计算RE的MAE
    re_Evaluation[0, 0] = mean_absolute_error(x_train[:, 9], x_pre_train[:, 9])  # 训练集
    re_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 9], x_pre_validation[:, 9])  # 验证集
    re_Evaluation[0, 2] = mean_absolute_error(x_test[:, 9], x_pre_test[:, 9])  # 测试集
    re_Evaluation[0, 3] = mean_absolute_error(x[:, 9], x_pre_all[:, 9])  # 全集
    # 计算RE的MSE
    re_Evaluation[1, 0] = mean_squared_error(x_train[:, 9], x_pre_train[:, 9])  # 训练集
    re_Evaluation[1, 1] = mean_squared_error(x_validation[:, 9], x_pre_validation[:, 9])  # 验证集
    re_Evaluation[1, 2] = mean_squared_error(x_test[:, 9], x_pre_test[:, 9])  # 测试集
    re_Evaluation[1, 3] = mean_squared_error(x[:, 9], x_pre_all[:, 9])  # 全集
    # 计算RE的R2
    re_Evaluation[2, 0] = r2_score(x_train[:, 9], x_pre_train[:, 9])  # 训练集
    re_Evaluation[2, 1] = r2_score(x_validation[:, 9], x_pre_validation[:, 9])  # 验证集
    re_Evaluation[2, 2] = r2_score(x_test[:, 9], x_pre_test[:, 9])  # 测试集
    re_Evaluation[2, 3] = r2_score(x[:, 9], x_pre_all[:, 9])  # 全集

    # 计算Al的MAE
    al_Evaluation[0, 0] = mean_absolute_error(x_train[:, 10], x_pre_train[:, 10])  # 训练集
    al_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 10], x_pre_validation[:, 10])  # 验证集
    al_Evaluation[0, 2] = mean_absolute_error(x_test[:, 10], x_pre_test[:, 10])  # 测试集
    al_Evaluation[0, 3] = mean_absolute_error(x[:, 10], x_pre_all[:, 10])  # 全集
    # 计算Al的MSE
    al_Evaluation[1, 0] = mean_squared_error(x_train[:, 10], x_pre_train[:, 10])  # 训练集
    al_Evaluation[1, 1] = mean_squared_error(x_validation[:, 10], x_pre_validation[:, 10])  # 验证集
    al_Evaluation[1, 2] = mean_squared_error(x_test[:, 10], x_pre_test[:, 10])  # 测试集
    al_Evaluation[1, 3] = mean_squared_error(x[:, 10], x_pre_all[:, 10])  # 全集
    # 计算Al的R2
    al_Evaluation[2, 0] = r2_score(x_train[:, 10], x_pre_train[:, 10])  # 训练集
    al_Evaluation[2, 1] = r2_score(x_validation[:, 10], x_pre_validation[:, 10])  # 验证集
    al_Evaluation[2, 2] = r2_score(x_test[:, 10], x_pre_test[:, 10])  # 测试集
    al_Evaluation[2, 3] = r2_score(x[:, 10], x_pre_all[:, 10])  # 全集

    # 计算Pb的MAE
    pb_Evaluation[0, 0] = mean_absolute_error(x_train[:, 11], x_pre_train[:, 11])  # 训练集
    pb_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 11], x_pre_validation[:, 11])  # 验证集
    pb_Evaluation[0, 2] = mean_absolute_error(x_test[:, 11], x_pre_test[:, 11])  # 测试集
    pb_Evaluation[0, 3] = mean_absolute_error(x[:, 11], x_pre_all[:, 11])  # 全集
    # 计算Pb的MSE
    pb_Evaluation[1, 0] = mean_squared_error(x_train[:, 11], x_pre_train[:, 11])  # 训练集
    pb_Evaluation[1, 1] = mean_squared_error(x_validation[:, 11], x_pre_validation[:, 11])  # 验证集
    pb_Evaluation[1, 2] = mean_squared_error(x_test[:, 11], x_pre_test[:, 11])  # 测试集
    pb_Evaluation[1, 3] = mean_squared_error(x[:, 11], x_pre_all[:, 11])  # 全集
    # 计算Pb的R2
    pb_Evaluation[2, 0] = r2_score(x_train[:, 11], x_pre_train[:, 11])  # 训练集
    pb_Evaluation[2, 1] = r2_score(x_validation[:, 11], x_pre_validation[:, 11])  # 验证集
    pb_Evaluation[2, 2] = r2_score(x_test[:, 11], x_pre_test[:, 11])  # 测试集
    pb_Evaluation[2, 3] = r2_score(x[:, 11], x_pre_all[:, 11])  # 全集

    # 计算Mn的MAE
    mn_Evaluation[0, 0] = mean_absolute_error(x_train[:, 12], x_pre_train[:, 12])  # 训练集
    mn_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 12], x_pre_validation[:, 12])  # 验证集
    mn_Evaluation[0, 2] = mean_absolute_error(x_test[:, 12], x_pre_test[:, 12])  # 测试集
    mn_Evaluation[0, 3] = mean_absolute_error(x[:, 12], x_pre_all[:, 12])  # 全集
    # 计算Mn的MSE
    mn_Evaluation[1, 0] = mean_squared_error(x_train[:, 12], x_pre_train[:, 12])  # 训练集
    mn_Evaluation[1, 1] = mean_squared_error(x_validation[:, 12], x_pre_validation[:, 12])  # 验证集
    mn_Evaluation[1, 2] = mean_squared_error(x_test[:, 12], x_pre_test[:, 12])  # 测试集
    mn_Evaluation[1, 3] = mean_squared_error(x[:, 12], x_pre_all[:, 12])  # 全集
    # 计算Mn的R2
    mn_Evaluation[2, 0] = r2_score(x_train[:, 12], x_pre_train[:, 12])  # 训练集
    mn_Evaluation[2, 1] = r2_score(x_validation[:, 12], x_pre_validation[:, 12])  # 验证集
    mn_Evaluation[2, 2] = r2_score(x_test[:, 12], x_pre_test[:, 12])  # 测试集
    mn_Evaluation[2, 3] = r2_score(x[:, 12], x_pre_all[:, 12])  # 全集

    # 计算Sb的MAE
    sb_Evaluation[0, 0] = mean_absolute_error(x_train[:, 13], x_pre_train[:, 13])  # 训练集
    sb_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 13], x_pre_validation[:, 13])  # 验证集
    sb_Evaluation[0, 2] = mean_absolute_error(x_test[:, 13], x_pre_test[:, 13])  # 测试集
    sb_Evaluation[0, 3] = mean_absolute_error(x[:, 13], x_pre_all[:, 13])  # 全集
    # 计算Sb的MSE
    sb_Evaluation[1, 0] = mean_squared_error(x_train[:, 13], x_pre_train[:, 13])  # 训练集
    sb_Evaluation[1, 1] = mean_squared_error(x_validation[:, 13], x_pre_validation[:, 13])  # 验证集
    sb_Evaluation[1, 2] = mean_squared_error(x_test[:, 13], x_pre_test[:, 13])  # 测试集
    sb_Evaluation[1, 3] = mean_squared_error(x[:, 13], x_pre_all[:, 13])  # 全集
    # 计算Sb的R2
    sb_Evaluation[2, 0] = r2_score(x_train[:, 13], x_pre_train[:, 13])  # 训练集
    sb_Evaluation[2, 1] = r2_score(x_validation[:, 13], x_pre_validation[:, 13])  # 验证集
    sb_Evaluation[2, 2] = r2_score(x_test[:, 13], x_pre_test[:, 13])  # 测试集
    sb_Evaluation[2, 3] = r2_score(x[:, 13], x_pre_all[:, 13])  # 全集

    # 计算S的MAE
    s_Evaluation[0, 0] = mean_absolute_error(x_train[:, 14], x_pre_train[:, 14])  # 训练集
    s_Evaluation[0, 1] = mean_absolute_error(x_validation[:, 14], x_pre_validation[:, 14])  # 验证集
    s_Evaluation[0, 2] = mean_absolute_error(x_test[:, 14], x_pre_test[:, 14])  # 测试集
    s_Evaluation[0, 3] = mean_absolute_error(x[:, 14], x_pre_all[:, 14])  # 全集
    # 计算S的MSE
    s_Evaluation[1, 0] = mean_squared_error(x_train[:, 14], x_pre_train[:, 14])  # 训练集
    s_Evaluation[1, 1] = mean_squared_error(x_validation[:, 14], x_pre_validation[:, 14])  # 验证集
    s_Evaluation[1, 2] = mean_squared_error(x_test[:, 14], x_pre_test[:, 14])  # 测试集
    s_Evaluation[1, 3] = mean_squared_error(x[:, 14], x_pre_all[:, 14])  # 全集
    # 计算S的R2
    s_Evaluation[2, 0] = r2_score(x_train[:, 14], x_pre_train[:, 14])  # 训练集
    s_Evaluation[2, 1] = r2_score(x_validation[:, 14], x_pre_validation[:, 14])  # 验证集
    s_Evaluation[2, 2] = r2_score(x_test[:, 14], x_pre_test[:, 14])  # 测试集
    s_Evaluation[2, 3] = r2_score(x[:, 14], x_pre_all[:, 14])  # 全集


# 线性拟合
def f_1(x, A, B):
    return A * x + B


# 拟合点
# 训练集
x0_train = x_train[:, 0]
y0_train = x_pre_train[:, 0]
for i in range(0, x_shape):
    x0_train = np.append(x0_train, x_train[:, i])
    y0_train = np.append(y0_train, x_pre_train[:, i])
# 验证集
x0_validation = x_validation[:, 0]
y0_validation = x_pre_validation[:, 0]
for i in range(0, x_shape):
    x0_validation = np.append(x0_validation, x_validation[:, i])
    y0_validation = np.append(y0_validation, x_pre_validation[:, i])
# 测试集
x0_test = x_test[:, 0]
y0_test = x_pre_test[:, 0]
for i in range(0, x_shape):
    x0_test = np.append(x0_test, x_test[:, i])
    y0_test = np.append(y0_test, x_pre_test[:, i])
# 全集
x0_all = x[:, 0]
y0_all = x_pre_all[:, 0]
for i in range(0, x_shape):
    x0_all = np.append(x0_all, x[:, i])
    y0_all = np.append(y0_all, x_pre_all[:, i])

# 平方误差、绝对误差、决定系数
# 训练集
ALL_Evaluation[0, 0] = mean_absolute_error(x0_train, y0_train)
ALL_Evaluation[1, 0] = mean_squared_error(x0_train, y0_train)
ALL_Evaluation[2, 0] = r2_score(x0_train, y0_train)
ALL_Evaluation[3, 0] = np.corrcoef(x0_train, y0_train)[0, 1]
# 验证集
ALL_Evaluation[0, 1] = mean_absolute_error(x0_validation, y0_validation)
ALL_Evaluation[1, 1] = mean_squared_error(x0_validation, y0_validation)
ALL_Evaluation[2, 1] = r2_score(x0_validation, y0_validation)
ALL_Evaluation[3, 1] = np.corrcoef(x0_validation, y0_validation)[0, 1]
# 测试集
ALL_Evaluation[0, 2] = mean_absolute_error(x0_test, y0_test)
ALL_Evaluation[1, 2] = mean_squared_error(x0_test, y0_test)
ALL_Evaluation[2, 2] = r2_score(x0_test, y0_test)
ALL_Evaluation[3, 2] = np.corrcoef(x0_test, y0_test)[0, 1]
# 全集
ALL_Evaluation[0, 3] = mean_absolute_error(x0_all, y0_all)
ALL_Evaluation[1, 3] = mean_squared_error(x0_all, y0_all)
ALL_Evaluation[2, 3] = r2_score(x0_all, y0_all)
ALL_Evaluation[3, 3] = np.corrcoef(x0_all, y0_all)[0, 1]

# 直线拟合与绘制
# 训练集
A_train, B_train = optimize.curve_fit(f_1, x0_train, y0_train)[0]
x1_train = np.arange(-100, 100)
y1_train = A_train * x1_train + B_train
ALL_Evaluation[4, 0] = A_train
# 验证集
A_validation, B_validation = optimize.curve_fit(f_1, x0_validation, y0_validation)[0]
x1_validation = np.arange(-100, 100)
y1_validation = A_validation * x1_validation + B_validation
ALL_Evaluation[4, 1] = A_validation
# 测试集
A_test, B_test = optimize.curve_fit(f_1, x0_test, y0_test)[0]
x1_test = np.arange(-100, 100)
y1_test = A_test * x1_test + B_test
ALL_Evaluation[4, 2] = A_test
# 全集
A_all, B_all = optimize.curve_fit(f_1, x0_all, y0_all)[0]
x1_all = np.arange(-100, 100)
y1_all = A_all * x1_all + B_all
ALL_Evaluation[4, 3] = A_all

# 绘图
# 设置参考的1：1虚线参数
xxx = [-100, 100]
yyy = [-100, 100]
font = {'family': 'Times New Roman', 'size': 12}
sns.set(style='ticks')
# 绘制散点图 (训练集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(x_train, x_pre_train, s=30, c='red', edgecolors='k', marker='o', alpha=0.8)
plt.plot(x1_train, y1_train, "black")  # 绘制拟合直线
plt.title('P2C ScatterPlot (train set) R2: %.4f' % ALL_Evaluation[2, 0], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('Train', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.xticks(font=font)
plt.xlim((-1, 20))  # 设置坐标轴范围
plt.ylim((-1, 20))
plt.show()

# 绘制散点图 (验证集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(x_validation, x_pre_validation, s=30, c='yellow', edgecolors='k', marker='o', alpha=0.8)
plt.plot(x1_validation, y1_validation, "black")  # 绘制拟合直线
plt.title('P2C ScatterPlot (validation set) R2: %.4f' % ALL_Evaluation[2, 1], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('Validation', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.xticks(font=font)
plt.xlim((-1, 20))  # 设置坐标轴范围
plt.ylim((-1, 20))
plt.show()

# 绘制散点图 (测试集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(x_test, x_pre_test, s=30, c='blue', edgecolors='k', marker='o', alpha=0.8)
plt.plot(x1_test, y1_test, "black")  # 绘制拟合直线
plt.title('P2C ScatterPlot (test set) R2: %.4f' % ALL_Evaluation[2, 2], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('Test', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.xticks(font=font)
plt.xlim((-1, 10))  # 设置坐标轴范围
plt.ylim((-1, 10))
plt.show()

# 绘制散点图 (全集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(x, x_pre_all, s=30, c='green', edgecolors='k', marker='o', alpha=0.8)
plt.plot(x1_all, y1_all, "black")  # 绘制拟合直线
plt.title('P2C ScatterPlot (all set) R2: %.4f' % ALL_Evaluation[2, 3], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('All', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.xticks(font=font)
plt.xlim((-1, 20))  # 设置坐标轴范围
plt.ylim((-1, 20))
plt.show()

# 输出表格 (训练集)
print('P2C-Train Set')
table = pt.PrettyTable(['Index', 'MAE', 'MSE', 'R2', 'R', 'Fitting'])
table.add_row(['All', '%.4f' % ALL_Evaluation[0, 0], '%.4f' % ALL_Evaluation[1, 0], '%.4f' % ALL_Evaluation[2, 0],
               '%.4f' % ALL_Evaluation[3, 0], '%.4f' % ALL_Evaluation[4, 0]])
table.add_row(
    ['Fe', '%.4f' % fe_Evaluation[0, 0], '%.4f' % fe_Evaluation[1, 0], '%.4f' % fe_Evaluation[2, 0], ' ', ' '])
table.add_row(['P', '%.4f' % p_Evaluation[0, 0], '%.4f' % p_Evaluation[1, 0], '%.4f' % p_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Ni', '%.4f' % ni_Evaluation[0, 0], '%.4f' % ni_Evaluation[1, 0], '%.4f' % ni_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Si', '%.4f' % si_Evaluation[0, 0], '%.4f' % si_Evaluation[1, 0], '%.4f' % si_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Mg', '%.4f' % mg_Evaluation[0, 0], '%.4f' % mg_Evaluation[1, 0], '%.4f' % mg_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Zn', '%.4f' % zn_Evaluation[0, 0], '%.4f' % zn_Evaluation[1, 0], '%.4f' % zn_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Sn', '%.4f' % sn_Evaluation[0, 0], '%.4f' % sn_Evaluation[1, 0], '%.4f' % sn_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Cr', '%.4f' % cr_Evaluation[0, 0], '%.4f' % cr_Evaluation[1, 0], '%.4f' % cr_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Zr', '%.4f' % zr_Evaluation[0, 0], '%.4f' % zr_Evaluation[1, 0], '%.4f' % zr_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['RE', '%.4f' % re_Evaluation[0, 0], '%.4f' % re_Evaluation[1, 0], '%.4f' % re_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Al', '%.4f' % al_Evaluation[0, 0], '%.4f' % al_Evaluation[1, 0], '%.4f' % al_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Pb', '%.4f' % pb_Evaluation[0, 0], '%.4f' % pb_Evaluation[1, 0], '%.4f' % pb_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Mn', '%.4f' % mn_Evaluation[0, 0], '%.4f' % mn_Evaluation[1, 0], '%.4f' % mn_Evaluation[2, 0], ' ', ' '])
table.add_row(
    ['Sb', '%.4f' % sb_Evaluation[0, 0], '%.4f' % sb_Evaluation[1, 0], '%.4f' % sb_Evaluation[2, 0], ' ', ' '])
table.add_row(['S', '%.4f' % s_Evaluation[0, 0], '%.4f' % s_Evaluation[1, 0], '%.4f' % s_Evaluation[2, 0], ' ', ' '])
print(table)

# 输出表格 (验证集)
print('P2C-Alidation Set')
table = pt.PrettyTable(['Index', 'MAE', 'MSE', 'R2', 'R', 'Fitting'])
table.add_row(['All', '%.4f' % ALL_Evaluation[0, 1], '%.4f' % ALL_Evaluation[1, 1], '%.4f' % ALL_Evaluation[2, 1],
               '%.4f' % ALL_Evaluation[3, 1], '%.4f' % ALL_Evaluation[4, 1]])
table.add_row(
    ['Fe', '%.4f' % fe_Evaluation[0, 1], '%.4f' % fe_Evaluation[1, 1], '%.4f' % fe_Evaluation[2, 1], ' ', ' '])
table.add_row(['P', '%.4f' % p_Evaluation[0, 1], '%.4f' % p_Evaluation[1, 1], '%.4f' % p_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Ni', '%.4f' % ni_Evaluation[0, 1], '%.4f' % ni_Evaluation[1, 1], '%.4f' % ni_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Si', '%.4f' % si_Evaluation[0, 1], '%.4f' % si_Evaluation[1, 1], '%.4f' % si_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Mg', '%.4f' % mg_Evaluation[0, 1], '%.4f' % mg_Evaluation[1, 1], '%.4f' % mg_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Zn', '%.4f' % zn_Evaluation[0, 1], '%.4f' % zn_Evaluation[1, 1], '%.4f' % zn_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Sn', '%.4f' % sn_Evaluation[0, 1], '%.4f' % sn_Evaluation[1, 1], '%.4f' % sn_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Cr', '%.4f' % cr_Evaluation[0, 1], '%.4f' % cr_Evaluation[1, 1], '%.4f' % cr_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Zr', '%.4f' % zr_Evaluation[0, 1], '%.4f' % zr_Evaluation[1, 1], '%.4f' % zr_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['RE', '%.4f' % re_Evaluation[0, 1], '%.4f' % re_Evaluation[1, 1], '%.4f' % re_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Al', '%.4f' % al_Evaluation[0, 1], '%.4f' % al_Evaluation[1, 1], '%.4f' % al_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Pb', '%.4f' % pb_Evaluation[0, 1], '%.4f' % pb_Evaluation[1, 1], '%.4f' % pb_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Mn', '%.4f' % mn_Evaluation[0, 1], '%.4f' % mn_Evaluation[1, 1], '%.4f' % mn_Evaluation[2, 1], ' ', ' '])
table.add_row(
    ['Sb', '%.4f' % sb_Evaluation[0, 1], '%.4f' % sb_Evaluation[1, 1], '%.4f' % sb_Evaluation[2, 1], ' ', ' '])
table.add_row(['S', '%.4f' % s_Evaluation[0, 1], '%.4f' % s_Evaluation[1, 1], '%.4f' % s_Evaluation[2, 1], ' ', ' '])
print(table)

# 输出表格 (训练集)
print('P2C-Test Set')
table = pt.PrettyTable(['Index', 'MAE', 'MSE', 'R2', 'R', 'Fitting'])
table.add_row(['All', '%.4f' % ALL_Evaluation[0, 2], '%.4f' % ALL_Evaluation[1, 2], '%.4f' % ALL_Evaluation[2, 2],
               '%.4f' % ALL_Evaluation[3, 2], '%.4f' % ALL_Evaluation[4, 2]])
table.add_row(
    ['Fe', '%.4f' % fe_Evaluation[0, 2], '%.4f' % fe_Evaluation[1, 2], '%.4f' % fe_Evaluation[2, 2], ' ', ' '])
table.add_row(['P', '%.4f' % p_Evaluation[0, 2], '%.4f' % p_Evaluation[1, 2], '%.4f' % p_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Ni', '%.4f' % ni_Evaluation[0, 2], '%.4f' % ni_Evaluation[1, 2], '%.4f' % ni_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Si', '%.4f' % si_Evaluation[0, 2], '%.4f' % si_Evaluation[1, 2], '%.4f' % si_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Mg', '%.4f' % mg_Evaluation[0, 2], '%.4f' % mg_Evaluation[1, 2], '%.4f' % mg_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Zn', '%.4f' % zn_Evaluation[0, 2], '%.4f' % zn_Evaluation[1, 2], '%.4f' % zn_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Sn', '%.4f' % sn_Evaluation[0, 2], '%.4f' % sn_Evaluation[1, 2], '%.4f' % sn_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Cr', '%.4f' % cr_Evaluation[0, 2], '%.4f' % cr_Evaluation[1, 2], '%.4f' % cr_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Zr', '%.4f' % zr_Evaluation[0, 2], '%.4f' % zr_Evaluation[1, 2], '%.4f' % zr_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['RE', '%.4f' % re_Evaluation[0, 2], '%.4f' % re_Evaluation[1, 2], '%.4f' % re_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Al', '%.4f' % al_Evaluation[0, 2], '%.4f' % al_Evaluation[1, 2], '%.4f' % al_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Pb', '%.4f' % pb_Evaluation[0, 2], '%.4f' % pb_Evaluation[1, 2], '%.4f' % pb_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Mn', '%.4f' % mn_Evaluation[0, 2], '%.4f' % mn_Evaluation[1, 2], '%.4f' % mn_Evaluation[2, 2], ' ', ' '])
table.add_row(
    ['Sb', '%.4f' % sb_Evaluation[0, 2], '%.4f' % sb_Evaluation[1, 2], '%.4f' % sb_Evaluation[2, 2], ' ', ' '])
table.add_row(['S', '%.4f' % s_Evaluation[0, 2], '%.4f' % s_Evaluation[1, 2], '%.4f' % s_Evaluation[2, 2], ' ', ' '])
print(table)

# 输出表格 (全集)
print('P2C-All Set')
table = pt.PrettyTable(['Index', 'MAE', 'MSE', 'R2', 'R', 'Fitting'])
table.add_row(['All', '%.4f' % ALL_Evaluation[0, 3], '%.4f' % ALL_Evaluation[1, 3], '%.4f' % ALL_Evaluation[2, 3],
               '%.4f' % ALL_Evaluation[3, 3], '%.4f' % ALL_Evaluation[4, 3]])
table.add_row(
    ['Fe', '%.4f' % fe_Evaluation[0, 3], '%.4f' % fe_Evaluation[1, 3], '%.4f' % fe_Evaluation[2, 3], ' ', ' '])
table.add_row(['P', '%.4f' % p_Evaluation[0, 3], '%.4f' % p_Evaluation[1, 3], '%.4f' % p_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Ni', '%.4f' % ni_Evaluation[0, 3], '%.4f' % ni_Evaluation[1, 3], '%.4f' % ni_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Si', '%.4f' % si_Evaluation[0, 3], '%.4f' % si_Evaluation[1, 3], '%.4f' % si_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Mg', '%.4f' % mg_Evaluation[0, 3], '%.4f' % mg_Evaluation[1, 3], '%.4f' % mg_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Zn', '%.4f' % zn_Evaluation[0, 3], '%.4f' % zn_Evaluation[1, 3], '%.4f' % zn_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Sn', '%.4f' % sn_Evaluation[0, 3], '%.4f' % sn_Evaluation[1, 3], '%.4f' % sn_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Cr', '%.4f' % cr_Evaluation[0, 3], '%.4f' % cr_Evaluation[1, 3], '%.4f' % cr_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Zr', '%.4f' % zr_Evaluation[0, 3], '%.4f' % zr_Evaluation[1, 3], '%.4f' % zr_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['RE', '%.4f' % re_Evaluation[0, 3], '%.4f' % re_Evaluation[1, 3], '%.4f' % re_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Al', '%.4f' % al_Evaluation[0, 3], '%.4f' % al_Evaluation[1, 3], '%.4f' % al_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Pb', '%.4f' % pb_Evaluation[0, 3], '%.4f' % pb_Evaluation[1, 3], '%.4f' % pb_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Mn', '%.4f' % mn_Evaluation[0, 3], '%.4f' % mn_Evaluation[1, 3], '%.4f' % mn_Evaluation[2, 3], ' ', ' '])
table.add_row(
    ['Sb', '%.4f' % sb_Evaluation[0, 3], '%.4f' % sb_Evaluation[1, 3], '%.4f' % sb_Evaluation[2, 3], ' ', ' '])
table.add_row(['S', '%.4f' % s_Evaluation[0, 3], '%.4f' % s_Evaluation[1, 3], '%.4f' % s_Evaluation[2, 3], ' ', ' '])
print(table)
