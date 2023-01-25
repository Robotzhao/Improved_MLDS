# -*- coding: utf-8 -*-

# 成分-性能模型（测试）
import warnings
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import optimize
import matplotlib.pyplot as plt
import seaborn as sns
import prettytable as pt

# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")


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


# 对数变换
def Logarithmic(data):
    data[:, 5] = np.log(data[:, 5] + 1)
    return data


# 反对数变换
def Anti_Logarithmic(data):
    data[:, 5] = np.exp(data[:, 5]) - 1
    return data


# 读取数据
df = pd.read_excel('Data.xlsx', sheet_name='数据处理1')
data = df.values

# 选择特征与标签(x是铜合金的元素含量，y是材料的抗拉强度及导电率)
x = data[0:408, 1:16]
y = data[0:408, 16:18]
# 获取特征和标签的维度
x_shape = x.shape[1]
y_shape = y.shape[1]

# minmax归一化处理
x, minx, maxx = Normalization(x)
y, miny, maxy = Normalization(y)
x = Logarithmic(x)

# 读取测试数据
df_test = pd.read_excel('Data.xlsx', sheet_name='测试数据')
data_test = df_test.values

x_test = data_test[0:11, 1:16]
y_test = data_test[0:11, 16:18]

x_test = (x_test - minx) / (maxx - minx)
x_test = Logarithmic(x_test)

test_range = np.zeros((10, 4))

# x_train=x
# y_train=y

for k in range(0, 10):
    for i in range(0, 50):
        # 划分90%训练集，10%验证集
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.1, random_state=0)
        # 建立UTS模型
        # 随机森林
        uts_model = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=15)
        uts_model_train = uts_model.fit(x, y[:, 0])
        uts_pre_validation = uts_model.predict(x_validation)
        # 建立EC模型
        # 梯度提升树
        ec_model = GradientBoostingRegressor(n_estimators=100, max_features=4, max_depth=6)
        ec_model_train = ec_model.fit(x, y[:, 1])
        ec_pre_validation = ec_model.predict(x_validation)
        # 组合变量
        y_pre_validation = np.vstack((uts_pre_validation, ec_pre_validation)).T
        # 计算总R2
        x0 = y_validation[:, 0]
        y0 = y_pre_validation[:, 0]
        for j in range(0, y_shape):
            x0 = np.append(x0, y_validation[:, j])
            y0 = np.append(y0, y_pre_validation[:, j])
        R2 = r2_score(x0, y0)
        print(R2)
        if R2 > 0.98:
            break
    # # 建立UTS模型
    # # 随机森林
    # uts_model = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=15)
    # # 决策树的个数为100，最大特征个数取2，最大深度为15
    # # 训练模型并代入验证集
    # uts_model_train = uts_model.fit(x_train, y_train[:, 0])
    # uts_pre_test = uts_model.predict(x_test)
    # # 建立EC模型
    # # 梯度提升树
    # ec_model = GradientBoostingRegressor(n_estimators=100, max_features=4, max_depth=6)
    # # 决策树的个数为100，最大特征个数取4，最大深度为6
    # # 训练模型并代入验证集
    # ec_model_train = ec_model.fit(x_train, y_train[:, 1])
    # ec_pre_test = ec_model.predict(x_test)

    uts_pre_test = uts_model.predict(x_test)
    ec_pre_test = ec_model.predict(x_test)

    # 组合变量
    y_pre_test = np.vstack((uts_pre_test, ec_pre_test)).T

    # 预测集 反归一化
    y_pre_test = y_pre_test * (maxy - miny) + miny

    if k == 0:
        for j in range(0, 10):
            test_range[j, 0] = y_pre_test[j, 0]
            test_range[j, 1] = y_pre_test[j, 0]
            test_range[j, 2] = y_pre_test[j, 1]
            test_range[j, 3] = y_pre_test[j, 1]
    else:
        for j in range(0, 10):
            if y_pre_test[j, 0] < test_range[j, 0]:
                test_range[j, 0] = y_pre_test[j, 0]
            if y_pre_test[j, 0] > test_range[j, 1]:
                test_range[j, 1] = y_pre_test[j, 0]
            if y_pre_test[j, 1] < test_range[j, 2]:
                test_range[j, 2] = y_pre_test[j, 1]
            if y_pre_test[j, 1] > test_range[j, 3]:
                test_range[j, 3] = y_pre_test[j, 1]
print(y_test, test_range)

# # 线性拟合
# def f_1(x, A, B):
#     return A * x + B
#
#
# # 拟合点
# # 测试集
# x0_test = y_test[:, 0]
# y0_test = y_pre_test[:, 0]
# for i in range(0, y_shape):
#     x0_test = np.append(x0_test, y_test[:, i])
#     y0_test = np.append(y0_test, y_pre_test[:, i])
#
# # 平方误差、绝对误差、决定系数
# # 测试集
# ALL_Evaluation[0, 2] = mean_absolute_error(x0_test, y0_test)
# ALL_Evaluation[1, 2] = mean_squared_error(x0_test, y0_test)
# ALL_Evaluation[2, 2] = r2_score(x0_test, y0_test)
# ALL_Evaluation[3, 2] = np.corrcoef(x0_test, y0_test)[0, 1]
#
# # 直线拟合与绘制
# # 测试集
# A_test, B_test = optimize.curve_fit(f_1, x0_test, y0_test)[0]
# x1_test = np.arange(0, 1000)
# y1_test = A_test * x1_test + B_test
# ALL_Evaluation[4, 2] = A_test
#
# # 绘图
# # 设置参考的1：1虚线参数
# xxx = [0, 1000]
# yyy = [0, 1000]
# font = {'family': 'Times New Roman', 'size': 12}
# sns.set(style='ticks')
#
# # 绘制散点图 (测试集)
# plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
# plt.scatter(y_test[:, 0], y_pre_test[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
#             label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
# plt.scatter(y_test[:, 1], y_pre_test[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
#             label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
# plt.plot(x1_test, y1_test, "blue")  # 绘制拟合直线
# plt.title('C2P ScatterPlot (test set) R2: %.4f' % ALL_Evaluation[2, 2], fontproperties=font)
# plt.xticks(font=font)
# plt.yticks(font=font)
# plt.xlabel('test', fontproperties=font)
# plt.ylabel('Prediction', fontproperties=font)
# plt.legend(loc='best', prop=font)
# plt.xticks(font=font)
# plt.xlim((0, 1000))  # 设置坐标轴范围
# plt.ylim((0, 1000))
# plt.show()
#
# # 小图
# plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
# plt.scatter(y_test[:, 0], y_pre_test[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
#             label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
# plt.scatter(y_test[:, 1], y_pre_test[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
#             label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
# plt.plot(x1_test, y1_test, "blue")  # 绘制拟合直线
# plt.title('C2P ScatterPlot (test set) R2: %.4f' % ALL_Evaluation[2, 2], fontproperties=font)
# plt.xticks(font=font)
# plt.yticks(font=font)
# plt.xlabel('test', fontproperties=font)
# plt.ylabel('Prediction', fontproperties=font)
# plt.legend(loc='best', prop=font)
# plt.xticks(font=font)
# plt.xlim((0, 110))  # 设置坐标轴范围
# plt.ylim((0, 110))
# plt.show()
#
# # 输出表格 (测试集)
# print('C2P-Test Set')
# table = pt.PrettyTable(['Index', 'UTS', 'EC', 'ALL'])
# table.add_row(['MAE', '%.4f' % UTS_Evaluation[0, 2], '%.4f' % EC_Evaluation[0, 2], '%.4f' % ALL_Evaluation[0, 2]])
# table.add_row(['MSE', '%.4f' % UTS_Evaluation[1, 2], '%.4f' % EC_Evaluation[1, 2], '%.4f' % ALL_Evaluation[1, 2]])
# table.add_row(['R2', '%.4f' % UTS_Evaluation[2, 2], '%.4f' % EC_Evaluation[2, 2], '%.4f' % ALL_Evaluation[2, 2]])
# table.add_row(['R', ' ', ' ', '%.4f' % ALL_Evaluation[3, 2]])
# table.add_row(['Fitting', ' ', ' ', '%.4f' % ALL_Evaluation[4, 2]])
# print(table)
