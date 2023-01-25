# -*- coding: utf-8 -*-

# 成分-性能模型
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
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)
y = min_max_scaler.fit_transform(y)
# 获取y的最值，便于后面进行反归一化
miny = min_max_scaler.data_min_
maxy = min_max_scaler.data_max_

# 划分90%训练集，10%测试集
x_train_, x_test, y_train_, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# 10折交叉检验划分训练集与验证集

UTS_Evaluation = np.zeros((4, 4))
EC_Evaluation = np.zeros((4, 4))
ALL_Evaluation = np.zeros((6, 4))

y_test = y_test * (maxy - miny) + miny
y = y * (maxy - miny) + miny


def mape(actual, pred):
    return np.mean(abs(actual - pred) / actual) * 100


for i in range(0, 10):
    KF = KFold(n_splits=10, shuffle=False)
    for train_index, validation_index in KF.split(x_train_):
        x_train, x_validation = x_train_[train_index], x_train_[validation_index]
        y_train, y_validation = y_train_[train_index], y_train_[validation_index]

        # 建立UTS模型

        # 随机森林
        uts_model = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=15)
        # 决策树的个数为100，最大特征个数取2，最大深度为15
        # 训练模型并代入验证集
        uts_model_train = uts_model.fit(x_train, y_train[:, 0])
        uts_pre_train = uts_model.predict(x_train)
        uts_pre_validation = uts_model.predict(x_validation)
        uts_pre_test = uts_model.predict(x_test)
        uts_pre_all = uts_model.predict(x)

        # 建立EC模型

        # 梯度提升树
        ec_model = GradientBoostingRegressor(n_estimators=100, max_features=4, max_depth=6)
        # 决策树的个数为100，最大特征个数取4，最大深度为6
        # 训练模型并代入验证集
        ec_model_train = ec_model.fit(x_train, y_train[:, 1])
        ec_pre_train = ec_model.predict(x_train)
        ec_pre_validation = ec_model.predict(x_validation)
        ec_pre_test = ec_model.predict(x_test)
        ec_pre_all = ec_model.predict(x)

        # 训练集/验证集/测试集/全集 反归一化
        y_train = y_train * (maxy - miny) + miny
        y_validation = y_validation * (maxy - miny) + miny

        # print(y_test)

        # UTS 反归一化
        uts_pre_train = uts_pre_train * (maxy[0] - miny[0]) + miny[0]
        uts_pre_validation = uts_pre_validation * (maxy[0] - miny[0]) + miny[0]
        uts_pre_test = uts_pre_test * (maxy[0] - miny[0]) + miny[0]
        uts_pre_all = uts_pre_all * (maxy[0] - miny[0]) + miny[0]

        # EC 反归一化
        ec_pre_train = ec_pre_train * (maxy[1] - miny[1]) + miny[1]
        ec_pre_validation = ec_pre_validation * (maxy[1] - miny[1]) + miny[1]
        ec_pre_test = ec_pre_test * (maxy[1] - miny[1]) + miny[1]
        ec_pre_all = ec_pre_all * (maxy[1] - miny[1]) + miny[1]

        # 计算UTS的MAE
        UTS_Evaluation[0, 0] = UTS_Evaluation[0, 0] + mean_absolute_error(y_train[:, 0], uts_pre_train)  # 训练集
        UTS_Evaluation[0, 1] = UTS_Evaluation[0, 1] + mean_absolute_error(y_validation[:, 0], uts_pre_validation)  # 验证集
        UTS_Evaluation[0, 2] = UTS_Evaluation[0, 2] + mean_absolute_error(y_test[:, 0], uts_pre_test)  # 测试集
        UTS_Evaluation[0, 3] = UTS_Evaluation[0, 3] + mean_absolute_error(y[:, 0], uts_pre_all)  # 全集
        # 计算UTS的MSE
        UTS_Evaluation[1, 0] = UTS_Evaluation[1, 0] + mean_squared_error(y_train[:, 0], uts_pre_train)  # 训练集
        UTS_Evaluation[1, 1] = UTS_Evaluation[1, 1] + mean_squared_error(y_validation[:, 0], uts_pre_validation)  # 验证集
        UTS_Evaluation[1, 2] = UTS_Evaluation[1, 2] + mean_squared_error(y_test[:, 0], uts_pre_test)  # 测试集
        UTS_Evaluation[1, 3] = UTS_Evaluation[1, 3] + mean_squared_error(y[:, 0], uts_pre_all)  # 全集
        # 计算UTS的R2
        UTS_Evaluation[2, 0] = UTS_Evaluation[2, 0] + r2_score(y_train[:, 0], uts_pre_train)  # 训练集
        UTS_Evaluation[2, 1] = UTS_Evaluation[2, 1] + r2_score(y_validation[:, 0], uts_pre_validation)  # 验证集
        UTS_Evaluation[2, 2] = UTS_Evaluation[2, 2] + r2_score(y_test[:, 0], uts_pre_test)  # 测试集
        UTS_Evaluation[2, 3] = UTS_Evaluation[2, 3] + r2_score(y[:, 0], uts_pre_all)  # 全集
        # 计算UTS的MAPE
        UTS_Evaluation[3, 0] = UTS_Evaluation[3, 0] + mape(y_train[:, 0], uts_pre_train)  # 训练集
        UTS_Evaluation[3, 1] = UTS_Evaluation[3, 1] + mape(y_validation[:, 0], uts_pre_validation)  # 验证集
        UTS_Evaluation[3, 2] = UTS_Evaluation[3, 2] + mape(y_test[:, 0], uts_pre_test)  # 测试集
        UTS_Evaluation[3, 3] = UTS_Evaluation[3, 3] + mape(y[:, 0], uts_pre_all)  # 全集

        # 计算EC的MAE
        EC_Evaluation[0, 0] = EC_Evaluation[0, 0] + mean_absolute_error(y_train[:, 1], ec_pre_train)  # 训练集
        EC_Evaluation[0, 1] = EC_Evaluation[0, 1] + mean_absolute_error(y_validation[:, 1], ec_pre_validation)  # 验证集
        EC_Evaluation[0, 2] = EC_Evaluation[0, 2] + mean_absolute_error(y_test[:, 1], ec_pre_test)  # 测试集
        EC_Evaluation[0, 3] = EC_Evaluation[0, 3] + mean_absolute_error(y[:, 1], ec_pre_all)  # 全集
        # 计算EC的MSE
        EC_Evaluation[1, 0] = EC_Evaluation[1, 0] + mean_squared_error(y_train[:, 1], ec_pre_train)  # 训练集
        EC_Evaluation[1, 1] = EC_Evaluation[1, 1] + mean_squared_error(y_validation[:, 1], ec_pre_validation)  # 验证集
        EC_Evaluation[1, 2] = EC_Evaluation[1, 2] + mean_squared_error(y_test[:, 1], ec_pre_test)  # 测试集
        EC_Evaluation[1, 3] = EC_Evaluation[1, 3] + mean_squared_error(y[:, 1], ec_pre_all)  # 全集
        # 计算EC的R2
        EC_Evaluation[2, 0] = EC_Evaluation[2, 0] + r2_score(y_train[:, 1], ec_pre_train)  # 训练集
        EC_Evaluation[2, 1] = EC_Evaluation[2, 1] + r2_score(y_validation[:, 1], ec_pre_validation)  # 验证集
        EC_Evaluation[2, 2] = EC_Evaluation[2, 2] + r2_score(y_test[:, 1], ec_pre_test)  # 测试集
        EC_Evaluation[2, 3] = EC_Evaluation[2, 3] + r2_score(y[:, 1], ec_pre_all)  # 全集
        # 计算EC的MAPE
        EC_Evaluation[3, 0] = EC_Evaluation[3, 0] + mape(y_train[:, 1], ec_pre_train)  # 训练集
        EC_Evaluation[3, 1] = EC_Evaluation[3, 1] + mape(y_validation[:, 1], ec_pre_validation)  # 验证集
        EC_Evaluation[3, 2] = EC_Evaluation[3, 2] + mape(y_test[:, 1], ec_pre_test)  # 测试集
        EC_Evaluation[3, 3] = EC_Evaluation[3, 3] + mape(y[:, 1], ec_pre_all)  # 全集

    # 取10次训练结果的平均值
    UTS_Evaluation = UTS_Evaluation / 10
    EC_Evaluation = EC_Evaluation / 10

y_pre_train = np.vstack((uts_pre_train, ec_pre_train)).T
y_pre_validation = np.vstack((uts_pre_validation, ec_pre_validation)).T
y_pre_test = np.vstack((uts_pre_test, ec_pre_test)).T
y_pre_all = np.vstack((uts_pre_all, ec_pre_all)).T


# 线性拟合
def f_1(x, A, B):
    return A * x + B


# 拟合点
# 训练集
x0_train = y_train[:, 0]
y0_train = y_pre_train[:, 0]
for i in range(0, y_shape):
    x0_train = np.append(x0_train, y_train[:, i])
    y0_train = np.append(y0_train, y_pre_train[:, i])
# 验证集
x0_validation = y_validation[:, 0]
y0_validation = y_pre_validation[:, 0]
for i in range(0, y_shape):
    x0_validation = np.append(x0_validation, y_validation[:, i])
    y0_validation = np.append(y0_validation, y_pre_validation[:, i])
# 测试集
x0_test = y_test[:, 0]
y0_test = y_pre_test[:, 0]
for i in range(0, y_shape):
    x0_test = np.append(x0_test, y_test[:, i])
    y0_test = np.append(y0_test, y_pre_test[:, i])
# 全集
x0_all = y[:, 0]
y0_all = y_pre_all[:, 0]
for i in range(0, y_shape):
    x0_all = np.append(x0_all, y[:, i])
    y0_all = np.append(y0_all, y_pre_all[:, i])

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
x1_train = np.arange(0, 1000)
y1_train = A_train * x1_train + B_train
ALL_Evaluation[4, 0] = A_train
# 验证集
A_validation, B_validation = optimize.curve_fit(f_1, x0_validation, y0_validation)[0]
x1_validation = np.arange(0, 1000)
y1_validation = A_validation * x1_validation + B_validation
ALL_Evaluation[4, 1] = A_validation
# 测试集
A_test, B_test = optimize.curve_fit(f_1, x0_test, y0_test)[0]
x1_test = np.arange(0, 1000)
y1_test = A_test * x1_test + B_test
ALL_Evaluation[4, 2] = A_test
# 全集
A_all, B_all = optimize.curve_fit(f_1, x0_all, y0_all)[0]
x1_all = np.arange(0, 1000)
y1_all = A_all * x1_all + B_all
ALL_Evaluation[4, 3] = A_all

ALL_Evaluation[5, 0] = mape(x0_train, y0_train)
ALL_Evaluation[5, 1] = mape(x0_validation, y0_validation)
ALL_Evaluation[5, 2] = mape(x0_test, y0_test)
ALL_Evaluation[5, 3] = mape(x0_all, y0_all)

# 绘图
# 设置参考的1：1虚线参数
xxx = [0, 1000]
yyy = [0, 1000]
font = {'family': 'Times New Roman', 'size': 12}
sns.set(style='ticks')
# 绘制散点图 (训练集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_train[:, 0], y_pre_train[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_train[:, 1], y_pre_train[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_train, y1_train, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (train set) R2: %.4f' % ALL_Evaluation[2, 0], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('Train', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 1000))  # 设置坐标轴范围
plt.ylim((0, 1000))
plt.show()

# 小图
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_train[:, 0], y_pre_train[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_train[:, 1], y_pre_train[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_train, y1_train, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (train set) R2: %.4f' % ALL_Evaluation[2, 0], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('Train', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 110))  # 设置坐标轴范围
plt.ylim((0, 110))
plt.show()

# 绘制散点图 (验证集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_validation[:, 0], y_pre_validation[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_validation[:, 1], y_pre_validation[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_validation, y1_validation, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (validation set) R2: %.4f' % ALL_Evaluation[2, 1], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('validation', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 1000))  # 设置坐标轴范围
plt.ylim((0, 1000))
plt.show()

# 小图
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_validation[:, 0], y_pre_validation[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_validation[:, 1], y_pre_validation[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_validation, y1_validation, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (validation set) R2: %.4f' % ALL_Evaluation[2, 1], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('validation', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 110))  # 设置坐标轴范围
plt.ylim((0, 110))
plt.show()

# 绘制散点图 (测试集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_test[:, 0], y_pre_test[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_test[:, 1], y_pre_test[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_test, y1_test, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (test set) R2: %.4f' % ALL_Evaluation[2, 2], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('test', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 1000))  # 设置坐标轴范围
plt.ylim((0, 1000))
plt.show()

# 小图
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_test[:, 0], y_pre_test[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_test[:, 1], y_pre_test[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_test, y1_test, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (test set) R2: %.4f' % ALL_Evaluation[2, 2], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('test', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 110))  # 设置坐标轴范围
plt.ylim((0, 110))
plt.show()

# 绘制散点图 (全集)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y[:, 0], y_pre_all[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y[:, 1], y_pre_all[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_all, y1_all, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (all set) R2: %.4f' % ALL_Evaluation[2, 3], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('all', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 1000))  # 设置坐标轴范围
plt.ylim((0, 1000))
plt.show()

# 小图
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y[:, 0], y_pre_all[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y[:, 1], y_pre_all[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x1_all, y1_all, "blue")  # 绘制拟合直线
plt.title('C2P ScatterPlot (all set) R2: %.4f' % ALL_Evaluation[2, 3], fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('all', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 110))  # 设置坐标轴范围
plt.ylim((0, 110))
plt.show()

# # 输出表格 (训练集)
# print('C2P-Train Set')
# table = pt.PrettyTable(['Index', 'UTS', 'EC', 'ALL'])
# table.add_row(['MAE', '%.4f' % UTS_Evaluation[0,0], '%.4f' % EC_Evaluation[0,0], '%.4f' % ALL_Evaluation[0,0]])
# table.add_row(['MSE', '%.4f' % UTS_Evaluation[1,0], '%.4f' % EC_Evaluation[1,0], '%.4f' % ALL_Evaluation[1,0]])
# table.add_row(['R2', '%.4f' % UTS_Evaluation[2,0], '%.4f' % EC_Evaluation[2,0], '%.4f' % ALL_Evaluation[2,0]])
# table.add_row(['R', ' ', ' ', '%.4f' % ALL_Evaluation[3,0]])
# table.add_row(['Fitting', ' ', ' ', '%.4f' % ALL_Evaluation[4,0]])
# print(table)

# # 输出表格 (验证集)
# print('C2P-Validation Set')
# table = pt.PrettyTable(['Index', 'UTS', 'EC', 'ALL'])
# table.add_row(['MAE', '%.4f' % UTS_Evaluation[0,1], '%.4f' % EC_Evaluation[0,1], '%.4f' % ALL_Evaluation[0,1]])
# table.add_row(['MSE', '%.4f' % UTS_Evaluation[1,1], '%.4f' % EC_Evaluation[1,1], '%.4f' % ALL_Evaluation[1,1]])
# table.add_row(['R2', '%.4f' % UTS_Evaluation[2,1], '%.4f' % EC_Evaluation[2,1], '%.4f' % ALL_Evaluation[2,1]])
# table.add_row(['R', ' ', ' ', '%.4f' % ALL_Evaluation[3,1]])
# table.add_row(['Fitting', ' ', ' ', '%.4f' % ALL_Evaluation[4,1]])
# print(table)

# 输出表格 (测试集)
print('C2P-Test Set')
table = pt.PrettyTable(['Index', 'UTS', 'EC', 'ALL'])
table.add_row(['MAE', '%.4f' % UTS_Evaluation[0, 2], '%.4f' % EC_Evaluation[0, 2], '%.4f' % ALL_Evaluation[0, 2]])
table.add_row(['MSE', '%.4f' % UTS_Evaluation[1, 2], '%.4f' % EC_Evaluation[1, 2], '%.4f' % ALL_Evaluation[1, 2]])
table.add_row(
    ['MAPE', '%.2f' % UTS_Evaluation[3, 2] + '%', '%.2f' % EC_Evaluation[3, 2] + '%',
     '%.2f' % ALL_Evaluation[5, 2] + '%'])
table.add_row(['R2', '%.4f' % UTS_Evaluation[2, 2], '%.4f' % EC_Evaluation[2, 2], '%.4f' % ALL_Evaluation[2, 2]])
table.add_row(['R', ' ', ' ', '%.4f' % ALL_Evaluation[3, 2]])
table.add_row(['Fitting', ' ', ' ', '%.4f' % ALL_Evaluation[4, 2]])
print(table)

# # 输出表格 (全集)
# print('C2P-All Set')
# table = pt.PrettyTable(['Index', 'UTS', 'EC', 'ALL'])
# table.add_row(['MAE', '%.4f' % UTS_Evaluation[0,3], '%.4f' % EC_Evaluation[0,3], '%.4f' % ALL_Evaluation[0,3]])
# table.add_row(['MSE', '%.4f' % UTS_Evaluation[1,3], '%.4f' % EC_Evaluation[1,3], '%.4f' % ALL_Evaluation[1,3]])
# table.add_row(['R2', '%.4f' % UTS_Evaluation[2,3], '%.4f' % EC_Evaluation[2,3], '%.4f' % ALL_Evaluation[2,3]])
# table.add_row(['R', ' ', ' ', '%.4f' % ALL_Evaluation[3,3]])
# table.add_row(['Fitting', ' ', ' ', '%.4f' % ALL_Evaluation[4,3]])
# print(table)
