# -*- coding: utf-8 -*-

# MLDS模型
import warnings
import numpy as np
import pandas as pd
import time
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
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

# 记录开始时间
time_start = time.perf_counter()

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


# 对数变换
def Logarithmic(data):
    data[:, 5] = np.log(data[:, 5] + 1)
    return data


# 反对数变换
def Anti_Logarithmic(data):
    data[:, 5] = np.exp(data[:, 5]) - 1
    return data


# 非负化
def Non_Negative(data):
    for i in range(0, data.shape[0]):
        if data.ndim == 1:
            if data[i] < 0.001:
                data[i] = 0
        else:
            for j in range(0, data.shape[1]):
                if data[i][j] < 0.001:
                    data[i][j] = 0
    return data


# 限制测试集范围
def Test_Range(data_x, data_y, bound):
    test_shape = 0
    for k in range(0, data_x.shape[0]):
        flag = False
        for g in range(0, data_x.shape[1]):
            if data_x[k, g] > bound[1] or data_x[k, g] < bound[0]:
                flag = True
        if flag == False:
            if test_shape == 0:
                new_x = data_x[k, :]
                new_y = data_y[k, :]
            else:
                new_x = np.vstack((new_x, data_x[k, :]))
                new_y = np.vstack((new_y, data_y[k, :]))
            test_shape = test_shape + 1
    print('测试集大小:', data_x.shape[0], '限制范围后测试集大小:', test_shape)
    return new_x, new_y


# C2P模型训练
def c2p_train(train_x, train_y):
    for i in range(0, 50):
        # 划分90%训练集，10%验证集
        x_train, x_validation, y_train, y_validation = train_test_split(train_x, train_y, test_size=0.1)
        # 建立UTS模型
        # 随机森林
        uts_model = RandomForestRegressor(n_estimators=100, max_features=2, max_depth=15)
        uts_model_train = uts_model.fit(train_x, train_y[:, 0])
        uts_pre_validation = uts_model.predict(x_validation)
        # 建立EC模型
        # 梯度提升树
        ec_model = GradientBoostingRegressor(n_estimators=100, max_features=4, max_depth=6)
        ec_model_train = ec_model.fit(train_x, train_y[:, 1])
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
        if R2 > 0.98:
            break
    return uts_model, ec_model


# C2P模型预测
def c2p_predict(test_x, uts_model, ec_model):
    uts_pre = uts_model.predict(test_x)
    ec_pre = ec_model.predict(test_x)
    # 组合变量
    y_pre = np.vstack((uts_pre, ec_pre)).T
    return y_pre


# P2C模型
def p2c(train_x, train_y, test_y):
    # 建立Fe模型
    # 梯度提升树
    fe_model = GradientBoostingRegressor(n_estimators=50, max_depth=7)
    fe_model_train = fe_model.fit(train_y, train_x[:, 0])
    fe_pre = fe_model.predict(test_y)
    # 建立P模型
    # 随机森林
    p_model = RandomForestRegressor(max_depth=9)
    p_model_train = p_model.fit(train_y, train_x[:, 1])
    p_pre = p_model.predict(test_y)
    # 建立Ni模型
    # 支持向量机
    ni_model = SVR(kernel='rbf', gamma=7.5)
    ni_model_train = ni_model.fit(train_y, train_x[:, 2])
    ni_pre = ni_model.predict(test_y)
    # 建立Si模型
    # XGBoost
    si_model = XGBRegressor(n_estimators=50, max_depth=9)
    si_model_train = si_model.fit(train_y, train_x[:, 3])
    si_pre = si_model.predict(test_y)
    # 建立Mg模型
    # GBDT
    mg_model = GradientBoostingRegressor(n_estimators=30, max_depth=3)
    mg_model_train = mg_model.fit(train_y, train_x[:, 4])
    mg_pre = mg_model.predict(test_y)
    # 建立Zn模型
    # BP神经网络
    zn_model = MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=17, max_iter=600)
    zn_model_train = zn_model.fit(train_y, train_x[:, 5])
    zn_pre = zn_model.predict(test_y)
    # 建立Sn模型
    # 随机森林
    sn_model = RandomForestRegressor(max_depth=6)
    sn_model_train = sn_model.fit(train_y, train_x[:, 6])
    sn_pre = sn_model.predict(test_y)
    # 建立Cr模型
    # BP神经网络
    cr_model = MLPRegressor(activation='relu', solver='lbfgs', hidden_layer_sizes=19, max_iter=1600)
    cr_model_train = cr_model.fit(train_y, train_x[:, 7])
    cr_pre = cr_model.predict(test_y)
    # 建立Zr模型
    # 随机森林
    zr_model = RandomForestRegressor(max_depth=1)
    zr_model_train = zr_model.fit(train_y, train_x[:, 8])
    zr_pre = zr_model.predict(test_y)
    # 建立RE模型
    # 随机森林
    re_model = RandomForestRegressor(max_depth=6)
    re_model_train = re_model.fit(train_y, train_x[:, 9])
    re_pre = re_model.predict(test_y)
    # 建立Al模型
    # 梯度提升树
    al_model = GradientBoostingRegressor(n_estimators=50, max_depth=3)
    al_model_train = al_model.fit(train_y, train_x[:, 10])
    al_pre = al_model.predict(test_y)
    # 建立Pb模型
    # XGBoost
    pb_model = XGBRegressor(n_estimators=50, max_depth=5)
    pb_model_train = pb_model.fit(train_y, train_x[:, 11])
    pb_pre = pb_model.predict(test_y)
    # 建立Mn模型
    # 随机森林
    mn_model = RandomForestRegressor(max_depth=1)
    mn_model_train = mn_model.fit(train_y, train_x[:, 12])
    mn_pre = mn_model.predict(test_y)
    # 建立Sb模型
    # 梯度提升树
    sb_model = GradientBoostingRegressor(n_estimators=200, max_depth=3)
    sb_model_train = sb_model.fit(train_y, train_x[:, 13])
    sb_pre = sb_model.predict(test_y)
    # 建立S模型
    # 支持向量机
    s_model = SVR(kernel='rbf', gamma=1.5)
    # 训练模型并代入验证集
    s_model_train = s_model.fit(train_y, train_x[:, 14])
    s_pre = s_model.predict(test_y)
    # 组合变量
    x_pre = np.vstack((fe_pre, p_pre, ni_pre, si_pre, mg_pre, zn_pre, sn_pre, cr_pre, zr_pre, re_pre, al_pre, pb_pre,
                       mn_pre, sb_pre, s_pre)).T
    return x_pre


# PSO
class PSO:
    def __init__(self, parameters):
        # 初始化
        self.time = parameters[0]  # 最大迭代次数
        self.pop_size = parameters[1]  # 种群大小
        self.var_num = len(parameters[2])  # 变量个数
        self.bound = []  # 变量的约束范围
        self.bound.append(parameters[2])
        self.bound.append(parameters[3])
        self.test_num = parameters[4]  # 测试集的数据标记
        self.init_var = parameters[5]
        self.uts_model = parameters[6]
        self.ec_model = parameters[7]
        self.y_test = parameters[8]
        self.y_pre = parameters[9]

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有粒子的位置
        self.pop_v = np.zeros((self.pop_size, self.var_num))  # 所有粒子的速度
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = -1000000
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = self.init_var[j]
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    # 个体适应值计算
    def fitness(self, ind_var):
        self.y_pre = c2p_predict([ind_var], self.uts_model, self.ec_model)
        # 预测性能反归一化
        self.y_pre = Anti_Normalization(self.y_pre, miny, maxy)
        err = np.max(abs(self.y_test - self.y_pre) / self.y_test)
        return err

    # 更新
    def update_operator(self, pop_size):
        c1 = 2  # 学习因子，一般为2
        c2 = 2
        w = 0.4  # 自身权重因子
        for i in range(pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(0, 1) * (self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.bound[0][j]:
                    self.pop_x[i][j] = self.bound[0][j]
                if self.pop_x[i][j] > self.bound[1][j]:
                    self.pop_x[i][j] = self.bound[1][j]
            # 更新p_best和g_best
            if self.fitness(self.pop_x[i]) < self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if self.fitness(self.pop_x[i]) < self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    def main(self):
        popobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        for gen in range(self.time):
            self.update_operator(self.pop_size)
            popobj.append(self.fitness(self.g_best))
            if self.fitness(self.g_best) < self.fitness(self.ng_best):
                self.ng_best = self.g_best
            print('- 第', gen + 1, '代 -')
            print(self.fitness(self.ng_best))
            if self.fitness(self.ng_best) < 0.05:
                break


# MLDS模型
def mlds(estimator, err0, train_x, train_y, test_y):
    # 成分预测值，性能预测值
    pre_x = np.zeros((test_y.shape[0], 15))
    pre_y = np.zeros((test_y.shape))
    # 误差
    err = []
    for test_index in range(0, test_y.shape[0]):
        print('- 第', test_index + 1, '条数据 -')
        print('------------------------------- MLDS -------------------------------')
        flag = False
        for i in range(0, estimator):
            print('- 第', i + 1, '次训练 -')
            # 输入所需材料性能test_y,预测得到材料成分pre_x
            pre_x[test_index, :] = p2c(train_x, train_y, [test_y[test_index, :]])
            pre_x[test_index, :] = Non_Negative(pre_x[test_index, :])
            # 预测得到的成分代入C2P得到预测性能
            pre_y[test_index, :] = c2p_predict([pre_x[test_index, :]], uts_model, ec_model)
            # 性能反归一化
            test_y[test_index, :] = Anti_Normalization(test_y[test_index, :], miny, maxy)
            pre_y[test_index, :] = Anti_Normalization(pre_y[test_index, :], miny, maxy)
            print(test_y[test_index, :], pre_y[test_index, :])
            # 计算预测性能和目标性能的误差
            err1 = np.max(abs(test_y[test_index, :] - pre_y[test_index, :]) / test_y[test_index, :])
            print(err1)
            # 误差与预设比较大小
            if i == 0:
                err_sign = err1
                x_sign = np.array(pre_x[test_index, :])
                y_sign = np.array(pre_y[test_index, :])
            if err1 < err0:
                err_sign = err1
                x_sign = np.array(pre_x[test_index, :])
                y_sign = np.array(pre_y[test_index, :])
                flag = True
                break
            else:
                if err1 < err_sign:
                    err_sign = err1
                    x_sign = np.array(pre_x[test_index, :])
                    y_sign = np.array(pre_y[test_index, :])
            test_y[test_index, :] = (test_y[test_index, :] - miny) / (maxy - miny)
        err.append(err_sign)
        if flag == False:
            test_y[test_index, :] = Anti_Normalization(test_y[test_index, :], miny, maxy)
            pre_x[test_index, :] = x_sign
            pre_y[test_index, :] = y_sign
            # PSO
            print('------------------------------- PSO -------------------------------')
            print('初始值')
            print(err[test_index], test_y[test_index, :], pre_y[test_index, :])

            times = 20
            popsize = 20
            low = 0.8 * pre_x[test_index, :]
            up = 1.2 * pre_x[test_index, :]

            parameters = [times, popsize, low, up, test_index, pre_x[test_index, :], uts_model, ec_model,
                          test_y[test_index, :], pre_y[test_index, :]]
            pso = PSO(parameters)
            pso.main()
            pre_x[test_index, :] = pso.ng_best
            pre_y[test_index, :] = pso.y_pre
            err[test_index] = pso.fitness(pso.ng_best)

    # 绘制误差曲线
    font = {'family': 'Times New Roman', 'size': 12}
    sns.set(style='ticks')
    plt.scatter(np.arange(1, test_index + 2), err)
    plt.plot(np.arange(1, test_index + 2), err0 * np.ones(test_index + 1), c='orange', linestyle='--')  # 绘制虚线
    plt.xlabel('data n', fontproperties=font)
    plt.ylabel('error', fontproperties=font)
    plt.title('error')
    plt.show()

    # 预测值的反对数变换、反归一化、非负化（根据实际情况）
    pre_x = Anti_Logarithmic(pre_x)
    pre_x = Anti_Normalization(pre_x, minx, maxx)
    pre_x = Non_Negative(pre_x)
    return pre_x, pre_y, err


# minmax归一化处理
x, minx, maxx = Normalization(x)
y, miny, maxy = Normalization(y)
x = Logarithmic(x)

# 划分90%训练集，10%测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# 测试集反对数化，反归一化
x_test = Anti_Logarithmic(x_test)
x_test = Anti_Normalization(x_test, minx, maxx)
#
# 限制测试集范围
test_bound = [0, 10]
x_test, y_test = Test_Range(x_test, y_test, test_bound)

# C2P模型的训练,得到预测效果比较好的C2P模型
uts_model, ec_model = c2p_train(x_train, y_train)

# MLDS预测
estimator = 100
err0 = 0.05
x_pre, y_pre, err = mlds(estimator, err0, x_train, y_train, y_test)

# 计算每个元素的MAE,MSE,R2
Evaluation = np.zeros((15, 3))
# Fe
Evaluation[0, 0] = mean_absolute_error(x_test[:, 0], x_pre[:, 0])
Evaluation[0, 1] = mean_squared_error(x_test[:, 0], x_pre[:, 0])
Evaluation[0, 2] = r2_score(x_test[:, 0], x_pre[:, 0])
# P
Evaluation[1, 0] = mean_absolute_error(x_test[:, 1], x_pre[:, 1])
Evaluation[1, 1] = mean_squared_error(x_test[:, 1], x_pre[:, 1])
Evaluation[1, 2] = r2_score(x_test[:, 1], x_pre[:, 1])
# Ni
Evaluation[2, 0] = mean_absolute_error(x_test[:, 2], x_pre[:, 2])
Evaluation[2, 1] = mean_squared_error(x_test[:, 2], x_pre[:, 2])
Evaluation[2, 2] = r2_score(x_test[:, 2], x_pre[:, 2])
# Si
Evaluation[3, 0] = mean_absolute_error(x_test[:, 3], x_pre[:, 3])
Evaluation[3, 1] = mean_squared_error(x_test[:, 3], x_pre[:, 3])
Evaluation[3, 2] = r2_score(x_test[:, 3], x_pre[:, 3])
# Mg
Evaluation[4, 0] = mean_absolute_error(x_test[:, 4], x_pre[:, 4])
Evaluation[4, 1] = mean_squared_error(x_test[:, 4], x_pre[:, 4])
Evaluation[4, 2] = r2_score(x_test[:, 4], x_pre[:, 4])
# Zn
Evaluation[5, 0] = mean_absolute_error(x_test[:, 5], x_pre[:, 5])
Evaluation[5, 1] = mean_squared_error(x_test[:, 5], x_pre[:, 5])
Evaluation[5, 2] = r2_score(x_test[:, 5], x_pre[:, 5])
# Sn
Evaluation[6, 0] = mean_absolute_error(x_test[:, 6], x_pre[:, 6])
Evaluation[6, 1] = mean_squared_error(x_test[:, 6], x_pre[:, 6])
Evaluation[6, 2] = r2_score(x_test[:, 6], x_pre[:, 6])
# Cr
Evaluation[7, 0] = mean_absolute_error(x_test[:, 7], x_pre[:, 7])
Evaluation[7, 1] = mean_squared_error(x_test[:, 7], x_pre[:, 7])
Evaluation[7, 2] = r2_score(x_test[:, 7], x_pre[:, 7])
# Zr
Evaluation[8, 0] = mean_absolute_error(x_test[:, 8], x_pre[:, 8])
Evaluation[8, 1] = mean_squared_error(x_test[:, 8], x_pre[:, 8])
Evaluation[8, 2] = r2_score(x_test[:, 8], x_pre[:, 8])
# RE
Evaluation[9, 0] = mean_absolute_error(x_test[:, 9], x_pre[:, 9])
Evaluation[9, 1] = mean_squared_error(x_test[:, 9], x_pre[:, 9])
Evaluation[9, 2] = r2_score(x_test[:, 9], x_pre[:, 9])
# Al
Evaluation[10, 0] = mean_absolute_error(x_test[:, 10], x_pre[:, 10])
Evaluation[10, 1] = mean_squared_error(x_test[:, 10], x_pre[:, 10])
Evaluation[10, 2] = r2_score(x_test[:, 10], x_pre[:, 10])
# Pb
Evaluation[11, 0] = mean_absolute_error(x_test[:, 11], x_pre[:, 11])
Evaluation[11, 1] = mean_squared_error(x_test[:, 11], x_pre[:, 11])
Evaluation[11, 2] = r2_score(x_test[:, 11], x_pre[:, 11])
# Mn
Evaluation[12, 0] = mean_absolute_error(x_test[:, 12], x_pre[:, 12])
Evaluation[12, 1] = mean_squared_error(x_test[:, 12], x_pre[:, 12])
Evaluation[12, 2] = r2_score(x_test[:, 12], x_pre[:, 12])
# Sb
Evaluation[13, 0] = mean_absolute_error(x_test[:, 13], x_pre[:, 13])
Evaluation[13, 1] = mean_squared_error(x_test[:, 13], x_pre[:, 13])
Evaluation[13, 2] = r2_score(x_test[:, 13], x_pre[:, 13])
# S
Evaluation[14, 0] = mean_absolute_error(x_test[:, 14], x_pre[:, 14])
Evaluation[14, 1] = mean_squared_error(x_test[:, 14], x_pre[:, 14])
Evaluation[14, 2] = r2_score(x_test[:, 14], x_pre[:, 14])


# 线性拟合
def f_1(x, A, B):
    return A * x + B


# component拟合点
x_component = x_test[:, 0]
y_component = x_pre[:, 0]
for j in range(0, x_shape):
    x_component = np.append(x_component, x_test[:, j])
    y_component = np.append(y_component, x_pre[:, j])

# 计算成分设计误差
mse1 = mean_squared_error(x_component, y_component)
mae1 = mean_absolute_error(x_component, y_component)
r2_1 = r2_score(x_component, y_component)
r_1 = np.corrcoef(x_component, y_component)[0, 1]

# 直线拟合与绘制
A1, B1 = optimize.curve_fit(f_1, x_component, y_component)[0]
x1 = np.arange(-1, 20, 0.1)
y1 = A1 * x1 + B1

# property拟合点
x_property = y_test[:, 0]
y_property = y_pre[:, 0]
for j in range(0, y_shape):
    x_property = np.append(x_property, y_test[:, j])
    y_property = np.append(y_property, y_pre[:, j])

# 计算成分设计误差
mse2 = mean_squared_error(x_property, y_property)
mae2 = mean_absolute_error(x_property, y_property)
r2_2 = r2_score(x_property, y_property)
r_2 = np.corrcoef(x_property, y_property)[0, 1]

# 直线拟合与绘制
A2, B2 = optimize.curve_fit(f_1, x_property, y_property)[0]
x2 = np.arange(0, 1000)
y2 = A2 * x2 + B2

# 绘图
# 设置参考的1：1虚线参数
xxx = [-100, 1000]
yyy = [-100, 1000]
font = {'family': 'Times New Roman', 'size': 12}
sns.set(style='ticks')
# 绘制散点图 (x)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(x_test, x_pre, s=30, c='red', edgecolors='k', marker='o', alpha=0.8)
plt.plot(x1, y1, "blue")  # 绘制拟合直线
plt.title('MLDS ScatterPlot(Component) R2: %.4f' % r2_1, fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('True', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.xticks(font=font)
plt.xlim((-1, 15))  # 设置坐标轴范围
plt.ylim((-1, 15))
plt.show()
# 绘制散点图 (y)
plt.plot(xxx, yyy, c='0', linewidth=1, linestyle=':', marker='.', alpha=0.3)  # 绘制虚线
plt.scatter(y_test[:, 0], y_pre[:, 0], s=30, c='red', edgecolors='k', marker='o', alpha=0.8,
            label='UTS')  # 绘制UTS散点图，横轴是真实值，竖轴是预测值
plt.scatter(y_test[:, 1], y_pre[:, 1], s=30, c='green', edgecolors='k', marker='o', alpha=0.8,
            label='EC')  # 绘制EC散点图，横轴是真实值，竖轴是预测值
plt.plot(x2, y2, "blue")  # 绘制拟合直线
plt.title('MLDS ScatterPlot(Property) R2: %.4f' % r2_2, fontproperties=font)
plt.xticks(font=font)
plt.yticks(font=font)
plt.xlabel('True', fontproperties=font)
plt.ylabel('Prediction', fontproperties=font)
plt.legend(loc='best', prop=font)
plt.xticks(font=font)
plt.xlim((0, 1000))  # 设置坐标轴范围
plt.ylim((0, 1000))
plt.show()

# # 输出真实值/预测值对比表格
# print('成分真实值：',x_test,'\n成分预测值：',x_pre)
# print('目标性能：',y_test,'\n预测性能：',y_pre)

# 输出表格
print('MLDS')
table = pt.PrettyTable(['Index', 'MAE', 'MSE', 'R2', 'R', 'Fitting'])
table.add_row(['All', '%.4f' % mae1, '%.4f' % mse1, '%.4f' % r2_1, '%.4f' % r_1, '%.4f' % A1])
table.add_row(['Fe', '%.4f' % Evaluation[0, 0], '%.4f' % Evaluation[0, 1], '%.4f' % Evaluation[0, 2], ' ', ' '])
table.add_row(['P', '%.4f' % Evaluation[1, 0], '%.4f' % Evaluation[1, 1], '%.4f' % Evaluation[1, 2], ' ', ' '])
table.add_row(['Ni', '%.4f' % Evaluation[2, 0], '%.4f' % Evaluation[2, 1], '%.4f' % Evaluation[2, 2], ' ', ' '])
table.add_row(['Si', '%.4f' % Evaluation[3, 0], '%.4f' % Evaluation[3, 1], '%.4f' % Evaluation[3, 2], ' ', ' '])
table.add_row(['Mg', '%.4f' % Evaluation[4, 0], '%.4f' % Evaluation[4, 1], '%.4f' % Evaluation[4, 2], ' ', ' '])
table.add_row(['Zn', '%.4f' % Evaluation[5, 0], '%.4f' % Evaluation[5, 1], '%.4f' % Evaluation[5, 2], ' ', ' '])
table.add_row(['Sn', '%.4f' % Evaluation[6, 0], '%.4f' % Evaluation[6, 1], '%.4f' % Evaluation[6, 2], ' ', ' '])
table.add_row(['Cr', '%.4f' % Evaluation[7, 0], '%.4f' % Evaluation[7, 1], '%.4f' % Evaluation[7, 2], ' ', ' '])
table.add_row(['Zr', '%.4f' % Evaluation[8, 0], '%.4f' % Evaluation[8, 1], '%.4f' % Evaluation[8, 2], ' ', ' '])
table.add_row(['RE', '%.4f' % Evaluation[9, 0], '%.4f' % Evaluation[9, 1], '%.4f' % Evaluation[9, 2], ' ', ' '])
table.add_row(['Al', '%.4f' % Evaluation[10, 0], '%.4f' % Evaluation[10, 1], '%.4f' % Evaluation[10, 2], ' ', ' '])
table.add_row(['Pb', '%.4f' % Evaluation[11, 0], '%.4f' % Evaluation[11, 1], '%.4f' % Evaluation[11, 2], ' ', ' '])
table.add_row(['Mn', '%.4f' % Evaluation[12, 0], '%.4f' % Evaluation[12, 1], '%.4f' % Evaluation[12, 2], ' ', ' '])
table.add_row(['Sb', '%.4f' % Evaluation[13, 0], '%.4f' % Evaluation[13, 1], '%.4f' % Evaluation[13, 2], ' ', ' '])
table.add_row(['S', '%.4f' % Evaluation[14, 0], '%.4f' % Evaluation[14, 1], '%.4f' % Evaluation[14, 2], ' ', ' '])
print(table)

# 记录结束时间
time_end = time.perf_counter()
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
time_hour = time_sum // 3600
time_minute = (time_sum % 3600) // 60
time_second = (time_sum % 3600) % 60
time_string = 'MLDS计算时间：'
if time_hour != 0:
    time_string = time_string + '%d' % time_hour + ' 时 '
if time_minute != 0:
    time_string = time_string + '%d' % time_minute + ' 分 '
time_string = time_string + '%d' % time_second + ' 秒 '
print(time_string)
