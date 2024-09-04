import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 定义数据目录
train_dir = '../clean_data/train'
test_dir = '../clean_data/test'

# 读取K值标签文件
k_values = pd.read_excel('../clean_data/K_value.xls')

# 定义需要标准化的列
columns_to_scale = ['电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)', '温度(℃)',
                    '步次时间(s)', '电流线电压(mV)', '压差(mV)',
                    '接触阻抗(mΩ)', '线路阻抗(mΩ)']

# 初始化MinMaxScaler
scaler = MinMaxScaler()


def load_and_preprocess_data(directory, k_values, columns_to_scale, scaler):
    X, y = [], []
    for file in os.listdir(directory):
        if file.endswith('.xlsx'):
            df = pd.read_excel(os.path.join(directory, file))
            battery_id = file[:24]  # 假设文件名为BatteryID.xlsx

            # 只选取需要的列
            data = df[columns_to_scale]

            # 数据标准化
            scaled_data = scaler.fit_transform(data)

            # 检查K值是否存在
            k_value_row = k_values[k_values['bar_code'] == battery_id]
            if not k_value_row.empty:
                k_value = k_value_row['K_value(mV/d)'].values[0]
            else:
                print(f"警告: 找不到与电池ID {battery_id} 匹配的K值. 跳过此文件.")
                continue  # 跳过这个电池文件

            # 将数据按时间顺序累加，取平均值，作为模型的输入
            X.append(np.mean(scaled_data, axis=0))
            y.append(k_value)

    return np.array(X), np.array(y)


# 加载并预处理数据
X_train, y_train = load_and_preprocess_data(train_dir, k_values, columns_to_scale, scaler)
X_test, y_test = load_and_preprocess_data(test_dir, k_values, columns_to_scale, scaler)

# 构建随机森林模型
model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 计算训练集的损失函数（均方误差）
train_loss = mean_absolute_error(y_train, y_pred_train)

# 计算测试集的平均误差率和R^2值
mean_error_rate = np.mean(np.abs(y_test - y_pred_test) / y_test) * 100
r2 = r2_score(y_test, y_pred_test)

print(f'训练集损失函数: {train_loss}')
print(f'测试集平均误差率: {mean_error_rate}%')
print(f'测试集R^2值: {r2}')

