import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.fft import fft


train_dir = '../clean_data/train'
test_dir = '../clean_data/test'
k_value_file = '../clean_data/K_value.xls'

k_values = pd.read_excel(k_value_file)

columns_to_scale = ['电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)', '温度(℃)', '步次时间(s)', '电流线电压(mV)',
                    '压差(mV)', '接触阻抗(mΩ)', '线路阻抗(mΩ)']

scaler = MinMaxScaler()

train_data, test_data, train_labels, test_labels = [], [], [], []


def process_log_files(log_dir, k_values, is_train=True):
    data = []
    labels = []

    for file in os.listdir(log_dir):
        battery_id = file[:24]
        log_file_path = os.path.join(log_dir, file)

        if battery_id in k_values['bar_code'].values:
            log_data = pd.read_excel(log_file_path)

            log_data = log_data[columns_to_scale]

            scaled_data = scaler.fit_transform(log_data)

            freq_data = np.abs(fft(scaled_data, axis=0))

            mean_freq_data = np.mean(freq_data, axis=0)

            k_value = k_values[k_values['bar_code'] == battery_id]['K_value(mV/d)'].values[0]

            data.append(mean_freq_data)
            labels.append(k_value)

    return data, labels


train_data, train_labels = process_log_files(train_dir, k_values, is_train=True)
test_data, test_labels = process_log_files(test_dir, k_values, is_train=False)

train_data = np.array(train_data)
test_data = np.array(test_data)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_data, train_labels)

test_predictions = model.predict(test_data)

mean_error_rate = np.mean(np.abs(test_labels - test_predictions) / test_labels) * 100
r2 = r2_score(test_labels, test_predictions)

print(f"平均误差率: {mean_error_rate:.2f}%")
print(f"R² 分数: {r2:.2f}")
