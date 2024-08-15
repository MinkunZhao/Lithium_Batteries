import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# 假设你有一个包含日志数据和K值标签的DataFrame
# df = pd.read_csv('log_data.csv')

# 生成示例数据（请用实际数据替换）
np.random.seed(42)
num_samples = 1000
data = {
    'voltage': np.random.normal(3700, 50, num_samples),
    'current': np.random.normal(1.5, 0.5, num_samples),
    'capacity': np.random.normal(1500, 100, num_samples),
    'energy': np.random.normal(5500, 200, num_samples),
    'step_number': np.random.randint(1, 10, num_samples),
    'step_name': np.random.choice(['Charge', 'Discharge', 'Rest'], num_samples),
    'temperature': np.random.normal(25, 5, num_samples),
    'step_time': np.random.randint(1, 1000, num_samples),
    'event_info': np.random.choice(['Event1', 'Event2', 'Event3'], num_samples),
    'current_wire_voltage': np.random.normal(10, 2, num_samples),
    'pressure_difference': np.random.normal(5, 1, num_samples),
    'contact_resistance': np.random.normal(1, 0.1, num_samples),
    'line_resistance': np.random.normal(0.5, 0.05, num_samples),
    'K_value': np.random.normal(0.1, 0.02, num_samples)  # K值标签
}

df = pd.DataFrame(data)


# 特征工程
def feature_engineering(df):
    df['voltage_diff'] = df['voltage'].diff().fillna(0)
    df['current_diff'] = df['current'].diff().fillna(0)
    df['capacity_diff'] = df['capacity'].diff().fillna(0)
    df['energy_diff'] = df['energy'].diff().fillna(0)
    df['temperature_diff'] = df['temperature'].diff().fillna(0)
    df['step_time_diff'] = df['step_time'].diff().fillna(0)

    df['voltage_diff_2'] = df['voltage'].diff(2).fillna(0)
    df['current_diff_2'] = df['current'].diff(2).fillna(0)
    df['capacity_diff_2'] = df['capacity'].diff(2).fillna(0)
    df['energy_diff_2'] = df['energy'].diff(2).fillna(0)
    df['temperature_diff_2'] = df['temperature'].diff(2).fillna(0)
    df['step_time_diff_2'] = df['step_time'].diff(2).fillna(0)

    return df


df = feature_engineering(df)

# 编码器（Encoding）
df_encoded = pd.get_dummies(df, columns=['step_name', 'event_info'])

# 标准化
scaler = StandardScaler()
features = df_encoded.drop(columns=['K_value'])
features_scaled = scaler.fit_transform(features)
labels = df_encoded['K_value']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# 模型训练
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 输出预测值和真实值对比
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())

# 保存模型
import joblib

joblib.dump(model, 'battery_k_value_model.pkl')

