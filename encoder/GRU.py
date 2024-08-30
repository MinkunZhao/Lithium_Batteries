import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
import torch.nn.utils.rnn as rnn_utils


# 加载数据
def load_data(log_files, k_value_file):
    log_data = []
    k_values = pd.read_excel(k_value_file)

    for file in log_files:
        data = pd.read_excel(file)
        log_data.append(data)

    return log_data, k_values


log_files = glob.glob("../clean_data/log2/*.xls")
k_value_file = "../clean_data/K_value.xls"

log_data, k_values = load_data(log_files, k_value_file)


# 数据标准化，仅针对指定的列
def preprocess_data(data):
    # 选择需要标准化的列
    columns_to_scale = ['电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)', '温度(℃)',
                        '步次时间(s)', '电流线电压(mV)', '压差(mV)', '接触阻抗(mΩ)', '线路阻抗(mΩ)']

    # 确保所有列都是数值类型
    data = data[columns_to_scale].apply(pd.to_numeric, errors='coerce')

    # 处理缺失值，可以选择填充或删除
    data = data.fillna(0)  # 这里用0填充缺失值，您可以选择其他策略

    # 数据标准化
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data


# 对所有日志文件进行预处理
scaled_log_data = [preprocess_data(data) for data in log_data]


class BatteryDataset(Dataset):
    def __init__(self, log_data, k_values):
        self.log_data = log_data
        self.k_values = k_values.apply(pd.to_numeric, errors='coerce').fillna(0)

    def __len__(self):
        return len(self.log_data)

    def __getitem__(self, idx):
        data = torch.tensor(self.log_data[idx], dtype=torch.float32)
        k_value = torch.tensor(self.k_values.iloc[idx], dtype=torch.float32)
        return data, k_value


def collate_fn(batch):
    data, labels = zip(*batch)
    # 使用 pad_sequence 将所有样本填充到相同长度
    data_padded = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return data_padded, labels


# 创建数据加载器，使用 collate_fn
dataset = BatteryDataset(scaled_log_data, k_values)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, device):
        super(GRUModel, self).__init__()
        self.device = device  # 添加 device 属性
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 定义 GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

        # 将模型移动到指定的设备上
        self.to(self.device)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 定义模型超参数
input_size = len(scaled_log_data[0][0])  # 输入特征数
hidden_size = 64  # 隐藏层神经元数
output_size = 1  # K值为单输出
num_layers = 2  # GRU层数

# 选择设备（CPU或GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型
model = GRUModel(input_size=10, hidden_size=50, output_size=1, num_layers=2, device=device)


# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
model.train()

for epoch in range(num_epochs):
    for i, (data, labels) in enumerate(dataloader):
        data = data.to(device)
        labels = labels.to(device)
        # 前向传播
        outputs = model(data)
        # 计算损失和反向传播
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    predictions = []
    actuals = []

    for data, labels in dataloader:
        data = data.to(model.device)
        labels = labels.to(model.device)
        outputs = model(data)
        predictions.append(outputs.cpu().numpy())
        actuals.append(labels.cpu().numpy())

# 转换成numpy数组
predictions = np.concatenate(predictions, axis=0)
actuals = np.concatenate(actuals, axis=0)

# 可以计算误差或进行可视化

mse = mean_squared_error(actuals, predictions)
rmse = np.sqrt(mse)
print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}')
