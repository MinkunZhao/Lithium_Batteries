import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 读取所有日志文件
log_files = glob.glob('../clean_data/log2/*.xls')
data_list = []

for file in log_files:
    df = pd.read_excel(file)
    data_list.append(df)

# 合并所有数据
data = pd.concat(data_list, ignore_index=True)

# 读取K值标签文件
k_values = pd.read_excel('../clean_data/K_value.xls')

# 数据标准化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)',
                                         '温度(℃)', '步次时间(s)', '电流线电压(mV)',
                                         '压差(mV)', '接触阻抗(mΩ)', '线路阻抗(mΩ)']])


# 转换为时间序列数据
def create_sequences(data, k_values, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = k_values[seq_length]  # K值
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


seq_length = 50  # 根据需要调整序列长度
sequences, labels = create_sequences(scaled_data, k_values['K_value(mV/d)'].values, seq_length)

# 转换为张量
sequences = torch.tensor(sequences, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# 创建DataLoader
dataset = TensorDataset(sequences, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


input_size = sequences.shape[2]
hidden_size = 128
num_layers = 2
output_size = 1

model = LSTMModel(input_size, hidden_size, num_layers, output_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
model.train()
for epoch in range(num_epochs):
    for sequences_batch, labels_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(sequences_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')


model.eval()
with torch.no_grad():
    predictions = model(sequences).cpu().numpy()

# 将预测值反归一化
predicted_k_values = scaler.inverse_transform(predictions)

plt.plot(k_values['K_value(mV/d)'].values, label='Actual K Values')
plt.plot(predicted_k_values, label='Predicted K Values')
plt.legend()
plt.show()

torch.save(model.state_dict(), 'lstm_model.pth')

