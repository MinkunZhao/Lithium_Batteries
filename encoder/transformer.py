import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.utils.rnn as rnn_utils


class BatteryDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.label_df = pd.read_excel(label_file)
        self.files = os.listdir(data_dir)
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.files[idx])
        df = pd.read_excel(file_path)
        # Selecting relevant columns
        features = df[['电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)', '温度(℃)',
                       '步次时间(s)', '电流线电压(mV)', '压差(mV)',
                       '接触阻抗(mΩ)', '线路阻抗(mΩ)']].values
        scaled_features = self.scaler.fit_transform(features)
        # Get K-value from label file
        k_value = self.label_df[self.label_df['bar_code'] == self.files[idx]]['K_value(mV/d)'].values
        return torch.tensor(scaled_features, dtype=torch.float32), torch.tensor(k_value, dtype=torch.float32)


def collate_fn(batch):
    inputs = [item[0] for item in batch]
    labels = [item[1] for item in batch]  # labels 是列表，而非张量
    # 将 inputs 列表 pad 成相同长度的序列
    padded_inputs = rnn_utils.pad_sequence(inputs, batch_first=True)
    # 生成 mask，标记哪些是 padding，哪些是真实数据
    mask = torch.ones(padded_inputs.shape[:2], dtype=torch.bool)
    for i, input_tensor in enumerate(inputs):
        mask[i, :input_tensor.size(0)] = False
    # 将 labels 列表转换为一维的 tensor
    labels = torch.tensor(labels, dtype=torch.float32)  # labels 应该是 [batch_size] 的一维张量
    return padded_inputs, labels, mask.T  # 返回时确保 mask 的维度是 [seq_len, batch_size]


train_dataset = BatteryDataset(data_dir='../clean_data/train', label_file='../clean_data/K_value.xls')
test_dataset = BatteryDataset(data_dir='../clean_data/test', label_file='../clean_data/K_value.xls')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, mask=None):
        # x: [batch_size, seq_len, input_dim]
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.fc(x.mean(dim=1))  # Mean pooling across the sequence
        return x.squeeze(-1)  # Ensure the output is of shape [batch_size]


input_dim = 10
num_heads = 2
hidden_dim = 64
num_layers = 3
model = TransformerModel(input_dim=input_dim, num_heads=num_heads, hidden_dim=hidden_dim, num_layers=num_layers)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, mask in train_loader:
            optimizer.zero_grad()
            # 输出模型预测
            outputs = model(inputs, mask)  # 输出是 [batch_size] 的一维张量
            # 确保 outputs 和 labels 都是一维的，并且维度匹配
            assert outputs.shape == labels.shape, f"Output shape {outputs.shape} and Label shape {labels.shape} do not match"
            # 计算损失
            loss = criterion(outputs, labels)  # 直接使用张量计算 loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predictions = outputs.squeeze().cpu().numpy()
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions)

    mean_error_rate = (abs(np.array(all_labels) - np.array(all_predictions)) / np.array(all_labels) * 100).mean()

    r2 = r2_score(all_labels, all_predictions)

    print(f'Mean Error Rate: {mean_error_rate:.2f}%')
    print(f'R^2 Score: {r2:.4f}')
    return mean_error_rate, r2


train_model(model, train_loader, num_epochs=20)
mean_error_rate, r2 = evaluate_model(model, test_loader)
