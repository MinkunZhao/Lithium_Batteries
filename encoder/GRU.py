import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import numpy as np


class BatteryDataset(Dataset):
    def __init__(self, file_paths, k_values, scaler):
        self.data = []
        self.labels = []

        for file_path, k_value in zip(file_paths, k_values):
            df = pd.read_excel(file_path)
            # 只保留指定列并标准化
            scaled_data = scaler.fit_transform(df[['电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)',
                                                   '温度(℃)', '步次时间(s)', '电流线电压(mV)',
                                                   '压差(mV)', '接触阻抗(mΩ)', '线路阻抗(mΩ)']])
            self.data.append(torch.tensor(scaled_data, dtype=torch.float32))
            self.labels.append(torch.tensor(k_value, dtype=torch.float32))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.gru.num_layers, x.size(0), self.gru.hidden_size).to(x.device)
        out, _ = self.gru(x, h_0)
        out = self.fc(out[:, -1, :])
        return out


def load_data(data_dir, k_values_file, scaler):
    k_values_df = pd.read_excel(k_values_file)
    k_values = k_values_df['K_value(mV/d)'].values

    file_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith('.xlsx')]

    dataset = BatteryDataset(file_paths, k_values, scaler)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader


def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def test_model(model, dataloader):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            predictions.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    mean_error_rate = np.mean(np.abs(actuals - predictions) / actuals) * 100
    r2 = r2_score(actuals, predictions)

    print(f'Mean Error Rate: {mean_error_rate:.2f}%')
    print(f'R^2 Score: {r2:.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = MinMaxScaler()

    train_loader = load_data('../clean_data/train', '../clean_data/K_value.xls', scaler)
    test_loader = load_data('../clean_data/test', '../clean_data/K_value.xls', scaler)

    input_size = 10  # 输入特征数量
    hidden_size = 64  # GRU隐藏层大小
    num_layers = 2  # GRU层数
    output_size = 1  # 输出为单一的K值
    model = GRUModel(input_size, hidden_size, num_layers, output_size).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    test_model(model, test_loader)


if __name__ == '__main__':
    main()
