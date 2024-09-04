import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


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


def collate_fn(batch):
    # 获取数据和标签
    data, labels = zip(*batch)

    # 获取每个序列的长度
    lengths = [len(seq) for seq in data]

    # 填充序列到相同的长度
    padded_data = pad_sequence(data, batch_first=True, padding_value=0)

    # 转换标签为 tensor
    labels = torch.tensor(labels, dtype=torch.float32)

    return padded_data, labels, lengths


def load_data(data_dir, k_values_file, scaler):
    # 加载 k_values_file 文件
    k_values_df = pd.read_excel(k_values_file)

    # 提取 bar_code 和 K_value(mV/d) 列
    bar_codes = k_values_df['bar_code'].values
    k_values = k_values_df['K_value(mV/d)'].values

    # 创建一个字典，将 bar_code 映射到对应的 k_value
    k_values_dict = dict(zip(bar_codes, k_values))

    # 遍历 data_dir 中的文件，检查是否有对应的 bar_code
    valid_file_paths = []
    valid_k_values = []

    for fname in tqdm(os.listdir(data_dir), desc='Processing files'):
        if fname.endswith('.xlsx'):
            bar_code = fname.split('_')[0]
            if bar_code in k_values_dict:
                valid_file_paths.append(os.path.join(data_dir, fname))
                valid_k_values.append(k_values_dict[bar_code])

    # 创建数据集和数据加载器，使用 collate_fn 处理可变长度序列
    dataset = BatteryDataset(valid_file_paths, valid_k_values, scaler)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, collate_fn=collate_fn)
    return dataloader


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        lengths = torch.tensor(lengths, dtype=torch.int64)
        # 打包序列，去除填充部分
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # 通过 LSTM 层
        packed_output, _ = self.lstm(packed_input, (h0, c0))

        # 解包序列
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)

        # 取最后一个有效时间步的输出
        out = self.fc(lstm_out[range(len(lstm_out)), lengths - 1])

        out = out.squeeze(-1)
        return out


def train_model(model, dataloader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        for data, labels, lengths in tqdm(dataloader, desc='training'):
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data, lengths)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


def test_model(model, dataloader):
    model.eval()
    preds = []
    grounds = []
    with torch.no_grad():
        for data, labels, lengths in tqdm(dataloader, desc='testing'):
            data, labels = data.cuda(), labels.cuda()
            outputs = model(data, lengths)
            preds.append(outputs.cpu().numpy())
            grounds.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    grounds = np.concatenate(grounds)
    mean_error_rate = np.mean(np.abs(grounds - preds) / grounds) * 100
    r2 = r2_score(grounds, preds)
    print(f'Mean Error Rate: {mean_error_rate:.2f}%', f'R^2 Score: {r2:.4f}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = MinMaxScaler()
    train_loader = load_data('../clean_data/train', '../clean_data/K_value.xls', scaler)
    test_loader = load_data('../clean_data/test', '../clean_data/K_value.xls', scaler)

    input_size = 10
    hidden_size = 64
    num_layers = 2
    output_size = 1
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, num_epochs=40)
    test_model(model, test_loader)


if __name__ == '__main__':
    main()
