import random
import pandas as pd
import numpy as np
import os

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
init = nn.init.xavier_uniform_

seed = 2024
def init_seed():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def search_bar_code(kf, code):
    for row in range(1, 11551):
        if kf.cell_value(row, 0) == code:
            return row
    return False


files = os.listdir("../clean_data/splited_log")

# num_samples = 11550
num_samples = 11550
log_steps = 5000
data = {
    'bar_code': [''] * num_samples,
    'voltage': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'current': [np.random.normal(1.5, 0.5, log_steps)] * num_samples,
    'capacity': [np.random.normal(1500, 100, log_steps)] * num_samples,
    'energy': [np.random.normal(5500, 200, log_steps)] * num_samples,
    'temperature': [np.random.normal(25, 5, log_steps)] * num_samples,
    'current_line_voltage': [np.random.normal(10, 2, log_steps)] * num_samples,
    'voltage_difference': [np.random.normal(5, 1, log_steps)] * num_samples,
    'contact_impedance': [np.random.normal(1, 0.1, log_steps)] * num_samples,
    'line_impedance': [np.random.normal(0.5, 0.05, log_steps)] * num_samples,
    'K_value': [0] * num_samples,  # K值标签
    'voltage_diff': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'current_diff': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'capacity_diff': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'energy_diff': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'temperature_diff': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'voltage_diff_2': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'current_diff_2': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'capacity_diff_2': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'energy_diff_2': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'temperature_diff_2': [np.random.normal(3700, 50, log_steps)] * num_samples,
    'step_name': [[''] * log_steps] * num_samples,
    'event_info': [[''] * log_steps] * num_samples
}
# print(data)

# k_value = xlrd.open_workbook("../clean_data/K_value.xls")
# table_k_value = k_value.sheets()[0]
# cnt = 0
# for f in tqdm(files, desc="data loading"):
#     log = openpyxl.load_workbook("../clean_data/splited_log/" + f)
#     table_log = log.active  # 获取第一个工作表
#
#     idx = search_bar_code(table_k_value, f[:24])
#     if idx is False:
#         continue
#     else:
#         data['bar_code'][cnt] = f[:24]
#         data['voltage'][cnt] = [table_log.cell(row=i, column=2).value for i in range(2, table_log.max_row + 1)]
#         data['current'][cnt] = [table_log.cell(row=i, column=3).value for i in range(2, table_log.max_row + 1)]
#         data['capacity'][cnt] = [table_log.cell(row=i, column=4).value for i in range(2, table_log.max_row + 1)]
#         data['energy'][cnt] = [table_log.cell(row=i, column=5).value for i in range(2, table_log.max_row + 1)]
#         data['temperature'][cnt] = [table_log.cell(row=i, column=8).value for i in range(2, table_log.max_row + 1)]
#
#         data['K_value'][cnt] = table_k_value.cell_value(idx, 6)
#         cnt += 1
# print('data num: {}'.format(cnt))
# df = pd.DataFrame(data)
# df.to_pickle('splited_log.pkl')
df = pd.read_pickle('../clean_data/splited_log.pkl')
# 特征工程
def feature_engineering(df):
    for i in range(num_samples):
        df.at[i, 'voltage_diff'] = pd.Series(df['voltage'][i]).diff().fillna(0).values
        df.at[i, 'current_diff'] = pd.Series(df['current'][i]).diff().fillna(0).values
        df.at[i, 'capacity_diff'] = pd.Series(df['capacity'][i]).diff().fillna(0).values
        df.at[i, 'energy_diff'] = pd.Series(df['energy'][i]).diff().fillna(0).values
        df.at[i, 'temperature_diff'] = pd.Series(df['temperature'][i]).diff().fillna(0).values

        df.at[i, 'voltage_diff_2'] = pd.Series(df['voltage'][i]).diff(2).fillna(0).values
        df.at[i, 'current_diff_2'] = pd.Series(df['current'][i]).diff(2).fillna(0).values
        df.at[i, 'capacity_diff_2'] = pd.Series(df['capacity'][i]).diff(2).fillna(0).values
        df.at[i, 'energy_diff_2'] = pd.Series(df['energy'][i]).diff(2).fillna(0).values
        df.at[i, 'temperature_diff_2'] = pd.Series(df['temperature'][i]).diff(2).fillna(0).values

    return df

df = feature_engineering(df)

# 选择用于回归的特征
features = ['voltage_diff', 'current_diff', 'capacity_diff', 'energy_diff', 'temperature_diff',
            'voltage_diff_2', 'current_diff_2', 'capacity_diff_2', 'energy_diff_2', 'temperature_diff_2'
            ]

# 标准化特征
# scaler = StandardScaler()
df[features] = df[features].map(lambda x: np.mean(x) if isinstance(x, np.ndarray) else x)
# df[features] = scaler.fit_transform(df[features].values)


class Battery(Dataset):
    def __init__(self, df):
        self.features = df[features].values
        self.labels = df['K_value'].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# 数据集拆分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = Battery(train_df)
test_dataset = Battery(test_df)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self._init_weight()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init(m.weight)
                nn.init.zeros_(m.bias)

input_dim = len(features)
model = RegressionModel(input_dim).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')


def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_error_rate = 0.0
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            running_loss += loss.item() * inputs.size(0)
            error_rate = torch.abs((targets.unsqueeze(1) - outputs) / targets.unsqueeze(1)) * 100
            running_error_rate += error_rate.mean().item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    epoch_loss = running_loss / len(test_loader.dataset)
    avg_error_rate = running_error_rate / len(test_loader.dataset)
    r2 = r2_score(all_targets, all_outputs)
    print(f'Test Loss: {epoch_loss:.4f}', f'Error_rate: {avg_error_rate:.4f}', f'R2: {r2:.4f}')


train_model(model, train_loader, criterion, optimizer, epochs=50)
evaluate_model(model, test_loader, criterion)

'''
# 编码器（Encoding）
# df_encoded = pd.get_dummies(df, columns=['step_name', 'event_info'])
# print(df_encoded)

num_features = 19

# 定义数据集类
class BatteryDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        num_data = row[num_features].values.astype(np.float32)
        step_name = row['step_name']
        event_info = row['event_info']
        k_value = row['K_value']
        return num_data, step_name, event_info, k_value


# 数据集拆分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = BatteryDataset(train_df)
test_dataset = BatteryDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 定义模型
class BatteryModel(nn.Module):
    def __init__(self, num_num_features, step_name_classes, event_info_classes, embedding_dim=10):
        super(BatteryModel, self).__init__()
        self.step_name_embedding = nn.Embedding(step_name_classes, embedding_dim)
        self.event_info_embedding = nn.Embedding(event_info_classes, embedding_dim)
        self.fc1 = nn.Linear(num_num_features + 2 * embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, num_data, step_name, event_info):
        step_name_emb = self.step_name_embedding(step_name)
        event_info_emb = self.event_info_embedding(event_info)
        x = torch.cat([num_data, step_name_emb, event_info_emb], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 模型参数
# num_num_features = len(num_features)
step_name_classes = df['step_name'].nunique()
event_info_classes = df['event_info'].nunique()
embedding_dim = 10

# 初始化模型
model = BatteryModel(num_features, step_name_classes, event_info_classes, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for num_data, step_name, event_info, k_value in train_loader:
            num_data = num_data.float()
            step_name = step_name.long()
            event_info = event_info.long()
            k_value = k_value.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(num_data, step_name, event_info)
            loss = criterion(outputs, k_value)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * num_data.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}')


# 评估模型
def evaluate_model(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for num_data, step_name, event_info, k_value in test_loader:
            num_data = num_data.float()
            step_name = step_name.long()
            event_info = event_info.long()
            k_value = k_value.float().unsqueeze(1)

            outputs = model(num_data, step_name, event_info)
            loss = criterion(outputs, k_value)
            running_loss += loss.item() * num_data.size(0)

    epoch_loss = running_loss / len(test_loader.dataset)
    print(f'Test Loss: {epoch_loss:.4f}')


# 训练和评估
train_model(model, train_loader, criterion, optimizer, epochs=50)
evaluate_model(model, test_loader, criterion)
'''
