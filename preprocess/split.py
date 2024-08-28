import pandas as pd
import numpy as np
import os
import xlwt

# 定义文件夹路径
folder_path = '../clean_data/log'
output_folder = '../clean_data/splited_log'

# 确保输出文件夹存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有Excel文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        # 构造完整的文件路径
        file_path = os.path.join(folder_path, file_name)

        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 确保时间戳列是日期时间格式
        df['时间日期'] = pd.to_datetime(df['时间日期'])

        # 创建一个空的DataFrame用于存储结果
        result = pd.DataFrame()

        # 获取唯一的序号
        unique_serials = df['工步号'].unique()

        # 遍历每一个序号
        for serial in unique_serials:
            # 筛选出对应序号的数据，并按时间戳排序
            subset = df[df['工步号'] == serial]

            # 如果这一组数据不足5个，直接全选
            if len(subset) <= 5:
                selected = subset
            else:
                # 否则，使用linspace在索引范围内均匀选择5个点
                indices = np.linspace(0, len(subset) - 1, 5).astype(int)
                selected = subset.iloc[indices]

            # 将选中的结果追加到result中
            result = pd.concat([result, selected])

        # 将处理后的结果保存到输出文件夹中，文件名与原文件名相同，但扩展名为 .xlsx
        output_file_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.xlsx')
        result.to_excel(output_file_path, index=False, engine='openpyxl')
        print(f'已处理并保存：{output_file_path}')
