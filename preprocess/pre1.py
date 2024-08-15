import os
import xlrd
import xlwt


def get_file_names(folder_path):
    try:
        files_and_folders = os.listdir(folder_path)
        file_names = [f[:24] for f in files_and_folders if os.path.isfile(os.path.join(folder_path, f))]
        return file_names
    except Exception as e:
        print(f"读取文件夹时出错: {e}")
        return []


folder_path = "../data_js"
file_names = get_file_names(folder_path)
# print(file_names)

data = xlrd.open_workbook("../K_value.xlsx")
table = data.sheets()[0]

k_value = xlwt.Workbook(encoding="utf-8")
sheet1 = k_value.add_sheet("Sheet1")
head = ['bar_code', 'ocv1', 'ocv1_resistance', 'time1', 'ocv2', 'time2', 'K_value(mV/d)']
for i in head:
    sheet1.write(0, head.index(i), i)

row_idx = 1
for i in range(1, 55570):
    if table.cell(i, 0).value in file_names:
        for j in range(6):
            sheet1.write(row_idx, j, table.cell(i, j).value)
        K = 1000 * ((float(table.cell(i, 1).value) - float(table.cell(i, 4).value)) /
                    (float(table.cell(i, 5).value) - float(table.cell(i, 3).value)))
        sheet1.write(row_idx, 6, K)
        row_idx += 1

k_value.save("../clean_data/K_value.xls")
