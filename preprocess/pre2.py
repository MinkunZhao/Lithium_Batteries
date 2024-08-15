import os
import xlrd
import xlwt

files = os.listdir("../data_js")

for f in files:
    data = xlrd.open_workbook("../data_js/" + f)
    table = data.sheets()[0]
    steps = table.col_values(5, start_rowx=0, end_rowx=45)
    flag_idx = []
    for i in range(2, len(steps)):
        if steps[i] != steps[i - 1]:
            flag_idx.append(i)
    attr = xlwt.Workbook(encoding="utf-8")
    sheet1 = attr.add_sheet("Sheet1")
    head = ['时间日期', '电压(mV)', '电流(mA)', '容量(mAh)', '能量(mWh)', '工步号', '工步名称', '温度(℃)',
            '步次时间(s)', '事件信息', '电流线电压(mV)', '压差(mV)', '接触阻抗(mΩ)', '线路阻抗(mΩ)']
    for i in head:
        sheet1.write(0, head.index(i), i)
    for i in range(1, table.nrows):
        for j in range(13):
            if j == 5:
                sheet1.write(i, j, max(1, table.cell(i, j).value - 2))
            elif j == 8:
                if i < flag_idx[0]:
                    sheet1.write(i, j, table.cell(i, j).value)
                elif flag_idx[0] <= i < flag_idx[1]:
                    t1 = table.cell(flag_idx[0] - 1, 0).value[-8:]
                    t2 = table.cell(flag_idx[0], 0).value[-8:]
                    interval = (int(t2[:2]) * 3600 + int(t2[3:5]) * 60 + int(t2[6:]) -
                                int(t1[:2]) * 3600 - int(t1[3:5]) * 60 - int(t1[6:]))
                    sheet1.write(i, j, table.cell(i, j).value + table.cell(flag_idx[0] - 1, j).value + interval)
                elif flag_idx[1] <= i < flag_idx[2]:
                    t1 = table.cell(flag_idx[0] - 1, 0).value[-8:]
                    t2 = table.cell(flag_idx[0], 0).value[-8:]
                    interval1 = (int(t2[:2]) * 3600 + int(t2[3:5]) * 60 + int(t2[6:]) -
                                 int(t1[:2]) * 3600 - int(t1[3:5]) * 60 - int(t1[6:]))
                    t3 = table.cell(flag_idx[1] - 1, 0).value[-8:]
                    t4 = table.cell(flag_idx[1], 0).value[-8:]
                    interval2 = (int(t4[:2]) * 3600 + int(t4[3:5]) * 60 + int(t4[6:]) -
                                 int(t3[:2]) * 3600 - int(t3[3:5]) * 60 - int(t3[6:]))
                    sheet1.write(i, j, table.cell(i, j).value + table.cell(flag_idx[0] - 1, j).value +
                                 table.cell(flag_idx[1] - 1, j).value + interval1 + interval2)
                else:
                    sheet1.write(i, j, table.cell(i, j).value)
            elif j == 9:
                if i+1 in flag_idx[:2]:
                    sheet1.write(i, j, "")
                else:
                    sheet1.write(i, j, table.cell(i, j).value)
            else:
                sheet1.write(i, j, table.cell(i, j).value)
        sheet1.write(i, 13, table.cell(i, 14).value)

    attr.save("../clean_data/log/" + f[:-4] + "xls")
