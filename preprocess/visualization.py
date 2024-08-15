import xlrd
import matplotlib.pyplot as plt

data = xlrd.open_workbook("../clean_data/K_value.xls")
table = data.sheets()[0]
k_values = table.col_values(6, start_rowx=1, end_rowx=11551)
# print(k_values)
plt.figure()
plt.hist(k_values, bins=150, density=True, alpha=0.6)
plt.xlabel('K_value')
plt.ylabel('Probability')
plt.title('PDF of K values')
plt.show()

data = xlrd.open_workbook("../clean_data/log/02KCBFN16050BCC730300008_L1_3950_1.xls")
table = data.sheets()[0]
# x = np.linspace(0, 4167, 4168)
t0 = table.cell(1, 0).value[-8:]
t = [0]
for i in range(2, 4169):
    ti = table.cell(i, 0).value[-8:]
    interval = (int(ti[:2]) * 3600 + int(ti[3:5]) * 60 + int(ti[6:]) -
                int(t0[:2]) * 3600 - int(t0[3:5]) * 60 - int(t0[6:]))
    t.append(interval)

voltage = table.col_values(1, start_rowx=1, end_rowx=4169)
plt.figure()
plt.plot(t, voltage)
plt.xlabel('time(s)')
plt.ylabel('Voltage(mV)')
plt.title('Trend of Voltage(mV)')
plt.show()

current = table.col_values(2, start_rowx=1, end_rowx=4169)
plt.figure()
plt.plot(t, current)
plt.xlabel('time(s)')
plt.ylabel('Current(mA)')
plt.title('Trend of Current(mA)')
plt.show()

capacity = table.col_values(3, start_rowx=1, end_rowx=4169)
plt.figure()
plt.plot(t, capacity)
plt.xlabel('time(s)')
plt.ylabel('Capacity(mAh)')
plt.title('Trend of Capacity(mAh)')
plt.show()

energy = table.col_values(4, start_rowx=1, end_rowx=4169)
plt.figure()
plt.plot(t, energy)
plt.xlabel('time(s)')
plt.ylabel('Energy(mWh)')
plt.title('Trend of Energy(mWh)')
plt.show()

temperature = table.col_values(7, start_rowx=1, end_rowx=4169)
plt.figure()
plt.plot(t, temperature)
plt.xlabel('time(s)')
plt.ylabel('Temperature(℃)')
plt.title('Trend of Temperature(℃)')
plt.show()

current_line_voltage = table.col_values(10, start_rowx=1, end_rowx=4169)
plt.figure()
plt.plot(t, current_line_voltage)
plt.xlabel('time(s)')
plt.ylabel('Current Line Voltage(mV)')
plt.title('Trend of Current Line Voltage(mV)')
plt.show()

voltage_difference = table.col_values(11, start_rowx=1, end_rowx=4169)
plt.figure(figsize=(16, 5))
plt.plot(t, voltage_difference)
plt.xlabel('time(s)')
plt.ylabel('Voltage Difference(mV)')
plt.title('Trend of Voltage Difference(mV)')
plt.show()

contact_impedance = table.col_values(12, start_rowx=1, end_rowx=4169)
plt.figure(figsize=(16, 5))
plt.plot(t, contact_impedance)
plt.xlabel('time(s)')
plt.ylabel('Contact Impedance(mΩ)')
plt.title('Trend of Contact Impedance(mΩ)')
plt.show()

line_impedance = table.col_values(13, start_rowx=1, end_rowx=4169)
plt.figure(figsize=(16, 5))
plt.plot(t, line_impedance)
plt.xlabel('time(s)')
plt.ylabel('Line Impedance(mΩ)')
plt.title('Trend of Line Impedance(mΩ)')
plt.show()

