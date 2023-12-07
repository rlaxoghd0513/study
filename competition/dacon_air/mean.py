import numpy as np
import pandas as pd

from scipy.stats import mode

data = pd.read_csv('./_save/dacon_air/0505_1513.csv')
data_delay = data['Not_Delayed']
data_not_delay = data['Delayed']
mode_delay = mode(data_delay)
mode_not_delay = mode(data_not_delay)

print(f"최빈값: {mode_delay[0][0]}, 개수: {mode_delay[1][0]}")
print(f"최빈값: {mode_not_delay[0][0]}, 개수: {mode_not_delay[1][0]}")

# 최빈값: 0.321, 개수: 317622
# 최빈값: 0.679, 개수: 317622

data2 = pd.read_csv('./_save/dacon_air/0505_1542.csv')
data2_delay = data2['Not_Delayed']
data2_not_delay = data2['Delayed']
mode2_delay = mode(data2_delay)
mode2_not_delay = mode(data2_not_delay)

print(f"최빈값: {mode2_delay[0][0]}, 개수: {mode2_delay[1][0]}")
print(f"최빈값: {mode2_not_delay[0][0]}, 개수: {mode2_not_delay[1][0]}")

# 최빈값: 0.316, 개수: 147575
# 최빈값: 0.684, 개수: 147575.

data3 = pd.read_csv('./_save/dacon_air/0506_0426.csv')
data3_delay = data3['Not_Delayed']
data3_not_delay = data3['Delayed']
mode3_delay = mode(data3_delay)
mode3_not_delay = mode(data3_not_delay)

print(f"최빈값: {mode3_delay[0][0]}, 개수: {mode3_delay[1][0]}")
print(f"최빈값: {mode3_not_delay[0][0]}, 개수: {mode3_not_delay[1][0]}")

# 최빈값: 0.312, 개수: 490606
# 최빈값: 0.688, 개수: 490606


data4 = pd.read_csv('./_save/dacon_air/0505_1535.csv')
data4_delay = data4['Not_Delayed']
data4_not_delay = data4['Delayed']
mode4_delay = mode(data4_delay)
mode4_not_delay = mode(data4_not_delay)

print(f"최빈값: {mode4_delay[0][0]}, 개수: {mode4_delay[1][0]}")
print(f"최빈값: {mode4_not_delay[0][0]}, 개수: {mode4_not_delay[1][0]}")

# 최빈값: 0.32, 개수: 254613
# 최빈값: 0.68, 개수: 254613

data5 = pd.read_csv('./_save/dacon_air/0506_1201.csv')
data5_delay = data5['Not_Delayed']
data5_not_delay = data5['Delayed']
mode5_delay = mode(data5_delay)
mode5_not_delay = mode(data5_not_delay)

print(f"최빈값: {mode5_delay[0][0]}, 개수: {mode5_delay[1][0]}")
print(f"최빈값: {mode5_not_delay[0][0]}, 개수: {mode5_not_delay[1][0]}")

# 최빈값: 0.334, 개수: 249249
# 최빈값: 0.666, 개수: 249249

data6 = pd.read_csv('./_save/dacon_air/0506_1510.csv')
data6_delay = data6['Not_Delayed']
data6_not_delay = data6['Delayed']
mode6_delay = mode(data6_delay)
mode6_not_delay = mode(data6_not_delay)

print(f"최빈값: {mode6_delay[0][0]}, 개수: {mode6_delay[1][0]}")
print(f"최빈값: {mode6_not_delay[0][0]}, 개수: {mode6_not_delay[1][0]}")

# 최빈값: 0.319, 개수: 287538
# 최빈값: 0.681, 개수: 287538

data7 = pd.read_csv('./_save/dacon_air/0506_1528.csv')
data7_delay = data7['Not_Delayed']
data7_not_delay = data7['Delayed']
mode7_delay = mode(data7_delay)
mode7_not_delay = mode(data7_not_delay)

print(f"최빈값: {mode7_delay[0][0]}, 개수: {mode7_delay[1][0]}")
print(f"최빈값: {mode7_not_delay[0][0]}, 개수: {mode7_not_delay[1][0]}")

# 최빈값: 0.32, 개수: 1000000
# 최빈값: 0.68, 개수: 1000000

