import matplotlib.pyplot as plt
import numpy as np


labels = ['Verizon Saturation', 'T-Mobile Saturation']
y1 = [1/24*100,3/47*100]
y2 = [(100-1/24*100)/13,(100-3/47*100)/6]
y3 = [(100-1/24*100)/13*12,(100-3/47*100)/6*5]

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(labels, y1, width, label='Others')
ax.bar(labels, y2, width, bottom = y1,
       label='CPU')
ax.bar(labels, y3, width, bottom = np.array(y1) + np.array(y2),
       label='Transceiver')

ax.set_ylabel('Power Comsumption proportion(%)')
ax.set_title('Power contribution of different components')
ax.legend()

plt.show()
