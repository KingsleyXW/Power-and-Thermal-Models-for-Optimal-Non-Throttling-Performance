import matplotlib.pyplot as plt
import numpy as np


labels = ['Verizon Saturation', 'T-Mobile Saturation']
y1 = [3700, 5700, 8800]
y2 = [1250, 5600, 6300]
x = np.arange(3)
plt.figure(0)
plt.bar(x, height=y1)
plt.xticks(x, ['Stress CPU','Stress Transceiver','Stress Both']);
plt.plot([-10,1,10], [y1[0],y1[0],y1[0]] ,color="y",linestyle='--', linewidth=1.5)
plt.plot([-10,1,10], [y1[1],y1[1],y1[1]] ,color="b",linestyle='--', linewidth=1.5)
plt.plot([-10,1,10], [y1[2],y1[2],y1[2]],color="r",linestyle='--', linewidth=1.5)
plt.ylabel("Total Power Consumption(mw)",size=12)
plt.xlabel("Stress Situation",size=12)
plt.legend(['3700','5700','8800'])
plt.title('Stress Under Maximum CPU Frequency Settings')
plt.xlim(-0.5,2.5)

plt.figure(1)
plt.bar(x, height=y2)
plt.xticks(x, ['Stress CPU','Stress Transceiver','Stress Both']);
plt.plot([-10,1,10], [y2[0],y2[0],y2[0]] ,color="y",linestyle='--', linewidth=1.5)
plt.plot([-10,1,10], [y2[1],y2[1],y2[1]] ,color="b",linestyle='--', linewidth=1.5)
plt.plot([-10,1,10], [y2[2],y2[2],y2[2]],color="r",linestyle='--', linewidth=1.5)
plt.ylabel("Total Power Consumption(mw)",size=12)
plt.xlabel("Stress Situation",size=12)
plt.legend(['1250','5600','6300'])
plt.title('Stress Under Low CPU Frequency Settings')
plt.xlim(-0.5,2.5)
plt.show()



