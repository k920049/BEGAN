import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

data1 = np.random.rand(50)
data2 = np.random.rand(30)

ax1.plot(data1, 'k--', label='data1')
ax2.plot(data2, 'k-', label='data2')

fig.show()