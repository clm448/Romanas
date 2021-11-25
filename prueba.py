import numpy as np
import matplotlib.pyplot as plt

data = np.load('PNOA_CYL_SW_2009_50cm_OF_rgb_etrs89_hu30_H10_0554_1-1.npy')

print(type(data))
plt.imshow(data)
plt.show()

data.size