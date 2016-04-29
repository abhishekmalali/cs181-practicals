import numpy as np
import matplotlib.pylab as plt

hist = np.load("hist.npy")
plt.plot(hist)
plt.show()
plt.plot(np.convolve(hist, np.ones(50)/50, mode="valid"))
plt.show()
