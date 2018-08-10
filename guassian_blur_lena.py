import numpy as np

from scipy.signal import convolve2d

import matplotlib.pyplot as plt

import matplotlib.image as mpimg


#original pic
img = mpimg.imread('/home/wout/Codebase/machine_learning_examples/cnn_class/lena.png')
# plt.imshow(img)
# plt.show()
# print('Shape of original image: ', img.shape)
#


#black and white

# mean on second axis because 2d convolution is only for 2d matrices
bw = img.mean(axis=2)
# print('Shape of bw: ', bw.shape)
# plt.imshow(bw,cmap='gray')
#plt.show()

W = np.zeros((20,20))

for i in range(20):
    for j in range(20):
        dist = (i - 9.5)**2 + (j - 9.5)**2
        W[i,j] = np.exp(-dist/50)

# plt.imshow(W, cmap='gray')
# plt.show()

out = convolve2d(bw, W, mode='same')

# plt.imshow(out, cmap='gray')
# plt.show()

print(out.shape)

out3 = np.zeros(img.shape)


# W /= W.sum()
for i in range(3):
    out3[:,:,i] = convolve2d(img[:,:,i], W, mode='same')
out3 /= out3.max() # can also do this if you didn't normalize the kernel
# plt.imshow(out3)
# plt.show()


Hx = np.array([
    [-1,0,1],
    [-2,0,2],
    [-1,0,1],
])

Hy = Hx.T

Gx = convolve2d(bw, Hx)

# plt.imshow(Gx, cmap='gray')
# plt.show()

Gy = convolve2d(bw, Hy)

# plt.imshow(Gy, cmap='gray')
# plt.show()

G = np.sqrt(Gx*Gx + Gy*Gy)

# plt.imshow(G, cmap='gray')
# plt.show()

theta = np.arctan2(Gy, Gx)

plt.imshow(theta, cmap='gray')
plt.show()







