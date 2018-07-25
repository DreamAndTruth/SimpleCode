import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_file(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo, encoding='latin1')
    return data

data = load_file('../data/cifar_10/cifar-10-batches-py/test_batch')
print(data.keys())
x = data['labels']
y = data['data']
print(x[0])
print(y.shape)

test = y[3, :]
fig = np.reshape(test, [3, 32, 32])
print(fig.shape)

red = fig[0].reshape(1024, 1)
green = fig[1].reshape(1024, 1)
blue = fig[2].reshape(1024, 1)
pic = np.hstack((red, green, blue))
pic_g = pic.reshape(32,32,3)

plt.imshow(pic_g)
plt.show()
