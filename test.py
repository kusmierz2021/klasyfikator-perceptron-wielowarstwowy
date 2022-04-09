import load_data as load
import neuron as n
import network as net
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

train_images, train_labels, val_images, val_labels, test_images, test_labels = load.load_dataset()

network = []
network.append(n.Dense(train_images.shape[1], 100))
network.append(n.ReLU())
network.append(n.Dense(100, 200))
network.append(n.ReLU())
network.append(n.Dense(200, 10))

val_try = []
train_try = []


# val_try.append(np.mean(net.predict(network, val_images) == val_labels))
# train_try.append(np.mean(net.predict(network, train_images) == train_labels))


# for train in trange(250):
#     net.train(network, train_images, train_labels)
#     val_try.append(np.mean(net.predict(network, val_images) == val_labels))
#     train_try.append(np.mean(net.predict(network, test_images) == test_labels))


for train in trange(250):
    for probe in trange(20):
        net.train(network, train_images, train_labels)
        val_try.append(np.mean(net.predict(network, val_images) == val_labels))
        train_try.append(np.mean(net.predict(network, train_images) == train_labels))


plt.plot(val_try, label='dokładność dla walidacyjnych', color='magenta')
plt.plot(train_try, label='dokładność dla treningowych', color='cyan')
plt.legend(loc='best')
plt.grid()
plt.show()


successfull = 0
for i in range(len(test_images)):

    x = net.predict(network, test_images[i])
    if(x == test_labels[i]):
        successfull = successfull+1

print('ilosc udanych prob -> %d na 10 000' % successfull)


for i in range(18):
    pre = net.predict(network, test_images[i])
    label = test_labels[i]
    plt.title('image number: %d  -> for label %d predicted %d' % (i+1, label, pre))
    plt.imshow(test_images[i].reshape([28, 28]))
    plt.show()
