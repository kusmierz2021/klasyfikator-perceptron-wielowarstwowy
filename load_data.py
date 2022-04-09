import idx2numpy


def load_dataset():
    train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

    train_images = train_images.astype(float) / 255.
    test_images = test_images.astype(float) / 255.
    # tworzymy 10000 el set validacyjny i 50000 el set treningowy
    train_images, val_images = train_images[:-10000], train_images[-10000:]
    train_labels, val_labels = train_labels[:-10000], train_labels[-10000:]
    train_images = train_images.reshape([train_images.shape[0], -1])
    val_images = val_images.reshape([val_images.shape[0], -1])
    test_images = test_images.reshape([test_images.shape[0], -1])
    return train_images, train_labels, val_images, val_labels, test_images, test_labels
