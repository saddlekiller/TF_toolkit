import numpy as np
import tensorflow as tf
from config_mapping import *
import pickle


def acc_sum(results, targets):
    return tf.reduce_sum(tf.cast(tf.equal(tf.argmax(results, 1), tf.argmax(targets, 1)), tf.float32))

def acc_mean(results, targets):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(results, 1), tf.argmax(targets, 1)), tf.float32))

def loss_sum(results, targets, config):
    return tf.reduce_sum(loss_mapping(config['loss'])(logits = results, labels = targets))

def loss_mean(results, targets, config):
    return tf.reduce_mean(loss_mapping(config['loss'])(logits = results, labels = targets))

def raw2ids(sentence, dictionary):
    words = []
    for word in sentence:
        if word in dictionary:
            words.append(dictionary.index(word))
        else:
            word.append(dictionary.index('UNKNOWN'))
    return words


def ids2raw(ids, dictionary):
    try:
        sentence = ''.join([dictionary[i] for i in ids])
        return sentence
    except:
        logging_io.ERROR_INFO('Some words are not included in vocabulary, please check it!')
    return None

# if __name__ == '__main__':
#     corpus_dir = '../../Basic_Tensorflow/corpus/'
#     # label_map = pickle.load(open(corpus_dir + 'poilabel_map.npz', 'rb'))
#     dictionary = pickle.load(open(corpus_dir + 'poiwords.dict', 'rb'))
#     raw_sentence = '我要去上海'
#     ids = raw2ids(raw_sentence, dictionary)
#     print(ids)
#     print(ids2raw(ids, dictionary))

def build_image(inputs, n_rows, bounder_size = 4):
    height, width, channel = inputs.shape
    height = int(height)
    width = int(width)
    channel = int(channel)
    # print(int(inputs.shape[0]))
    n_cols = int(channel / n_rows)
    if int(channel / n_rows) != (channel / n_rows):
        raise TypeError
    x_size = (n_rows + 1) * bounder_size + n_rows * height
    y_size = (n_cols + 1) * bounder_size + n_cols * width
    # print(x_size, y_size)
    image = np.zeros((x_size, y_size))
    # print(width, height)
    # print(image)
    # print(n_rows)
    # print(n_cols)
    # print(inputs[:, :, i * n_cols + j])
    # print(inputs[:, :, i * n_cols + j])
    for i in range(n_rows):
        for j in range(n_cols):
            image[(i+1) * bounder_size + i * width: (i+1) * bounder_size + (i + 1) * width, (j+1) * bounder_size + j * height: (j+1) * bounder_size + (j + 1) * height] = inputs[:, :, i * n_cols + j]
    return image
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()

# if __name__ == '__main__':
#     data = []
#     for i in range(16):
#         temp = np.random.random((28, 28))
#         data.append(temp)
#     data = np.array(data).reshape([28, 28, 16])
#     print(data.shape)
#     build_image(data, 4)
