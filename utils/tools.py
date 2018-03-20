import numpy as np
import tensorflow as tf
from config_mapping import *
import pickle
import matplotlib.pyplot as plt
import urllib.request
import os
import socket
import sys



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


def build_image_(inputs, n_rows, bounder_size = 4):
    n, height, width, channel = inputs.shape
    n = int(n)
    height = int(height)
    width = int(width)
    channel = int(channel)
    # print(int(inputs.shape[0]))
    n_cols = int(n / n_rows)
    if int(n / n_rows) != (n / n_rows):
        raise TypeError
    x_size = (n_rows + 1) * bounder_size + n_rows * height
    y_size = (n_cols + 1) * bounder_size + n_cols * width
    image = np.zeros((x_size, y_size, channel))
    for i in range(n_rows):
        for j in range(n_cols):
            for k in range(3):
                image[(i+1) * bounder_size + i * width: (i+1) * bounder_size + (i + 1) * width, (j+1) * bounder_size + j * height: (j+1) * bounder_size + (j + 1) * height, k] = inputs[i * n_cols + j, :, :, k]
    return image


def retrieveURLDirs(direction):

    if direction[-1] != '/':
        direction += '/'
    upper_dir = '/'.join(direction.split('/')[:-2]) + '/'
    # print(upper_dir)
    filenames = [i for i in os.listdir(direction) if i.find('.txt') != -1]
    # writer = open(upper_dir + 'labels.txt', 'w')
    for filename in filenames:
        name = filename[:-4]
        basepath = direction
        # print(basepath, name)
        retrieveURLFile(basepath, name)
        # ids, urls, locs, name = retrieveURLFile(basepath, name)

        # print(ids, urls, locs, names)

        # for id_, loc_ in zip(ids, locs):
        #     writer.write('#'.join([name, id_, loc_]) + '\n')
        # writer.close()
        # break


def retrieveURLFile(basepath, name):
    if basepath[-1] != '/':
        basepath += '/'
    if name[-4:] != '.txt':
        name += '.txt'
    filename = basepath + name
    lines = open(filename).readlines()
    savepath = '/'.join(basepath.split('/')[:-2]) + '/images'
    try:
        os.mkdir(savepath)
    except:
        pass
    for line in lines:
        splited = line.split()
        id_ = splited[0]
        url_ = splited[1]
        loc_ = ' '.join(splited[2:])
        retrieveURL(url_, name[:-4] + '+' + id_ + '.jpg', savepath)


def retrieveURL(url, name, savepath):
    # try:
    #     urllib.request.urlretrieve(url, savepath + '/' + name)
    # except:
    #     count = 0
    #     while count <= 1:
    #         try:
    #             urllib.request.urlretrieve(url, savepath + '/' + name)
    #             break
    #         except:
    #             count += 1
    #             pass
    #     # if count > 1:
    #     temp = name[:-4].split('+')
    #     person = temp[0]
    #     index = temp[1]
    #     print('[FAILURE] name: %s, id: %s'%(person, index))
    #     pass
    socket.setdefaulttimeout(3)
    try:
        ss = urllib.request.urlopen(url)
        response = ss.read()
        ss.close()
        fp = open(savepath + '/' + name,"wb")
        fp.write(response)
        fp.close()
    except socket.timeout:
        temp = name[:-4].split('+')
        person = temp[0]
        index = temp[1]
        print('[FAILURE] name: %s, id: %s'%(person, index))
        pass
    except:
        temp = name[:-4].split('+')
        person = temp[0]
        index = temp[1]
        print('[FAILURE] name: %s, id: %s'%(person, index))
        pass



def retrieveCheck(images_dir, labels_dir):

    images = [i for i in os.listdir(images_dir) if i.find('.jpg') != -1]
    images_pairs = []
    for image in images:
        temp = image[:-4].split('+')
        images_pairs.append(temp)

    new_labels = []
    labels = open(labels_dir).readlines()
    for label in labels:
        pairs = label.split('#')
        if pairs in images_pairs:
            new_labels.append(label)
        else:
            print(pairs)





#
if __name__ == '__main__':
    # retrieveURLDirs('/home/cheng/github/TF_toolkit/data/FACE/files')
    # args = sys.argv
    # retrieveURLFile(args[1], args[2])
    retrieveURLFile('/home/cheng/github/TF_toolkit/data/FACE/files', 'A.J._Buckley.txt')

    # retrieveURL('http://www.contactmusic.com/pics/ld/active_for_life_arrivals_090110/a.j_buckley_2706152.jpg')

    # data = []
    # for i in range(16):
    #     temp = np.random.random((28, 28, 3))
    #     data.append(temp)
    # data = np.array(data)
    # print(data.shape)
    # m = build_image_(data, 4)
    # plt.imshow(m)
    # plt.show()
