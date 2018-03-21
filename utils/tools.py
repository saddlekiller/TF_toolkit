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

def annotateImageDir(image_dir, loc):
    width = loc[2] - loc[0]
    height = loc[3] - loc[1]
    img=cv2.imread(image_dir)
    plt.figure(12)
    plt.imshow(img)
    currentAxis=plt.gca()
    rect=patches.Rectangle((loc[0], loc[1]),width,height,linewidth=1,edgecolor='r',facecolor='none')
    currentAxis.add_patch(rect)
    plt.show()

def annotateImage(image, loc):
    width = loc[2] - loc[0]
    height = loc[3] - loc[1]
    print('Weight %f Height %f' % (width, height))
    plt.figure(12)
    plt.imshow(image)
    currentAxis=plt.gca()
    rect=patches.Rectangle((loc[0], loc[1]),width,height,linewidth=1,edgecolor='r',facecolor='none')
    currentAxis.add_patch(rect)
    plt.show()

def reshapeImage(image, resize, loc):
    raw_shape = image.shape
    if len(resize) == 2:
        resize = (resize[0], resize[1], 3)
    else:
        resize = (resize[0], resize[1], resize[2])
    image_reshape = transform.resize(image, resize)
    x_ratio = resize[0] / raw_shape[0]
    y_ratio = resize[1] / raw_shape[1]
    print('Ratio => ', x_ratio, y_ratio)
    new_loc1 = loc[0] * y_ratio
    new_loc2 = loc[1] * x_ratio
    new_loc3 = loc[2] * y_ratio
    new_loc4 = loc[3] * x_ratio

    new_loc = [new_loc1, new_loc2, new_loc3, new_loc4] + loc[4:]
    return image_reshape, new_loc

def reshapeImageDir(image_dir, resize, loc):
    image=cv2.imread(image_dir)
    raw_shape = image.shape
    if len(resize) == 2:
        resize = (resize[0], resize[1], 3)
    else:
        resize = (resize[0], resize[1], resize[2])
    image_reshape = transform.resize(image, resize)
    x_ratio = resize[0] / raw_shape[0]
    y_ratio = resize[1] / raw_shape[1]
    new_loc1 = loc[0] * y_ratio
    new_loc2 = loc[1] * x_ratio
    new_loc3 = loc[2] * y_ratio
    new_loc4 = loc[3] * x_ratio

    new_loc = [new_loc1, new_loc2, new_loc3, new_loc4] + loc[4:]
    return image_reshape, new_loc

def checkImages(images_dir, labels_dir):

    image_filenames = [i[:-4] for i in os.listdir(images_dir) if i.find('.jpg') != -1]
    label_filenames = [i for i in os.listdir(labels_dir) if i.find('.txt') != -1]
    labels_dict = {}
    if labels_dir[-1] != '/':
        labels_dir += '/'
    upper_dir = '/'.join(labels_dir.split('/')[:-2]) + '/'

    for label_filename in label_filenames:
        lines = open(labels_dir + label_filename).readlines()
        for line in lines:
            splited = line.split()
            name_ = label_filename[:-4]
            id_ = splited[0]
            location = ' '.join(splited[2:])
            labels_dict[name_ + '+' + id_] = location
        # print(labels_dict)
        # break
    print('Loading files completed.')
    # new_labels = []
    writer = open(upper_dir + 'labels.txt', 'w')
    for name_id in labels_dict.keys():
        print(name_id)
        if name_id in image_filenames:
            new_line = ' '.join(name_id.split('+')) + ' ' + labels_dict[name_id]
            # new_labels.append(new_labels)
            writer.write(new_line + '\n')
    writer.close()
    print('Labels have been generated.')

def reshapeImageAll(images_dir, label_dir, shape):
    if images_dir[-1] != '/':
        images_dir += '/'
    lines = open(label_dir).readlines()
    label_dict = {}
    for line in lines:
        splited = line.split()
        label_dict[splited[0] + '+' + splited[1]] = [float(i) for i in splited[2:]]
    upper_dir = '/'.join(images_dir.split('/')[:-2]) + '/'
    savepath = upper_dir + 'reshape_images/'
    try:
        os.mkdir(savepath)
    except:
        pass
    image_filenames = [i for i in os.listdir(images_dir) if i.find('.jpg') != -1]

    writer = open(upper_dir + 'reshape_labels.txt', 'w')
    for filename in image_filenames:
        key = filename[:-4]
        try:
            image = cv2.imread(images_dir + filename)
            # reshape_image = transform.rescale(image, 0.5)
            reshape_image, new_loc = reshapeImage(image, shape, label_dict[key])
            reshape_image = reshape_image[:, :, ::-1]
            s1 = image.shape
            s2 = reshape_image.shape
            # print('Shape transormation: (%d, %d, %d) => (%d, %d, %d)'%(s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]))
            plt.imsave(savepath + filename, reshape_image)
            writer.write(key + ' ' + ' '.join([str(round(i, 2)) for i in new_loc]) + '\n')
        except:
            print('[ERROR_INFO]: ' + filename)
    writer.close()


#
if __name__ == '__main__':
    # retrieveURLDirs('/home/cheng/github/TF_toolkit/data/FACE/files')
    args = sys.argv
    retrieveURLFile(args[1], args[2])
    # retrieveURLFile('/home/cheng/github/TF_toolkit/data/FACE/files', 'A.J._Buckley.txt')

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
