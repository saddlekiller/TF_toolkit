import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage import transform
import os
import sys


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

def reshapeImageAll(images_dir, label_dir, shape, isLinux = False):
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
            plt.show()
            s1 = image.shape
            s2 = reshape_image.shape
            # print('Shape transormation: (%d, %d, %d) => (%d, %d, %d)'%(s1[0], s1[1], s1[2], s2[0], s2[1], s2[2]))
            if isLinux == True:
                filename = filename[:-4] + '.png'
            # print(filename)
            plt.imsave(savepath + filename, reshape_image)
            # image_writer = open(savepath + filename, 'wb')
            # image_writer.write(reshape_image)
            # image_writer.close()
            writer.write(key + ' ' + ' '.join([str(round(i, 2)) for i in new_loc]) + '\n')
        except:
            # print(filename)
            print('[ERROR_INFO]: ' + filename)
        # break
    writer.close()




# image_dir = '../data/FACE/images/A.J._Buckley+00000206.jpg'
# loc = [float(i) for i in '98.86 112.93 225.20 239.27 3.00 3.98 1'.split()]
# image = cv2.imread(image_dir)
# print(image.shape)
# annotateImage(image, loc)
# image_reshape, new_loc = reshapeImage(image, (100, 80, 3), loc)
# annotateImage(image_reshape, new_loc)
# print(loc)
# print(new_loc)

# checkImages('../data/FACE/images_v1', '../data/FACE/files_v1')
reshapeImageAll('../data/FACE/images_v1', '../data/FACE/labels.txt', [256, 128, 3], True)
