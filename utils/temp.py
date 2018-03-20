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
    new_loc1 = loc[0] * x_ratio
    new_loc2 = loc[1] * y_ratio
    new_loc3 = loc[2] * x_ratio
    new_loc4 = loc[3] * y_ratio

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
    new_loc1 = loc[0] * x_ratio
    new_loc2 = loc[1] * y_ratio
    new_loc3 = loc[2] * x_ratio
    new_loc4 = loc[3] * y_ratio

    new_loc = [new_loc1, new_loc2, new_loc3, new_loc4] + loc[4:]
    return image_reshape, new_loc

def checkImages(images_dir, labels_dir):
    image_filenames = [i[:-4] for i in os.listdir(images_dir) if i.find('.jpg') != -1]
    label_filenames = [i for i in os.listdir(labels_dir) if i.find('.txt') != -1]
    labels_dict = {}
    if labels_dir[-1] != '/':
        labels_dir += '/'
    for label_filename in label_filenames:
        lines = open(labels_dir + label_filename).readlines()
        for line in lines:
            splited = line.split()
            name_ = label_filename[:-4]
            id_ = splited[0]
            location = ' '.join(splited[2:])
            labels_dict[name_ + '+' + id_] = location
        print(labels_dict)
        break
    new_labels = []
    for name_id in labels_dict.keys():
        if name_id in image_filenames:
            new_line = ' '.join(name_id.split('+')) + ' ' + labels_dict[name_id]
            new_labels.append(new_labels)
            print(new_line)

# image_dir = '../data/FACE/images/Sarah_Roemer+00000006.jpg'
# loc = [float(i) for i in '107.00 85.67 202.67 181.33 3.00 3.07 0'.split()]
# image = cv2.imread(image_dir)
# annotateImage(image, loc)
# image_reshape, new_loc = reshapeImage(image, (800, 666), loc)
# annotateImage(image_reshape, new_loc)

# checkImages('../data/FACE/images', '../data/FACE/files')
