from IPython.display import clear_output
from sklearn.utils import shuffle
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


def resize_image(image, dim=512):
    """
    resizes the input iput image using the value of dim
    """
    img = cv2.resize(image, (dim, dim))
    return img


def collapse_channel(image,flag='macro'):
    """
    takes an image with 3 channels and tranforms it to 1-channeled image
    """
    # print('[ STATUS ====== collapsing channels ]')
    additional_encoding = np.array([[128, 0, 0], [128, 128, 128], [128, 0, 128], [128, 128, 0]])
    collapsed = np.zeros((image.shape[0], image.shape[1]))
    uniq_arr = np.unique(image.reshape(-1, image.shape[2]), axis=0)
    for i, c in enumerate(uniq_arr):
        indices = np.where(np.all(image == c, axis=-1))
        # TODO: make this piece more dynamic
        if flag == 'micro':
            present = np.where(np.all(additional_encoding==c, axis=-1))
            if len(present[0]) != 0:
                collapsed[indices] = 2
            else:
                collapsed[indices] = i
        else:
            collapsed[indices] = i
    print()

    return collapsed


def normalize_image(image):
    """
    returns the normalized image
    """
    img = np.asarray(image).astype(float)
    return img/255.0


def data_augmentation():
    data_aug = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
    )

    return data_aug


def get_data(image_paths, mask_paths, flag='train', dim=512):
    """
    reads the images: normal and mask
    returns two arrays: one for images and one for 1channel mask
    """
    print('[ STATUS ===== fetching data ]')
    images = []
    masks = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image_names = os.listdir(image_path)
        mask_names = os.listdir(mask_path)

        if 'courtney' in image_path or 'tim' in image_path:
            group = 'micro'
        else:
            group='macro'

        for img_name, msk_name in zip(image_names, mask_names):
            # read images
            img = cv2.imread(os.path.join(image_path, img_name))
            mask = cv2.imread(os.path.join(mask_path, img_name[:-4] + '.png'))

            # transform mask: 3->1 channel
            mask = collapse_channel(np.array(mask), group)

            # resize 
            img = resize_image(img)            
            mask = resize_image(mask)

            # normalize
            img = normalize_image(img)

            images.append(img)
            masks.append(mask)

    # shuffle images and masks to avoid feeding the NN correlated images
    print('[ STATUS: end data fetch ===== shuffling set and return ]')
    images, masks = shuffle(images, masks)
    return np.array(images), np.array(masks)


def display_images(elements):
    """
    displays the images to check consistency between 
    """
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(elements)):
        plt.subplot(1, len(elements), i + 1)
        plt.title(title[i])
        plt.imshow(elements[i])
        plt.axis('off')
    plt.show()

