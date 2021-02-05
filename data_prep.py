from IPython.display import clear_output
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def resize_image(image, dim=512):
    img = cv2.resize(image, (dim, dim))
    return img

def normalize_image(image):
    img = np.asarray(image).astype(float)
    return img/255.0


def get_data(image_paths, mask_paths, flag='train', dim=512):
    images = []
    masks = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        image_names = os.listdir(image_path)
        mask_names = os.listdir(mask_path)

        for img_name, msk_name in zip(image_names, mask_names):
            img = cv2.imread(os.path.join(image_path, img_name))
            mask = cv2.imread(os.path.join(mask_path, img_name[:-4] + '.png'))

            # normalize train images
            img = resize_image(img)
            img = normalize_image(img)

            mask = resize_image(mask)

            images.append(img)
            masks.append(mask)
    return np.array(images), np.array(masks)


def display_images(elements):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(elements)):
        plt.subplot(1, len(elements), i + 1)
        plt.title(title[i])
        plt.imshow(elements[i])
        plt.axis('off')
    plt.show()

