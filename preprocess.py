# Import required modules=====================
from skimage.color import rgb2gray
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import timeit
#============================================

def image_segmentain(img_flat):
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_flat.reshape(-1, 1))
    """Kmeans lables had issue with masking so center of each cluster
      is assigned for corresponding labels"""

    kmeans_centers = kmeans.cluster_centers_[kmeans.labels_]

    return kmeans_centers.flatten()


def image_mask(kmeans_labels, img_gray_orig):
    mask_img = np.zeros((img_gray_orig.shape[0], img_gray_orig.shape[1]))
    kmeans_labels_arr = kmeans_labels.reshape(img_gray_orig.shape[0], img_gray_orig.shape[1])

    sort_labels = sorted(pd.Series(kmeans_labels).unique(), reverse=True)
    just_bone = ()

    if (np.sum(kmeans_labels_arr == sort_labels[0])) > 8000:
        just_bone = np.where(kmeans_labels_arr == sort_labels[0])
        mask_img[just_bone] = 1
    #         print('test1')

    if (np.sum(kmeans_labels_arr == sort_labels[1])) > 8000 and (np.sum(kmeans_labels_arr == sort_labels[1])) < 60000:
        just_bone = np.where(kmeans_labels_arr == sort_labels[1])
        mask_img[just_bone] = 1
    #         print('test2')
    if (np.sum(kmeans_labels_arr == sort_labels[2])) > 8000 and (np.sum(kmeans_labels_arr == sort_labels[2])) < 70000:
        just_bone = np.where(kmeans_labels_arr == sort_labels[2])
        mask_img[just_bone] = 1
    #         print('test3')
    if (np.sum(kmeans_labels_arr == sort_labels[3])) > 8000 and (np.sum(kmeans_labels_arr == sort_labels[3])) < 70000:
        just_bone = np.where(kmeans_labels_arr == sort_labels[3])
        mask_img[just_bone] = 1
    #         print('test4')
    if not just_bone:
        just_bone = np.where(kmeans_labels_arr == sort_labels[1])
        mask_img[just_bone] = 1
    #     print('test4')

    #   plt.imshow(mask_img)
    #   plt.show()
    return just_bone, mask_img


def img_resize(img, img_height):
    img_width = int(img_height * img.shape[1] / img.shape[0])

    img_pil = Image.fromarray(img)  # convert array back to image

    img_pil_resize = img_pil.resize((img_width, img_height), Image.LANCZOS)  # resize

    return np.array(img_pil_resize)


def img_pad_resize(img_just_bone, image_size):
    size_diff = img_just_bone.shape[0] - img_just_bone.shape[1]

    if size_diff > 0:  # hieght is longer than width
        top = 0
        bottom = 0
        left = int(abs(size_diff) / 2.)
        right = (abs(size_diff) - left)
    elif size_diff < 0:  # hieght is shorter than width
        left = 0
        right = 0
        top = int(abs(size_diff) / 2.)
        bottom = (abs(size_diff) - top)
    else:
        top = 0
        bottom = 0
        left = 0
        right = 0

    img_bone_square = np.pad(img_just_bone, ((top, bottom), (left, right)), 'constant')

    img_bone = img_resize(img_bone_square, image_size)

    #   plt.imshow(img_bone)
    #   plt.show()

    return img_bone


def img_preprocess_core(img_gray_orig):
    img_flat = img_gray_orig.reshape(img_gray_orig.shape[0] * img_gray_orig.shape[1])

    kmeans_labels = image_segmentain(img_flat)

    kmeans_labels_arr = kmeans_labels.reshape(img_gray_orig.shape[0], img_gray_orig.shape[1])

    just_bone, mask_img = image_mask(kmeans_labels, img_gray_orig)

    img_clean_background = mask_img * img_gray_orig

    img_just_bone = img_clean_background[min(just_bone[0]):max(just_bone[0]), \
                    min(just_bone[1]):max(just_bone[1])]
    return img_just_bone


def img_preprocessing(img_path, filename, pre_filename):
    image_size = 256
    save_path_filename = img_path + pre_filename

    image = plt.imread(img_path + filename)
    img_gray_orig_0 = rgb2gray(image)

    #     plt.imshow(img_gray_orig_0)
    #     plt.show()
    img_gray_orig = img_resize(img_gray_orig_0, 2 * image_size)

    img_just_bone = img_preprocess_core(img_gray_orig)

    try:
        img_bone = img_pad_resize(img_just_bone, 2 * image_size)
        # Second iteration of image segmentation
        img_just_bone = img_preprocess_core(img_bone)
        img_bone = img_pad_resize(img_just_bone, image_size)

        plt.imsave(save_path_filename, img_bone)

    except ValueError:
        print(filename)
    return img_bone