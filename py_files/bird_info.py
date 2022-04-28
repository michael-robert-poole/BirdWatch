import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import py_files
from py_files import const
import cv2

   
def visualise_10_10_images(images, labels, labels_value, common_name=True):
    
    random_indexes=[np.random.choice((np.where(labels==label)[0]), 10, replace = False) for label in labels_value]
    random_indexes=np.array(random_indexes).reshape(100)
    plt.figure(figsize=(30, 30))
    for i, r_index in enumerate(random_indexes):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(cv2.cvtColor(images[r_index], cv2.COLOR_BGR2RGB))
        plt.title(const.BIRDS_DICT[labels[r_index]][common_name])
        plt.axis('off')
            