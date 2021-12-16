import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import py_files
from py_files import const

   
def visualise_10_10_images(data, labels_value, latin=False):
    common_name = 0 if latin else 1
    random_indexes=[np.random.choice((np.where(data[:,1]==label)[0]), 10) for label in labels_value]
    random_indexes=np.array(random_indexes).reshape(100)
    plt.figure(figsize=(30, 30))
    for i, r_index in enumerate(random_indexes):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(data[r_index,0])
        plt.title(const.BIRDS_DICT[data[r_index,1]][0])
        plt.axis('off')
            