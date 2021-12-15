import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import py_files
from py_files import const

   
def visualise_10_10_images(data, latin=False):
    common_name = 0 if latin else 1
    random_indexes=[np.random.choice((np.where(data[:,1]==label)[0]), 10) for label in labels_value]
    plt.figure(figsize=(30, 30))
    for i, r_index in enumerate(random_indexes):
        ax = plt.subplot(10, 10, i + 1)
        plt.imshow(data[r_index,0])
        plt.title(const.BIRDS_DICT[data[r_index,1]][common_name])
        plt.axis('off')
      
            