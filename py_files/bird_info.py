import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import py_files
from py_files import const

   
def visualise_10_10_images(data, latin=False):
    common_name = 0 if latin else 1
    indexes=np.arange(start=0, stop=len(data), step=math.ceil(len(data)/100))
    plt.figure(figsize=(30, 30))
    j=0
    for i in indexes:
        ax = plt.subplot(10, 10, j + 1)
        plt.imshow(data[i,0])
        plt.title(const.BIRDS_DICT[data[i,1]][0])
        plt.axis('off')
        j+=1
            