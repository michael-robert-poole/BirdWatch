import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math
import const

   
def visualise_images(images, labels, latin=False):
    common_name = 0 if latin else 1
    indexes=np.arange(start=0, stop=len(images)-2, step=math.ceil(len(images)/100))
    plt.figure(figsize=(30, 30))
    j=0
    for i in indexes:
        ax = plt.subplot(10, 10, j + 1)
        plt.imshow(images[i])
        plt.title(const.BIRDS_DICT[labels[i]][common_name])
        plt.axis('off')
        j+=1
            