import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import math


COMMON_NAME=["Wood Pigeon", "Long-Tailed Tit",
                 "Eurasian Magpie","European Goldfinch","European Robin","Eurasian Blue Tit","Great Tit","House Sparrow","Common Starling","Common Blackbird"]
LATIN_NAME=["Columba palumbus", "Aegithalos caudatus", "Pica pica", "Carduelis carduelis", "Erithacus_rubecula", "Cyanistes caeruleus", "Parus major", "Passer_domesticus"
                , "Sturnus vulgaris", "Turdus merula"]
   
def visualise_images(images, labels, latin=False):
    names = LATIN_NAME if latin else COMMON_NAME
    print(len(images))
    indexes=np.arange(start=0, stop=len(images)-2, step=math.ceil(len(images)/100))
    print(labels[indexes[99]])
    plt.figure(figsize=(30, 30))
    j=0
    for i in indexes:
        ax = plt.subplot(10, 10, j + 1)
        plt.imshow(images[i])
        plt.title(names[labels[i]] + str(images[i].shape))
        plt.axis('off')
        j+=1
            