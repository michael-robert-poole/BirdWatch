import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

DIRECTORY="birds"
IMG_SIZE=[256,256]
VAL_SPLIT=0.1
SEED=133
   

def split_data(subset, label_mode,labels):
    dataset=keras.preprocessing.image_dataset_from_directory(
        directory=DIRECTORY,
        image_size=IMG_SIZE,
        validation_split=VAL_SPLIT,
        seed=SEED,
        subset=subset,
        label_mode= label_mode,
        labels=labels,
        batch_size=16
  )
    return dataset

def load_data_from_folder(parent_dir):
    images = []
    labels = []
    label = 0

    parent_folder = os.listdir(parent_dir)
    parent_folder.sort()
    print("Loading images....")
    for f in parent_folder:
        print(f)
        sub_folder=os.listdir(parent_dir+"/"+f)
        for image in sub_folder:
            theImage=cv2.imread(parent_dir+"/"+f+"/"+image, cv2.IMREAD_COLOR)
            img_resize = cv2.resize(theImage, (256,256))
            img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)
            labels.append(label)
        label=label+1
    return images, labels
