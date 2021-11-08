import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

DIRECTORY="all_birds"
IMG_SIZE=[256,256]
VAL_SPLIT=0.2;
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
        batch_size=4
  )
    return dataset