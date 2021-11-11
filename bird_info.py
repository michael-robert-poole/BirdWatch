import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

class Bird_info:
    
    COMMON_NAME=["Wood Pigeon", "Long-Tailed Tit",
                 "Eurasian Magpie","European Goldfinch","European Robin","Eurasian Blue Tit","Great Tit","House Sparrow","Common Starling","Common Blackbird"]
    LATIN_NAME=["Columba palumbus", "Aegithalos caudatus", "Pica pica", "Carduelis carduelis", "Erithacus_rubecula", "Cyanistes caeruleus", "Parus major", "Passer_domesticus"
                , "Sturnus vulgaris", "Turdus merula"]

    
   
    def visualise_images(self, data, latin=False):
        names = self.LATIN_NAME if latin else self.COMMON_NAME
        
        plt.figure(figsize=(10, 10))
        for images, categories in data.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                class_index = np.where(categories[i].numpy()==1)
                plt.title(names[int(class_index[0])])
                plt.axis("off")