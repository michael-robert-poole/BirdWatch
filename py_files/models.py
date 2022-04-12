import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Dropout, Rescaling, RandomFlip, RandomRotation

def leCun(INPUT_SHAPE):
    leCun_model = keras.Sequential(
        [
            Conv2D(filters=6, kernel_size=(5,5), activation="relu", input_shape=(INPUT_SHAPE)),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=16, kernel_size=(5,5), activation="relu"),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=120, kernel_size=(5,5), activation="relu"),
            Flatten(),
            Dense(84, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )
    return leCun_model

def deeper_model(INPUT_SHAPE):
    
    deeper_model = keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        
        Conv2D(filters=16, kernel_size=(3, 3), strides=1, kernel_regularizer=tf.keras.regularizers.l2(0.0005),padding="valid", activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=32, kernel_size=(3, 3), strides=1, kernel_regularizer=tf.keras.regularizers.l2(0.0005), padding="valid", activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=56, kernel_size=(3, 3), strides=1, padding="valid", activation='relu'),
        MaxPooling2D(pool_size=2),
        Conv2D(filters=86, kernel_size=(3, 3), strides=1, padding="valid", activation='relu'),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dropout(0.5),
        Dense(units=320, activation='relu'),
        Dense(units=140, activation='relu'),
        Dropout(0.5),
        Dense(units=60, activation='relu'),
        Dense(units=10, activation = 'softmax'),
    ])
    return deeper_model

from tensorflow.keras.applications.resnet50 import ResNet50

def resNet_50_transfer_learning_model(INPUT_SHAPE):
    base_model=ResNet50(weights='imagenet', include_top = False)

    base_model.trainable=False
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = base_model(inputs, training=False)

    
    x = keras.layers.GlobalAveragePooling2D()(x)
   
    outputs = keras.Sequential(
        [
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(622, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(124, activation="relu"),
            keras.layers.Dense(62, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )(x)
    model = keras.Model(inputs, outputs)
    
    return model

def VGG(INPUT_SHAPE):
    base_model=keras.applications.VGG16(weights='imagenet', include_top = False)
    base_model.trainable=False
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = base_model(inputs, training=False)

    
    x = keras.layers.GlobalAveragePooling2D()(x)
   
    outputs = keras.Sequential(
        [
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dense(622, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(124, activation="relu"),
            keras.layers.Dense(62, activation="relu"),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )(x)
    model = keras.Model(inputs, outputs)
    
    return model
    
 
def alex_net():
    #Instantiation
    model=keras.Sequential([
    
    RandomFlip("horizontal"),
    RandomRotation(0.1),

    #1st Convolutional Layer - 96 filters instead of two parllel 48 filters
    Conv2D(filters=96, input_shape=(256, 256,3), kernel_size=(3,3), strides=(4,4), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

    #2nd Convolutional Layer
    Conv2D(filters=256, kernel_size=(5, 5), strides=(1,1), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

    #3rd Convolutional Layer
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    #4th Convolutional Layer
    Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same'),
    BatchNormalization(),
    Activation('relu'),

    #5th Convolutional Layer
    Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'),

    #Passing it to a Fully Connected layer
    Flatten(),
    # 1st Fully Connected Layer
    Dense(4096, input_shape=(32,32,3,)),
    BatchNormalization(),
    Activation('relu'),
    # Add Dropout to prevent overfitting
    Dropout(0.4),

    #2nd Fully Connected Layer
    Dense(4096),
    BatchNormalization(),
    Activation('relu'),
    #Add Dropout
    Dropout(0.4),

    #3rd Fully Connected Layer
    Dense(1000),
    BatchNormalization(),
    Activation('relu'),
    #Add Dropout
    Dropout(0.4),

    #Output Layer
    Dense(10),
    BatchNormalization(),
    Activation('softmax'),
    ])
    return model
 