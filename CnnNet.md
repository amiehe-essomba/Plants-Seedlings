L'architecture de notre modèle est définie comme suit :

![logo](/images/cnn_schema.png)                                 

La taille de l'image d'entrée est 160x160 avec 3 canaux. Cette image est soumise à 5 couches de convolution, suivies d'une couche entièrement connectée et d'une couche de classification de 12 neurones.

> La première couche de convolution (cov1) utilise un noyau de convolution de 11x11, un pas de 4x4 et comporte 96 filtres.cette La couche est suivie d'une opération de max pooling avec un pooling size de 2x2. La fonction d'activation utilisée dans cov1 est la fonction ReLU (Rectified Linear Unit).

> La deuxième couche de convolution (cov2) utilise un noyau de 5x5, un pas de 1x1 et comporte 256 filtres. Après la cov2, nous avons une opération de max pooling avec un pooling size de 2x2. La fonction d'activation dans cov2 est ReLU.

> Les troisième et quatrième couches de convolutions (cov3 et cov4) utilisent un noyau de 3x3, un pas de 1x1 et comportent 384 filtres chacune. Après les cov3 et cov4, nous avons une opération de max pooling avec un pooling size de 1x1. La fonction d'activation dans cov3 et cov4 est ReLU. Après le max pooling, il y a une opération de dropout pour réduire le surapprentissage.

> La cinquième couche de convolution (cov5) utilise un noyau de 3x3, un pas de 1x1 et comporte 512 filtres. Après cov5, nous avons une opération de max pooling avec un pooling size de 1x1. La fonction d'activation dans cov5 est ReLU. Après le max pooling, il y a une opération de dropout pour réduire le surapprentissage.

> Après les 5 couches de convolution, il y a une opération Flatten() pour aplatir les données et obtenir une matrice de dimension (m, n), où m est le nombre d'images dans le train set et n est le nombre de paramètres après la 5ème couche de convolution, soit (384x3x3x512 paramètres).

> Ensuite, une première couche entièrement connectée (FC) de 4096 neurones est ajoutée.

> Enfin, une couche de classification constituée de 12 neurones est ajoutée pour la détection des 12 classes de notre échantillon.

```python
import numpy as np
import panas as pd 
import tensorflow as tf 
from keras import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, Input, ReLU
from keras.models import Model

def ConvNet(
  shape     : tuple = (128, 128, 3),  # dimension des images 
  classes   : int   = 12              # nombre de classes
  ):

  # creating the input layer
    inputs = tf.keras.layers.Input(shape=input_shape)

    # première couche de convolution stride = (4,4), padding="valid", kernel=(11, 11), pol_size = (2,2), filters=96
    X = tf.keras.layers.Conv2D(
        filters=32, kernel_size=(9, 9), 
        kernel_initializer="glorot_uniform",
        strides=(2, 2), padding="valid"
        )(inputs)
    # fonction  d'activation 
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # réduction de dimension par 2
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2)
        )(X)
    X = tf.keras.layers.Dropout(rate=0.4)(X)

    # deuxième couche de convolution stride = (1,1), padding="valid", kernel=(5,5), pol_size = (2,2), filters=256
    X = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(7, 7), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="valid"
        )(X)
    # fonction  d'activation 
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # réduction de dimension par 2
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1), padding="valid"
        )(X)
    X = tf.keras.layers.Dropout(rate=0.4)(X)

    # troisième couche de convolution stride = (1,1), padding="valid", kernel=(5,5), pol_size = (1, 1), filters =384
    X = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="valid"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1), padding="valid"
        )(X)
    X = tf.keras.layers.Dropout(rate=0.3)(X)

    # quatrième couche de convolution stride = (1,1), padding="valid", kernel=(1, 1), 
    # filters =384, pol_size = (1, 1), drop_out = 0.7
    X = tf.keras.layers.Conv2D(
        filters=126, kernel_size=(3, 3), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="valid"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1) 
        )(X)
    # déconnection de 20 % des couches de façon random pour limiter le surapprentissage 
    X = tf.keras.layers.Dropout(rate=0.2)(X)

    # 5eme couche de convolution stride = (1,1), padding="valid", kernel=(1, 1), filters = 512, drop_out = 0.5
    X = tf.keras.layers.Conv2D(
        filters=384, kernel_size=(3, 3), 
        kernel_initializer="glorot_uniform",
        strides=(2, 2), padding="valid"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1) 
        )(X)
    X = tf.keras.layers.Dropout(rate=0.4)(X)
    # 6eme couche de convolution stride = (1,1), padding="valid", kernel=(1, 1), filters = 512, drop_out = 0.5
    X = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="valid"
        )(X)
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1) 
        )(X)
    X = tf.keras.layers.Dropout(rate=0.5)(X)
    # 7eme couche de convolution 
    
    X = tf.keras.layers.Conv2D(
        filters=1024, kernel_size=(3, 3), 
        kernel_initializer="glorot_uniform",
        strides=(1, 1), padding="same"
        )(X)
    # fonction d'activation
    X = tf.keras.layers.ReLU()(X)
    X = tf.keras.layers.BatchNormalization(axis=3)(X)
    # conversation de la taille de l'image padding = "same" ---> stride = (1, 1)
    X = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(1, 1) 
        )(X)
    # déconnection de 20 % des couches de façon random pour limiter le surapprentissage 
    X = tf.keras.layers.Dropout(rate=0.5)(X)
   
    # applatissement 
    X = tf.keras.layers.Flatten()(X)

    # 1ere couche full connected, avec 4096 neurones et relu comme function d'activation
    X = tf.keras.layers.Dense(units= 4096, 
                              activation=tf.keras.activations.relu)(X)
    # déconnection de 50 % de neurone de façon random pour limiter le surapprentissage 
    X = tf.keras.layers.Dropout(rate=0.5)(X)

    # couche de classification (12 classes)
    X = tf.keras.layers.Dense(units=classes, activation=tf.keras.activations.softmax)(X)

    # output 
    outputs = X

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    return model

```

```python
def Callbacks(epoch):

    log_dir = "./embeding_models/logs"
    callback_tf = tf.keras.callbacks.\
        TensorBoard(log_dir=log_dir, histogram_freq=1)

    path_models = "./embeding_models/model-{epoch:04d}.h5"
    callbacks_models = tf.keras.callbacks.\
        ModelCheckpoint(filepath=path_models, verbose=1)
    
    path_best_models = f"./embeding_models/best-{color}-{formats}-model.h5"
    callbacks_best_models = tf.keras.callbacks.\
        ModelCheckpoint(filepath=path_best_models, 
            monitor="val_accuracy", verbose=1, save_best_only=True)
    

    earlystopping = tf.keras.callbacks.\
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            min_delta=0.01,
            verbose=1,
            restore_best_weights=False
        )

    improve_learnin_rate = tf.keras.callbacks. \
        ReduceLROnPlateau(
          monitor="val_accuracy",
          factor=0.1,
          patience=5,
          cooldown=3,
          min_delta=0.001,
          verbose=1
          )
    
    return callback_tf, callbacks_models, callbacks_best_models, earlystopping, improve_learnin_rate
```