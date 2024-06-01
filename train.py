# -*- coding: utf-8 -*-
import glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = 64
CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

path = "./Data/"


def plot_training_analysis():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', linestyle="--", label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'b', linestyle="--", label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def load_data(new_path, classes, image_size=64):
    # Liste les fichiers présents dans le dossier path
    file_path = glob.glob(new_path)

    # Initialise les structures de données
    x = np.zeros((350, image_size, image_size, 3))
    y = np.zeros((350, 1))

    for j in range(len(file_path)):
        file_path_emotion = glob.glob(file_path[j]+'/*')
        for i in range(len(file_path_emotion)):
            # Lecture de l'image
            img = Image.open(file_path_emotion[i])
            # Mise à l'échelle de l'image
            img = img.resize((image_size, image_size), Image.LANCZOS)

            # Remplissage de la variable x
            x[i+(j-1)*50] = np.asarray(img)

            import re
            img_path_split = re.split(r'[\\/]', file_path[j])
            img_name_split = img_path_split[-1]

            class_label = classes.index(img_name_split)
            y[i+(j-1)*50] = class_label

    return x, y


x_train, y_train = load_data('./Data/*', CLASSES, image_size=IMAGE_SIZE)
x_val, y_val = load_data('./Data/*', CLASSES, image_size=IMAGE_SIZE)
# Normalisation des entrées via une division par 255 des valeurs de pixel.
x_train = x_train/255
x_val = x_val/255

y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_val = to_categorical(y_val, num_classes=len(CLASSES))
# Première approche : réseau convolutif de base

# Correction du sur apprentissage

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Modèle 

model = Sequential()

model.add(Conv2D(32, 3, activation="relu", input_shape=(64, 64, 3), padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(96, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, 3, activation="relu", padding="same"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())    # "Mise à plat" (vectorisation) du tenseur pour permettre de la connecter à une couche dense
model.add(Dense(512, activation="relu"))   # Couche dense, à 512 neurones
model.add(Dense(len(CLASSES), activation="softmax"))   # Couche de sortie

model.summary()

model.compile(loss="categorical_crossentropy",
              optimizer=optimizers.Adam(learning_rate=3e-4),
              metrics=['accuracy'])

history = model.fit(train_datagen.flow(x_train, y_train, batch_size=10),
                    validation_data=train_datagen.flow(x_val, y_val, batch_size=10),
                    epochs=200,
                    )

# Analyse des résultats

plot_training_analysis()
model.save('model.h5')
