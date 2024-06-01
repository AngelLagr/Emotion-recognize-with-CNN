import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras 
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator , img_to_array, load_img
from keras.applications.mobilenet import MobileNet, preprocess_input 
from keras.losses import categorical_crossentropy


from keras.models import load_model

def Recognize():
    model = load_model("model.h5")
    
    train_datagen = ImageDataGenerator(
         zoom_range = 0.2, 
         shear_range = 0.2, 
         horizontal_flip=True, 
         rescale = 1./255
    )
    
    train_data = train_datagen.flow_from_directory(directory= "train/", 
                                                   target_size=(224,224), 
                                                   batch_size=32,
                                      )
    # path for the image to see if it predics correct class
    
    path = "train/sad/download (1).jpg"
    img = load_img(path, target_size=(224,224) )
    
    i = img_to_array(img)/255
    input_arr = np.array([i])
    input_arr.shape
    
    pred = np.argmax(model.predict(input_arr))
    
    
    # just to map o/p values
    
    
    op = dict(zip( train_data.class_indices.values(), train_data.class_indices.keys()))
    print(f" the image is of {op[pred]}")
    
    # to display the image  
    plt.imshow(input_arr[0])
    plt.title("input image")
    plt.show()