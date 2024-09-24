#import inline as inline
import matplotlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
#import matplotlib.image as mpimg
# from PIL import Image as pil_image
import image as pil_image
import matplotlib.pyplot as plt
# %matplotlib inline
#get_ipython().run_line_magic('matplotlib', 'inline')
from keras import backend as K
import tensorflow as tf
#from PyQt5 import QtGui
#from PyQt5.QtWidgets import QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit, QDialog
#import sys
#from PyQt5.QtGui import QPixmap
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from tensorflow.keras.preprocessing.image import ImageDataGenerator

image_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range=0.1,
                              height_shift_range=0.1,
                              rescale=1/255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              horizontal_flip=True,
                              fill_mode='nearest')

image_shape=(150, 150, 3)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

model = Sequential()


model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='Adam',
             metrics=['accuracy'])




model.summary()

batch_size=5

train_image_gen=image_gen.flow_from_directory('teeth_dataset/Training',
                                             target_size=image_shape[:2],
                                             batch_size=batch_size,
                                             class_mode='binary')

test_image_gen=image_gen.flow_from_directory('teeth_dataset/test',
                                             target_size=image_shape[:2],
                                             batch_size=batch_size,
                                             class_mode='binary')

train_image_gen.class_indices

import warnings
warnings.filterwarnings('ignore')

results = model.fit(train_image_gen,
                              epochs=30,
                             steps_per_epoch=20,
                             validation_data=test_image_gen,
                             validation_steps=12)


class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("")
        self.minsize(600, 400)
        #self.wm_iconbitmap('icon.ico')
        
        self.labelFrame=ttk.LabelFrame(self, text="browse a image")
        self.labelFrame.grid(column = 0, row = 1, padx = 20, pady = 20)
        self.button()
        
        
    def button(self):
        self.button = ttk.Button(self.labelFrame, text = "browse", command = self.fileDialog)
        self.button.grid(column=1, row=1)
        
        
    def fileDialog(self):
        self.filename = filedialog.askopenfilename(initialdir = "/", title = "select a Image", filetype = (("jpeg", "*.jpg"), ("All Files", "*.*")))
        def classifier(path):
            raw_img = image.load_img(path, target_size=(150, 150))
            raw_img = image.img_to_array(raw_img)
            raw_img = np.expand_dims(raw_img, axis=0)
            raw_img = raw_img / 255
            prediction = model.predict(raw_img)[0][0]
            plt.imshow(cv2.imread(path))
            if prediction:
                print("Healthy tooth")
            else:
                print("cavity infected tooth")
            plt.show()

        classifier(self.filename)

if __name__ == '__main__':
    root = Root()
    root.mainloop()