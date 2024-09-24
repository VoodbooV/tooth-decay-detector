import os
import shutil
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
import cv2
import numpy as np
import tensorflow as tf

class ToothDecayApp(App):
    def build(self):
        self.model = tf.keras.models.load_model('tooth_decay_model.h5')  # Load the trained model
        layout = BoxLayout(orientation='vertical')
        self.image = Image(size_hint=(1, 0.8))
        self.label = Label(size_hint=(1, 0.1))
        choose_btn = Button(text="Choose Image", on_press=self.open_filechooser, size_hint=(1, 0.1))
        layout.add_widget(self.image)
        layout.add_widget(self.label)
        layout.add_widget(choose_btn)
        return layout

    def open_filechooser(self, instance):
        content = BoxLayout(orientation='vertical')
        self.filechooser = FileChooserIconView()
        content.add_widget(self.filechooser)
        load_btn = Button(text="Load", size_hint=(1, 0.1), on_press=self.load_image)
        close_btn = Button(text="Close", size_hint=(1, 0.1), on_press=self.close_popup)
        btn_layout = BoxLayout(size_hint=(1, 0.1))
        btn_layout.add_widget(load_btn)
        btn_layout.add_widget(close_btn)
        content.add_widget(btn_layout)
        self.popup = Popup(title="Choose an image", content=content, size_hint=(0.9, 0.9))
        self.popup.open()

    def close_popup(self, instance):
        self.popup.dismiss()

    def load_image(self, instance):
        selection = self.filechooser.selection
        if selection:
            filepath = selection[0]
            temp_filepath = 'temp_image.jpg'
            shutil.copy(filepath, temp_filepath)
            self.popup.dismiss()
            self.display_image(temp_filepath)

    def display_image(self, filepath):
        self.image.source = filepath
        self.image.reload()
        self.predict(filepath)

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (128, 128))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img

    def predict(self, image_path):
        img = self.preprocess_image(image_path)
        prediction = self.model.predict(img)
        if np.argmax(prediction) == 0:
            self.label.text = "No Decay Detected"
        else:
            self.label.text = "Tooth Decay Detected"

if __name__ == '__main__':
    ToothDecayApp().run()
