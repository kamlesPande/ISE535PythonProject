# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 01:34:56 2021

@author: kamlesh
"""
import cv2
import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import streamlit
import matplotlib.pyplot as plt


def image_preprocess(image):
  #img = cv2.imread(imgPath)
  #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #plt.imshow(img)
  #plt.axis("on")
  return cv2.resize(image/255.0, (224,224)).reshape(-1,224,224,3)


root = os.getcwd()
modelPath = os.path.join(root, "modelMobile_2.h5")

model = tf.keras.models.load_model("modelMobile_2.h5")

uploaded_file = st.file_uploader("Please Choose am image file to classify", type="jpg")

className = ["Healthy", "Multiple Disease", "Rust", "Scab"]

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224)).reshape(-1, 224, 224, 3)
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")
    
    #resized = image_preprocess(resized)
    #img_reshape = resized[np.newaxis,...]
    st.write("The shape of image is {}".format(resized.shape))
    st.write(model.summary())
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        prediction = model.predict(resized)
        st.title("Predicted Label for the image is {}".format(className[np.argmax(prediction)]))
