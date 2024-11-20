import paho.mqtt.client as paho
import time
import json
import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

def on_publish(client,userdata,result):             #create function for callback
    print("el dato ha sido publicado \n")
    pass

def on_message(client, userdata, message):
    global message_received
    time.sleep(2)
    message_received=str(message.payload.decode("utf-8"))
    st.write(message_received)

        


broker="broker.emqx.io"
port=1883
client1= paho.Client("Ioav")
client1.on_message = on_message
client1.on_publish = on_publish
client1.connect(broker,port)

model = load_model('keras_model4.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Modo Robot")

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.8:
      st.header('Modo Defensa 🛡️')
      client1.publish("Ioav_Voz","{'mensaje': 'Defensa'}",qos=0, retain=False)
      time.sleep(0.2)
    if prediction[0][1]>0.8:
      st.header('Modo Ataque 🗡️')
      client1.publish("Ioav_Voz","{'mensaje': 'Ataque'}",qos=0, retain=False)
      time.sleep(0.2)  
    if prediction[0][2]>0.8:
      st.header('Secuencia de apagado 😴')
      client1.publish("Ioav_Voz","{'mensaje': 'Apagar'}",qos=0, retain=False)
      time.sleep(0.2)  
