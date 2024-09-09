import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
model2 = load_model('keras_model2.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
image = Image.open('OIG5.jpg')
st.image(image, width=350)

with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usarlo en esta app para identificar")


opcion = st.selectbox("Modo:", ("Detecta Cara", "Detecta Gesto"))


img_file_buffer = st.camera_input("Toma una Foto")

# Selector de mo
# Lógica para "Detecta Cara"
if opcion == "Detecta Cara":
    if img_file_buffer is not None:
        # Leer la imagen desde el buffer
        img = Image.open(img_file_buffer)
        
        # Redimensionar la imagen a 224x224 (como el tamaño esperado por el modelo)
        newsize = (224, 224)
        img = img.resize(newsize)
        
        # Convertir la imagen en un array de numpy
        img_array = np.array(img)
        
        # Normalizar la imagen
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Realizar la inferencia con el modelo de detección de caras
        prediction = model.predict(data)

        # Mostrar los resultados de la predicción
        if prediction[0][0] > 0.5:
            st.header(f'Andrea, con probabilidad: {prediction[0][0]:.2f}')
        if prediction[0][1] > 0.5:
            st.header(f'Ioav, con probabilidad: {prediction[0][1]:.2f}')

# Lógica para "Detecta Gesto"
if opcion == "Detecta Gesto":
    if img_file_buffer is not None:
        # Leer la imagen desde el buffer
        img = Image.open(img_file_buffer)
        
        # Redimensionar la imagen a 224x224
        newsize = (224, 224)
        img = img.resize(newsize)
        
        # Convertir la imagen en un array de numpy
        img_array = np.array(img)
        
        # Normalizar la imagen
        normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array

        # Realizar la inferencia con el modelo de detección de gestos
        prediction = model2.predict(data)

        # Mostrar los resultados de la predicción
        if prediction[0][0] > 0.5:
            st.header(f'Mano Abierta, con probabilidad: {prediction[0][0]:.2f}')
        if prediction[0][1] > 0.5:
            st.header(f'Mano cerrada, con probabilidad: {prediction[0][1]:.2f}')
