import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform
import paho.mqtt.client as paho
import json

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("interfaz.css")
# Configuración del cliente MQTT
def on_publish(client, userdata, result):
    print("El dato ha sido publicado \n")
    pass

def on_message(client, userdata, message):
    st.write("Mensaje recibido:", str(message.payload.decode("utf-8")))

broker = "broker.mqttdashboard.com"  # Dirección del broker MQTT
port = 1883
client = paho.Client("Ioav_Mizra")
client.on_publish = on_publish
client.on_message = on_message
client.connect(broker, port)

# Mostrar versión de Python y detalles adicionales
#st.write("Versión de Python:", platform.python_version())

# Carga de modelos
model2 = load_model('keras_model3.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Modo del Robot")


# Barra lateral con subheader
with st.sidebar:
    st.text('El robot está listo para lo que decidas. Puedes activar su modo de ataque, donde reaccionará con rapidez y fuerza, o ponerlo en modo tranquilo, en el que observará el entorno sin intervenir. Tú tienes el control: elige el modo según lo que necesites.')
# Entrada de imagen desde la cámara
img_file_buffer = st.camera_input("Toma una Foto")

# Función para publicar mensajes MQTT
def publicar_mensaje(topico, mensaje):
    mensaje_json = json.dumps({"mensaje": mensaje})
    client.publish(topico, mensaje_json)

# Lógica para "Detecta Gesto"
if img_file_buffer is not None:
    # Procesar la imagen de entrada
    img = Image.open(img_file_buffer).resize((224, 224))
    img_array = np.array(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar la inferencia con el modelo de detección de gestos
    prediction = model2.predict(data)

    # Mostrar y enviar los resultados de la predicción
    if prediction[0][0] > 0.5:
        mensaje = "Mpdo Defensa"
        st.header(mensaje)
        publicar_mensaje("Ioav_Voz", mensaje)
    elif prediction[0][1] > 0.5:
        mensaje = "Modo Ataque"
        st.header(mensaje)
        publicar_mensaje("Ioav_Voz", mensaje)
