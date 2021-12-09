import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'model.h5')


model = load_model('model.h5')


st.title('My Digit Recognizer')
st.markdown('''
Try to write a digit!
''')



SIZE = 192

canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw",
    key='canvas')

if canvas_result.image_data is not None:
    img = canvas_result.image_data
    
    image = Image.fromarray((img[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    test_x = tf.convert_to_tensor(image)
    
    
    
    st.image(image)

if st.button('Predict'):
    val = model.predict(test_x)
    st.write(f'Result: {np.argmax(val[0])}')