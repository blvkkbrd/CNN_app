import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
import streamlit as st
from streamlit_drawable_canvas import st_canvas

MODEL_DIR = os.path.join(os.path.dirname("C:/Users/utilisateur/Desktop/JupyterNotebook/00_CNN"), 'model.h5')
#if not os.path.isdir(MODEL_DIR):
#    os.system('CNN.ipynb')

model = tf.keras.models.load_model("model.h5")
st.markdown('<style>body{color: red; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.title('Your Number')
st.markdown('''
Write the number you want!
''')

data = np.random.rand(28,28)
img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)
# import base64

# @st.cache(allow_output_mutation=True)
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     body {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return


SIZE = 192
mode = st.checkbox("Enregistrer (ou Supprimer)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    X_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(X_test.reshape(1, 28, 28))
    st.write(f'Prediction: {np.argmax(val[0])}')
if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

#if st.button('Predict'):
    #test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #val = model.predict(X_test.reshape(1, 28, 28))
    #st.write(f'result: {np.argmax(val[0])}')
    #st.bar_chart(val[0])