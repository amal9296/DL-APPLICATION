import streamlit as st 
import pickle 
from PIL import Image 
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import argmax

class Basic_functions:
    def upload_image():
        input_data = st.file_uploader("Upload your file here...", type=['png', 'jpeg', 'jpg'])
        if input_data is not None:
            file_bytes = np.asarray(bytearray(input_data.read()), dtype=np.uint8)
            opencv_image = cv2.imdecode(file_bytes, 1)
            return opencv_image


    def open_model(model_name):
        load_model = open(model_name, 'rb') 
        model = pickle.load(load_model)
        return model
    


    def pred(input_data,model,model_name):

        if model_name == 'CNN_tumor.pkl':

            img=Image.fromarray(input_data)
            img=img.resize((128,128))
            img=np.array(img)
            input_img = np.expand_dims(img, axis=0)
            res = model.predict(input_img)
            if res:
                st.write("Tumor Detected")
            else:
                st.write("No Tumor")




        if model_name == 'RNN_smsspam1.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences(input_data)
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if preds[0] == [0]:
                st.write('The given message is ham')

            elif preds[0]== [1]:
                st.write('The given message is spam')


        if model_name == 'LSTM_custom1.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 50
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if preds[0] == [0]:
                st.write('The given message is ham')

            elif preds[0]== [1]:
                st.write('The given message is spam') 



        if model_name == 'DNN_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = (model.predict(input_data) > 0.5).astype("int32")
            if argmax(preds)==0:
                st.write('The given message is ham')

            elif argmax(preds)==1: 
                st.write('The given message is spam')   



        if model_name == 'Backprop_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = model.predict([input_data])
            if preds==[0]:
                st.write('The given message is ham')

            elif preds==[1]: 
                st.write('The given message is spam')



        if model_name == 'perceptron_spam_model.pkl':
            st.write('------------')
            load_model = open('RNN_smsspam_tokeniser.pkl', 'rb') 
            tokeniser = pickle.load(load_model)
            max_length = 10
            encoded_test = tokeniser.texts_to_sequences([input_data])
            input_data = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
            preds = model.predict([input_data])
            if preds==[0]:
                st.write('The given message is ham')

            elif preds==[1]: 
                st.write('The given message is spam') 