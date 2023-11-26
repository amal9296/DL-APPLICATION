
import streamlit as st
from functions import Basic_functions
from model_define import BackPropogation, Perceptron



def main():

    st.title("ML MODEL")     

    option = st.selectbox("Why are you here?",('Tumor prediction','Sentiment analysis'),index=None,placeholder="Select problem method...",)

    st.write('You selected:', option)

    if option == None:
        pass

    elif option == 'Tumor prediction':

        
        st.write('Upload tumer data')
        model_name = 'CNN_tumor.pkl'
        
        input_data = Basic_functions.upload_image()
        if input_data is not None:
            st.image(input_data, channels="BGR")

        but = st.button("Predict", type="primary")    
        if but:
            model = Basic_functions.open_model('CNN_tumor.pkl')
            Basic_functions.pred(input_data,model,model_name)       
        
    elif option == 'Sentiment analysis':
        out =  st.radio(
            "Select your prediction 👉",
            key="visibility",
            options=["Recurrent Neural Network", "LSTM", "DEEP NEURAL NETWORK", "Back propagation", "Perceptron"],)  

    

        if out == "Recurrent Neural Network":
            model_name = 'RNN_smsspam1.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)


        elif out == "LSTM":
            model_name = 'LSTM_custom1.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name) 

        elif out == "DEEP NEURAL NETWORK": 
            model_name = 'DNN_spam_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)


        elif out == "Back propagation":
            model_name = 'Backprop_spam_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)




        elif out == "Perceptron": 
            model_name = 'perceptron_spam_model.pkl'
            input_data = st.text_input("Text to analyze",)
            but = st.button("Predict", type="primary") 

            if but:
                model = Basic_functions.open_model(model_name)
                Basic_functions.pred(input_data,model,model_name)



if __name__ == "__main__":
    main()           


