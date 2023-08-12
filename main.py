import numpy as np
import pickle
import streamlit as st








#loading the saved model
loaded_model = pickle.load(open('C:/Users/Masterkim/projects/iris/trained_model.sav', 'rb'))


#create a function for prediction

def iris_pred(input_data):
    #changing the input data to numpy array

    input_array=np.asarray(input_data)

    #reshape

    input_array_reshape = input_array.reshape(1,-1)
    prediction= loaded_model.predict(input_array_reshape)
    print(prediction)

    if (prediction[0] == 0):
        print('The flower is iris setosa')
    elif(prediction[0] == 0) :
        print('The flower is iris versicolor')
    else:
        print('the flower is iris virginica')
        

def main():

    #giving a title
    st.title("Iris flower classifier web app")

    #getting input data from users

    SepalLengthCm = st.text_input('Sepal length')
    SepalWidthCm = st.text_input('Sepal width')
    PetalLengthCm  = st.text_input('Petal length')
    PetalWidthCm  = st.text_input('Petal width')



    #code for prediction
    diagnosis = ' '
    
    #creating a button for prediction

    if st.button('Check result'):
        diagnosis = iris_pred([SepalLengthCm, SepalWidthCm,  PetalLengthCm , PetalWidthCm ])

    st.success(diagnosis)







if __name__ == "__main__":
    main()



















