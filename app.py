import re
import joblib
import nltk
import streamlit as st
import pandas as pd
from nltk.corpus import stopwords

st.title('Nobel Prize Motivation Text Classification')
st.header('By Carlos Enrique Aguilar Maza Test Module 4 ')
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def limpiar_motivacion(texto):
  texto = texto.lower()
  texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
  palabras = texto.split()
  palabras = [palabra for palabra in palabras if palabra not in stop_words]
  return ' '.join(palabras)

def user_input_features():
  texto = st.text_area('Input your Validation Text:')
  user_input_data = {'Text': texto}
  features = pd.DataFrame(user_input_data, index=[0])
  return features

df = user_input_features()
vect = joblib.load('vectorizer.pkl')
nb = joblib.load('model.pkl')

if st.button('Predict'):
  df['Text'] = df['Text'].apply(limpiar_motivacion)
  df_dtm = vect.transform(df['Text'])
  prediction = nb.predict(df_dtm)
  st.subheader('Prediction')
  st.success('Estimated Category: ' + str(prediction[0]))
