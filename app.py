import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


def transform_mess(message):
    message = message.lower()
    message = nltk.word_tokenize(message)

    y = []
    for i in message:
        if i.isalnum():
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    message = y[:]
    y.clear()

    for i in message:
        y.append(ps.stem(i))

    return ' '.join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms=st.text_area('Enter the message')

#display
if st.button('Predict',key="predict_button"):

    # preprocess
    transformed_sms = transform_mess(input_sms)
    # vectorize
    vector_sms = tfidf.transform([transformed_sms])
    # predict
    result = model.predict(vector_sms)[0]

    if result == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')



