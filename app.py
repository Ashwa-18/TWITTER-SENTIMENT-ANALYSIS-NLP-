import streamlit as st
import pickle
model = pickle.load(open('trained_model.sav','rb'))
vectorize = pickle.load(open('vectorize.sav','rb'))

st.title("Twitter Sentiment Analysis")
tweet = st.text_area("Enter your Tweet")

if st.button("analyze"):
    tweet_vec = vectorize.transform([tweet])
    prediction = model.predict(tweet_vec)

    if prediction[0] == 0:
        st.error("Negative Tweet 😡")
    else:
        st.success("Positive Tweet 😊")