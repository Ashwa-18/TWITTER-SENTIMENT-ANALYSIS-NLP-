import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load model & vectorizer
model = pickle.load(open('trained_model.sav', 'rb'))
vectorizer = pickle.load(open('vectorize.sav', 'rb'))

st.title("Twitter Sentiment Analysis App")
st.write("Upload a CSV file containing tweets and get sentiment counts")

# File upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(
        uploaded_file,
        encoding='latin1',
        header=None,
        names=['target', 'ids', 'date', 'flag', 'user', 'text']
    )

    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    if 'text' not in df.columns:
        st.error("CSV file must contain a column named 'text'")
    else:
        tweets = df['text'].astype(str).tolist()

        # Loading spinner
        with st.spinner("Analyzing tweets... Please wait ⏳"):
            tweets_vec = vectorizer.transform(tweets)
            predictions = model.predict(tweets_vec)

        # Count sentiments
        positive_count = np.sum(predictions == 1)
        negative_count = np.sum(predictions == 0)

        # Percentages
        total = len(predictions)
        positive_percent = (positive_count / total) * 100
        negative_percent = (negative_count / total) * 100

        st.subheader("Sentiment Results")

        st.success(f"Positive Tweets 😊 : {positive_count}")
        st.error(f"Negative Tweets 😡 : {negative_count}")

        st.write(f"Positive Percentage: {positive_percent:.2f}%")
        st.write(f"Negative Percentage: {negative_percent:.2f}%")

        # Bar chart
        chart_data = pd.DataFrame({
            'Sentiment': ['Positive', 'Negative'],
            'Count': [positive_count, negative_count]
        })

        st.subheader("Sentiment Visualization")
        st.bar_chart(chart_data.set_index('Sentiment'))

        # Add predictions to dataframe
        df['sentiment'] = predictions
        df['sentiment_label'] = df['sentiment'].map({
            1: 'Positive',
            0: 'Negative'
        })

        # Download result file
        st.download_button(
            label="Download file with predictions",
            data=df.to_csv(index=False),
            file_name="tweets_with_sentiment.csv",
            mime="text/csv"
        )