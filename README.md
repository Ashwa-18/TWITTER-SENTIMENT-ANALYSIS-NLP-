# TWITTER-SENTIMENT-ANALYSIS-NLP-
# Twitter Sentiment Analysis

This project performs sentiment analysis on Twitter data using Machine Learning and NLP techniques.

## Project Overview
- Tweets are preprocessed and converted into numerical features using TF-IDF Vectorizer.
- A machine learning classification model is trained to classify tweets as Positive or Negative.
- The trained model and vectorizer are saved using Pickle.
- The saved model is reused for sentiment prediction using a Streamlit application.

## Files in this Repository
- twitter_sentiment_analysis.ipynb : Model training and saving
- app.py : Streamlit app for sentiment prediction
- trained_model.sav : Saved trained model
- vectorize.sav : Saved TF-IDF vectorizer
- requirements.txt : Required Python libraries

