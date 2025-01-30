import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
import plotly.graph_objs as go

# Streamlit configuration
st.set_page_config(page_title="Movie Review Sentiment Analysis", page_icon="🎬", layout="wide")

# Display TensorFlow version
st.write(f"TensorFlow version: {tf.__version__}")

# Load the IMDB dataset word index
@st.cache_data
def load_word_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = load_word_index()

# Load model
@st.cache_resource
def load_sentiment_model():
    model = load_model('simple_rnn_imdb.h5')
    return model

model = load_sentiment_model()

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [min(word_index.get(word, 2), 9999) for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        padding-bottom: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        padding-bottom: 20px;
    }
    .result-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>🎬 IMDB Movie Review Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Enter a movie review to classify it as positive or negative.</p>", unsafe_allow_html=True)

# User input
user_input = st.text_area("Movie Review", height=150)

if st.button('Analyze'):
    if user_input and model is not None:
        with st.spinner('Analyzing your review...'):
            preprocessed_input = preprocess_text(user_input)
            try:
                prediction = model.predict(preprocessed_input)
                sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
                
                # Display the result
                st.markdown("<h2 class='sub-header'>Analysis Result</h2>", unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<p class='result-text'>Sentiment: {sentiment}</p>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<p class='result-text'>Confidence: {prediction[0][0]:.2%}</p>", unsafe_allow_html=True)
                
                # Visualization
                st.markdown("<h3>Sentiment Distribution</h3>", unsafe_allow_html=True)
                fig = go.Figure(data=[go.Pie(labels=['Positive', 'Negative'], values=[prediction[0][0], 1 - prediction[0][0]], marker=dict(colors=["#66BB6A", "#EF5350"]))])
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    elif model is None:
        st.error("Model weights could not be loaded. Please check your model file and TensorFlow version.")
    else:
        st.warning('Please enter a movie review.')

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit and TensorFlow")