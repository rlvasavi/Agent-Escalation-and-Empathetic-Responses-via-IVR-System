from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pickle
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('ticket_resolution_model.h5')

# Load the tokenizer and label encoder
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Set max_len used during training
max_len = 100  # Set this to the same max_len used in training

# Load sentiment analysis model
sentiment_analyzer = pipeline('sentiment-analysis')

# Preprocessing utilities
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[0-9]+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)


def predict_resolution_and_sentiment(subject, description):
    try:
        # Combine and preprocess text
        text = preprocess_text(subject + ' ' + description)

        # Predict resolution
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=max_len)  # Use max_len from training
        prediction = model.predict(padded_sequence)
        predicted_class = np.argmax(prediction)
        confidence_score = np.max(prediction)
        resolution = label_encoder.inverse_transform([predicted_class])[0]

        # Predict sentiment
        sentiment_result = sentiment_analyzer(text)
        sentiment = sentiment_result[0]['label'].lower()  # Get sentiment ('POSITIVE', 'NEGATIVE', 'NEUTRAL')

        return str(resolution), float(confidence_score), sentiment
    except Exception as e:
        print(f"Error in prediction: {e}")  # Consider logging this error
        return "Error in processing the request", 0.0, "unknown"


# Define routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    subject = data.get('subject', '')
    description = data.get('description', '')
    resolution_text, confidence_score, sentiment = predict_resolution_and_sentiment(subject, description)
    return jsonify({
        'resolution': resolution_text,
        'confidence': confidence_score,
        'sentiment': sentiment
    })


# Flask app run statement
if __name__ == '__main__':
    app.run(debug=True)  # Set debug to False in production
