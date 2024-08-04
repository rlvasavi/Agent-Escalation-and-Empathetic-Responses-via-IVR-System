import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from transformers import pipeline

# Load data from CSV file
file_path = 'customer_support_tickets.csv'
data = pd.read_csv(file_path)

columns_to_keep = [
    'Ticket ID', 'Product Purchased', 'Ticket Subject',
    'Ticket Description', 'Ticket Priority', 'Ticket Status',
    'Resolution', 'Customer Satisfaction Rating'
]
data = data[columns_to_keep]

# Fill missing values
data['Resolution'] = data['Resolution'].fillna('No resolution provided till now.Contact +91 1234567890')
data['Customer Satisfaction Rating'] = data['Customer Satisfaction Rating'].fillna(
    data['Customer Satisfaction Rating'].mean())

# Preprocess text data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)


data['Ticket Subject'] = data['Ticket Subject'].apply(preprocess_text)
data['Ticket Description'] = data['Ticket Description'].apply(preprocess_text)

data['Text'] = data['Ticket Subject'] + ' ' + data['Ticket Description']

# future use
one_hot_encoder = OneHotEncoder()
categorical_columns = ['Product Purchased', 'Ticket Priority', 'Ticket Status']
encoded_features = one_hot_encoder.fit_transform(data[categorical_columns]).toarray()

# Prepare target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['Resolution'])

# Split data into training and validation sets 80 20
X_train, X_val, y_train, y_val = train_test_split(data['Text'], y_encoded, test_size=0.2, random_state=42)

max_words = 10000
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_len)

X_val_sequences = tokenizer.texts_to_sequences(X_val)
X_val_padded = pad_sequences(X_val_sequences, maxlen=max_len)

# Define LSTM model
model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(len(label_encoder.classes_), activation='softmax')  # Number of classes
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(X_train_padded, y_train, epochs=3, validation_data=(X_val_padded, y_val),
                    callbacks=[early_stopping])

# Save model
model.save('ticket_resolution_model.h5')

# Save the tokenizer
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Load sentiment analysis model for emotion capturing
sentiment_analyzer = pipeline('sentiment-analysis')


# Predict function
def predict_resolution_and_sentiment(subject, description):
    text = preprocess_text(subject + ' ' + description)

    # Predict resolution
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    predicted_class = np.argmax(prediction)
    resolution = label_encoder.inverse_transform([predicted_class])[0]

    # Predict sentiment
    sentiment_result = sentiment_analyzer(text)
    sentiment = sentiment_result[0]['label'].lower()

    return resolution, sentiment


# Example usage
result_resolution, result_sentiment = predict_resolution_and_sentiment("Product set up",
                                                                       "I'm having an issue with the product. Please assist")
print(f"Resolution: {result_resolution}, Sentiment: {result_sentiment}")
