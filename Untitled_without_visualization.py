

# DataFrame
import pandas as pd

# Matplot
import matplotlib.pyplot as plt
%matplotlib inline

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Word2vec
import gensim

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load the data
df = pd.read_csv("output.csv", sep=";")  # Update with your actual CSV file name

# Preprocess the data
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def preprocess_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

df['processed_text'] = df['review_thoughts'].apply(preprocess_text)

# Split the data into train and test sets
X = df['processed_text'].values
y = df['review_stars'].values

le = LabelEncoder()
y = le.fit_transform(y)
y = y.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and padding
max_features = 10000  # Adjust the maximum number of features as needed
maxlen = 100  # Adjust the maximum sequence length as needed

tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_train = pad_sequences(X_train, maxlen=maxlen)

X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Model training
embedding_dim = 100  # Adjust the embedding dimension as needed

model = Sequential()
model.add(Embedding(max_features, embedding_dim, input_length=maxlen))
model.add(Conv1D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

batch_size = 64  # Adjust the batch size as needed
epochs = 10  # Adjust the number of epochs as needed

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test),
                    callbacks=[ReduceLROnPlateau(), EarlyStopping(patience=3)])

# Define the predict function
def predict(text, include_neutral=True):
    start_at = time.time()
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxlen)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = "POSITIVE" if score > 0.5 else "NEGATIVE"

    return {"label": label, "score": float(score),
            "elapsed_time": time.time() - start_at}

# Example usage of the predict function
result = predict("I love the product")
res2 = predict("bad")
print(result)
print(res2)



import os
import glob
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Define a function to load and preprocess each dataset
def load_and_preprocess(filename):
    # Load the data
    data = pd.read_csv(filename)

    # Rename columns
    data = data.rename(columns={'review_text': 'text', 'rating': 'sentiment'})

    # Replace NaNs with empty strings
    data['text'] = data['text'].fillna('')

    # Preprocess the text data
    data['text'] = data['text'].apply(lambda x: x.lower())

    # Convert ratings into sentiment
    data['sentiment'] = data['sentiment'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

    return data


# Load all datasets
#dataframes = [load_and_preprocess(filename) for filename in glob.glob('datasets/sephora/*.csv')]

csv_files = glob.glob('datasets/sephora/*.csv')

# Initialize an empty list to store the dataframes
dataframes = []

# Iterate over the CSV files
for filename in csv_files:
    # Load and preprocess the data
    dataframe = load_and_preprocess(filename)
    
    # Append the dataframe to the list
    dataframes.append(dataframe)

# Combine all the dataframes
data = pd.concat(dataframes, ignore_index=True)

# Tokenization
tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(data['text'].values)

# Convert text to sequence of tokens and pad them to ensure equal length
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X, maxlen=100)


# Initialize the label encoder
le = LabelEncoder()

# Fit and transform the sentiment values to integer
data['sentiment'] = le.fit_transform(data['sentiment'])

# Convert labels to categorical
y = to_categorical(data['sentiment'].values)

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the LSTM model
model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))  # Reduced embedding dimension
model.add(SpatialDropout1D(0.2)) 
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))  # Removed one LSTM layer
model.add(Dense(y.shape[1], activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with batch_size=32 (default) for faster training
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))  # Reduced number of epochs

# Save the model
os.makedirs('model', exist_ok=True)
model.save('model/sentiment_analysis_model.h5')

import time

# Define the predict function
def predict(text, include_neutral=True):
    # Set maxlen to the value used in training
    maxlen = 50  # Or whatever value you used in training

    start_at = time.time()

    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=maxlen)
    
    # Predict
    score = model.predict([x_test])[0]

    # Decode sentiment
    if include_neutral:        
        label_idx = np.argmax(score)
        label = ["NEGATIVE", "NEUTRAL", "POSITIVE"][label_idx]
    else:
        label = "NEGATIVE" if score[0] > 0.5 else "POSITIVE"

    return {"label": label, "score": float(score[label_idx]),
       "elapsed_time": time.time()-start_at}  


# Example usage of the predict function
result = predict("I love the product")
res2 = predict("bad")
res3 = predict("I don't like it very much, it makes my skin itch and burns, i dont recommend it")
res4 = predict("Maybe it's good, i haven't test it yet")
res5 = predict("it works for me")
res6 = predict("doesnt work for me")
print(result)
print(res2)
print(res3)
print(res4)
print(res5)
print(res6)





# Load new data
new_data = pd.read_csv("./exchange/input/input.csv", sep=";")

new_data.head()


# Predict sentiment for new data
new_data['predicted_sentiment'] = new_data['review_thoughts'].apply(lambda x: predict(x))

# Categorize reviews based on sentiment score
new_data['sentiment_category'] = new_data['predicted_sentiment'].apply(lambda x: x['label'])

new_data.head()





import csv

# Save the dataframe with new sentiment information
new_data.to_csv("./exchange/output/output_with_sentiments.csv", sep=';', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

# Count the sentiment categories
sentiment_counts = new_data['sentiment_category'].value_counts()

print(sentiment_counts)

new_data[new_data['sentiment_category'] == 'NEUTRAL']



from nltk.corpus import stopwords
from rake_nltk import Rake

nltk.download('punkt')


# NLTK Stop word list
stopwords = set(stopwords.words('english'))

def extract_keywords(text):
    r = Rake(stopwords=stopwords)  # Uses stopwords for english from NLTK, and all puntuation characters.
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()  # To get keyword phrases ranked highest to lowest.


new_data['keywords'] = new_data['review_thoughts'].apply(extract_keywords)

new_data.head()

new_data.shape()





# Save the dataframe with new sentiment information
new_data.to_csv("./exchange/output/output_with_sentiments_keywords.csv", sep=';', index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)

# Display all positive reviews with a 1-star rating
positive_1_star_reviews = new_data[(new_data['review_stars'] == 1) & (new_data['sentiment_category'] == 'POSITIVE')]

# Print the review text for these reviews
for i, row in positive_1_star_reviews.iterrows():
    print(f'Review {i}:\n{row["review_thoughts"]}\n')


import matplotlib.pyplot as plt
import seaborn as sns



from tensorflow.keras.models import load_model

# Load the model
#model = load_model('models/sentiment_analysis_model.h5')











