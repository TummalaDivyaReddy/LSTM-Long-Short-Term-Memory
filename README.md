# LSTM

Source Code: [LSTM](LSTM.ipynb)

LSTM (Long Short-Term Memory) is a type of recurrent neural network (RNN) that is specifically designed to handle long-term dependencies. RNNs are a type of neural network that can process sequential data, such as text or speech. However, RNNs can have difficulty learning long-term dependencies, which are important for tasks such as machine translation and sentiment analysis.

LSTMs address this problem by using a gating mechanism that allows them to selectively remember or forget information from previous time steps. This allows LSTMs to learn long-term dependencies and perform tasks that would be difficult or impossible for traditional RNNs

### Types of RNN

There are two main types of RNNs:
 Vanilla RNNs 
 LSTMs.
 
Vanilla RNNs are the simplest type of RNN and are often used for tasks such as text classification and speech recognition. However, vanilla RNNs can have difficulty learning long-term dependencies, which is why LSTMs were developed.

LSTMs are more complex than vanilla RNNs, but they are also more powerful. LSTMs are often used for tasks such as machine translation, sentiment analysis, and natural language generation.

In this project we are using Sentiment analysis to determine the emotional sentiment of a piece of text. This can be done for a variety of purposes, such as customer feedback analysis, social media monitoring, and political polling.

One way to perform sentiment analysis is to use an LSTM. LSTMs can be trained on a dataset of labeled tweets, and then used to predict the sentiment of new tweets. This can be done by feeding the LSTM the text of a tweet, and then outputting a probability that the tweet is positive, negative, or neutral.

For example, an LSTM could be trained on a dataset of tweets that have been labeled as either positive or negative. The LSTM would learn to identify the features of a tweet that are associated with each sentiment. Once the LSTM is trained, it can be used to predict the sentiment of new tweets.

##  Save the model and use the saved model to predict on new text data (ex, “A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump”) and Apply GridSearchCV on the source code provided in the class

```ruby
from google.colab import files
 
uploaded = files.upload()
```
the above line is used to load our CSV file


```ruby
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import re
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
# Data Preprocessing
data = pd.read_csv('Sentiment.csv')
data = data[['text', 'sentiment']]
data = data[data.sentiment != 'Neutral']

data['text'] = data['text'].apply(lambda x: x.lower())
data['text'] = data['text'].apply(lambda x: re.sub('[^a-zA-z0-9\s]', '', x))

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

# LSTM Model Building
def create_model():
    model = Sequential()
    model.add(Embedding(max_fatures, embed_dim, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.4))
    model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Wrap the Keras model inside a KerasClassifier from Sci-Keras
model = KerasClassifier(build_fn=create_model, verbose=0)

# Data Splitting
labelencoder = LabelEncoder()
integer_encoded = labelencoder.fit_transform(data['sentiment'])
y = to_categorical(integer_encoded)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# GridSearchCV with Sci-Keras
param_grid = {
    'batch_size': [32, 64],
    'epochs': [5, 10],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid_search.fit(X_train, Y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Training the model with best hyperparameters
best_batch_size = grid_result.best_params_['batch_size']
best_epochs = grid_result.best_params_['epochs']

model = create_model()  # Create a new instance of the model with best hyperparameters
model.fit(X_train, Y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2)

# Evaluation
score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=best_batch_size)
print("Test accuracy: %.2f%%" % (acc * 100))

# Save the model
model.save('model.h5')

# Load the model and predict on new text data
model = load_model('model.h5')
tweet = ['A lot of good things are happening. We are respected again throughout the world, and that''s a great thing.@realDonaldTrump']
tweet = tokenizer.texts_to_sequences(tweet)
tweet = pad_sequences(tweet, maxlen=X.shape[1], value=0)
print(tweet)
sentiment = model.predict(tweet, batch_size=1, verbose=1)[0]
print(sentiment)
if np.argmax(sentiment) == 0:
    print("Negative")
elif np.argmax(sentiment) == 1:
    print("Positive")

```

The first few lines of code import the necessary libraries, including NumPy, Pandas, Keras, and scikit-learn and then we load the data from the Sentiment.csv file. 
The data is a CSV file with two columns: text and sentiment. The text column contains the text of a tweet, and the sentiment column contains the sentiment of the tweet, which can be either positive, negative, or neutral.
We perform some data preprocessing. The text column is converted to lowercase, and all non-alphabetical characters are removed. The rt abbreviation is also replaced with a space.
And then we create a tokenizer, which is a tool that can be used to convert text into a sequence of integers. The tokenizer is trained on the text column of the data, and it is used to convert the text into a sequence of integers. The sequence of integers represents the words in the text, and each word is assigned a unique integer.
Then create the LSTM model. The LSTM model is a type of recurrent neural network that is well-suited for tasks such as sentiment analysis. The LSTM model has three layers: an embedding layer, an LSTM layer, and a dense layer. The embedding layer converts the sequence of integers into a sequence of vectors. The LSTM layer learns to identify the features of the text that are associated with each sentiment. The dense layer outputs a probability that the tweet is positive, negative, or neutral.
Here we use GridSearchCV to find the best hyperparameters for the LSTM model. GridSearchCV is a tool that can be used to find the best combination of hyperparameters for a machine learning model. The hyperparameters that are tuned are the batch size and the number of epochs and then we train the LSTM model with the best hyperparameters. The model is trained for 10 epochs with a batch size of 64.
Then It evaluate the LSTM model. The model is evaluated on the test set, and the accuracy is reported and then saved in the LSTM model. The model is saved to a file named model.h5.
It loads the LSTM model and predicts the sentiment of a new tweet. The new tweet is "A lot of good things are happening. We are respected again throughout the world, and that's a great thing.@realDonaldTrump". The LSTM model predicts that the sentiment of the tweet is positive.


Youtube video: [LSTM](https://youtu.be/5nP9uxAB-wM)

