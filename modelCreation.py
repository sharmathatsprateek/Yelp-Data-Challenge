import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Conv1D, MaxPooling1D, Dropout
import pickle
seed = 7
np.random.seed(seed)


#load the users and text
with open("keras_text.pickle", "rb") as f:
    texts_loaded= pickle.load(f)

with open("keras_user.pickle", "rb") as f:
    users_str_loaded= pickle.load(f)


texts=texts_loaded[:20000] ## training for first 30000 inputs only
users_str=users_str_loaded[:20000] ## training for first 30000 inputs only


# encode class values as integers
encoder = LabelEncoder()
encoder.fit(users_str)
encoded_users = encoder.transform(users_str)
# convert integers to dummy variables (i.e. one hot encoded)
users = np_utils.to_categorical(encoded_users)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=300)

print '----NN input is ready---'


model = Sequential()
model.add(Embedding(20000, 128, input_length=300))
model.add(Dropout(0.2))
model.add(Conv1D(64, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128))
model.add(Dense(len(users), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(data, np.array(users), validation_split=0.5, epochs=3)


# save the tokenizer,users,encoded_users(in integers),encoder(so that it can be used to tranform back the integer to user) and model
with open("keras_tokenizer.pickle", "wb") as f:
   pickle.dump(tokenizer, f)

with open("keras_users_encode.pickle", "wb") as f:
   pickle.dump(encoded_users, f)

with open("keras_encoder.pickle", "wb") as f:
    pickle.dump(encoder, f)

model.save("yelp_recomender_model.hdf5")
