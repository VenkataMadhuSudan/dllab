import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf, keras

vocab_size, max_length = 10000, 200

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
x_train = keras.utils.pad_sequences(x_train, maxlen=max_length)
x_test = keras.utils.pad_sequences(x_test, maxlen=max_length)

print("Training samples:", len(x_train))
print("Test samples:", len(x_test))

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 128),
    keras.layers.LSTM(64),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, acc = model.evaluate(x_test, y_test)
print("LSTM Test Accuracy:", acc)