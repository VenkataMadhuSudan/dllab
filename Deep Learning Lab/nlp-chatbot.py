import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf, keras, numpy as np, random

intents = {
    "greeting": {
        "patterns": ["hi", "hello", "hey", "good morning", "good evening"],
        "responses": ["Hello!", "Hi there!", "Hey! How can I help?"]
    },
    "goodbye": {
        "patterns": ["bye", "goodbye", "see you", "take care"],
        "responses": ["Goodbye!", "See you later!", "Take care!"]
    },
    "thanks": {
        "patterns": ["thanks", "thank you", "thankyou", "thx"],
        "responses": ["You're welcome!", "Happy to help!", "Anytime!"]
    }
}

tags = list(intents)
X, y = [], []
for i, t in enumerate(tags):
    for p in intents[t]["patterns"]:
        X.append(p)
        y.append(i)

X, y = tf.constant(X), tf.constant(y)

vec = keras.layers.TextVectorization(max_tokens=1000, output_sequence_length=10, standardize="lower")
vec.adapt(X)

model = keras.Sequential([
    vec,
    keras.layers.Embedding(1000, 16),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(len(tags), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X, y, epochs=300, verbose=0)

print("Training complete!")
print("\nChatbot is running! (type 'quit' to exit)")

while True:
    s = input("You: ").lower().strip()
    if s == "quit":
        print("Bot: Goodbye!")
        break
    tag = tags[np.argmax(model.predict(tf.constant([s]), verbose=0))]
    print("Bot:", random.choice(intents[tag]["responses"]))