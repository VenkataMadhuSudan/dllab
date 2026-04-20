import tensorflow as tf
import matplotlib.pyplot as plt

# Load data
(xtr,_),(xte,_)=tf.keras.datasets.fashion_mnist.load_data()
xtr,xte=xtr/255.0,xte/255.0

# Autoencoder model
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(28*28,activation='sigmoid'),
    tf.keras.layers.Reshape((28,28))
])

# Compile & Train
model.compile(optimizer='adam',loss='mse')
model.fit(xtr,xtr,epochs=10,validation_data=(xte,xte))

# Reconstruct
dec=model.predict(xte)

# Show results
plt.figure(figsize=(20,4))
for i in range(10):
    plt.subplot(2,10,i+1); plt.imshow(xte[i],cmap='gray'); plt.axis('off')
    plt.subplot(2,10,i+11); plt.imshow(dec[i],cmap='gray'); plt.axis('off')
plt.show()