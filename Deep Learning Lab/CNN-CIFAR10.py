import tensorflow as tf
import matplotlib.pyplot as plt

(xtr,ytr),(xte,yte)=tf.keras.datasets.cifar10.load_data()
xtr,xte=xtr/255.0,xte/255.0

names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([]); plt.yticks([]); plt.grid(False)
    plt.imshow(xtr[i])
    plt.xlabel(names[ytr[i][0]])
plt.show()

model=tf.keras.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(10)
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

h=model.fit(xtr,ytr,epochs=10,validation_data=(xte,yte))

plt.plot(h.history['accuracy'],label='Train Accuracy')
plt.plot(h.history['val_accuracy'],label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

test_loss,test_acc=model.evaluate(xte,yte,verbose=2)
print("Test accuracy:",test_acc)