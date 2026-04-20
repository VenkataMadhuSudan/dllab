import tensorflow as tf, seaborn as sns, numpy as np , matplotlib.pyplot as plt

print("TF version:",tf.__version__)
(xtr,ytr),(xte,yte)=tf.keras.datasets.mnist.load_data()

sns.countplot(x=ytr);plt.title("Label Distribution");plt.show()
print("Nans:",np.isnan(xtr).any(),np.isnan(xte).any())

xtr, xte = xtr[...,None]/255., xte[...,None]/255.
ytr, yte = tf.one_hot(ytr,10), tf.one_hot(yte,10)

plt.imshow(xtr[100,:,:,0],cmap='gray');plt.title(f"Label:{np.argmax(ytr[100])}");plt.show()

model=tf.keras.Sequential([
    tf.keras.Input((28,28,1)),
    tf.keras.layers.Conv2D(32,5,padding="same",activation='relu'),
    tf.keras.layers.Conv2D(32,5,padding="same",activation='relu'),
    tf.keras.layers.MaxPooling2D(),tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64,3,padding="same",activation='relu'),
    tf.keras.layers.Conv2D(64,3,padding="same",activation='relu'),
    tf.keras.layers.MaxPooling2D(),tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.compile('rmsprop','categorical_crossentropy',metrics=['accuracy'])

class CB(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs=None):
        if logs['accuracy']>0.995:
            print("accuracy reached 99.5% - stopping training")
            self.model.stop_training=True

h=model.fit(xtr,ytr,batch_size=64,epochs=5,validation_split=0.1,callbacks=[CB()])

fig,ax = plt.subplots(2,1,figsize=(8,8))
for a,t,v,l1,l2 in [(ax[0],'loss','val_loss','Train loss','Val loss'),(ax[1],'accuracy','val_accuracy','Train acc','Val acc')]:
    a.plot(h.history[t],label=l1);a.plot(h.history[v],label=l2);a.legend()
plt.show()

test_loss, test_acc = model.evaluate(xte,yte)
y_pred=np.argmax(model.predict(xte),1)
y_true=np.argmax(yte,1)
cm=tf.math.confusion_matrix(y_true,y_pred).numpy()

plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
plt.title("Confusion Matrix");plt.xlabel("Predicted");plt.ylabel("True")
plt.show()