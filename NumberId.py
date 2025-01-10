import os
import cv2 #Computer Vision to process images
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
#myenv\Scripts\activate to open the virtual enviornment


mnist = tf.keras.datasets.mnist
(xTrain, yTrain) , (xTest, yTest) = mnist.load_data()

xTrain = tf.keras.utils.normalize(xTrain, axis=1)
xTest = tf.keras.utils.normalize(xTest, axis=1)
"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(xTrain, yTrain, epochs=3)


model.save('handwritten.model')
"""
#Model has already been trained, so you can comment it out

model = tf.keras.models.load_model('handwritten.model')

"""loss, accuracy = model.evaluate(xTest, yTest)

print(loss) #want to be low as possible
print(accuracy) #want to be high as possible"""

imageNumber = 1

while os.path.isfile(f"digits/digit {imageNumber}.png"):
    try:
        img = cv2.imread(f"digits/digit {imageNumber}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'The number is probably {np.argmax(prediction)}')
        plt.imshow(img[0], cmap=plt.cm.binary)
    
    except:
        print("Error!")
    
    finally:
        imageNumber += 1


