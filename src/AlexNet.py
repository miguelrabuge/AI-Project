# Based on https://towardsdatascience.com/implementing-alexnet-cnn-architecture-using-tensorflow-2-0-and-keras-2113e090ad98

import loader
from tensorflow import random
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

random.set_seed(42)
input_size = (224, 224, 3)

def AlexNet_cnn():
    def AlexNet_model():
        model = models.Sequential()
        model.add(layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=input_size))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

        model.add(layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))

        model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())

        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=(3,3), strides=(2,2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(5, activation='softmax'))

        model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])
        return model

    return KerasClassifier(AlexNet_model)

if __name__ == "__main__":
    EPOCHS = 10
    # Loading Data
    X_train, y_train, X_val, y_val = loader.main()

    # Classify
    cnn = AlexNet_cnn()
    cnn.fit(X_train, y_train, epochs=EPOCHS)
    cnn.score(X_val, y_val)