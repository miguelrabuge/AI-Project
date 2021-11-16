# Based on https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

import loader
from tensorflow import random
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

random.set_seed(42)
input_size = (224, 224, 3)

def VGG16_cnn():
    def VGG16_model():
        model = models.Sequential()
        model.add(layers.Conv2D(input_shape=input_size, filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=4096,activation="relu"))
        model.add(layers.Dense(units=5, activation="softmax"))
        
        model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])
        return model

    return KerasClassifier(VGG16_model)

if __name__ == "__main__":
    EPOCHS = 2
    # Load Data
    X_train, y_train, X_val, y_val = loader.main()

    # Classifier
    cnn = VGG16_cnn()
    cnn.fit(X_train, y_train, epochs=EPOCHS)
    cnn.score(X_val, y_val)