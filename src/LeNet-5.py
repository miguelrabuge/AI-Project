import loader
from tensorflow.keras import models, layers, losses, optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

input_size = (224, 224, 3)

def LeNet5_cnn():
    def LeNet5_model():
        model = models.Sequential()

        model.add(layers.Conv2D(6, kernel_size=(3, 3), activation='relu', input_shape=input_size))
        model.add(layers.AveragePooling2D())

        model.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())

        model.add(layers.Flatten())
        model.add(layers.Dense(120, activation='relu'))
        model.add(layers.Dense(84, activation='relu'))
        model.add(layers.Dense(5, activation = 'softmax'))
        
        model.compile(loss=losses.categorical_crossentropy, optimizer=optimizers.Adam(), metrics=['accuracy'])
        return model

    return KerasClassifier(LeNet5_model)

if __name__ == "__main__":
    EPOCHS = 20
    # Load Data
    X_train, y_train, X_val, y_val = loader.main()

    # Classifier
    cnn = LeNet5_cnn()
    cnn.fit(X_train, y_train, epochs=EPOCHS)
    cnn.score(X_val, y_val)