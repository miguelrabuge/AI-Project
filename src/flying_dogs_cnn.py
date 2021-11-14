from tensorflow.keras import models, layers, losses
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from load_images import load_images

input_size = (224, 224, 3)

def flying_dogs_cnn():
    def flying_dogs_model():
        model = models.Sequential()
        
        model.add(layers.Conv2D(6, (3, 3), activation='relu', input_shape=input_size))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.AveragePooling2D())
        
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(5, activation='relu'))
        model.add(layers.Dropout(0.25))

        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    return KerasClassifier(flying_dogs_model)

if __name__ == "__main__":
    # Loading Data
    DIR = "processed" 
    train_dir_path = f"../data/{DIR}/train"
    val_dir_path = f"../data/{DIR}/val"
    X_train, y_train, X_val, y_val = load_images(train_dir_path, val_dir_path)

    # Classify
    cnn = flying_dogs_cnn()
    cnn.fit(X_train, y_train)
    cnn.score(X_val, y_val)

