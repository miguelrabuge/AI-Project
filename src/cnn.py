# from keras import models, layers, losses
# from keras.wrappers.scikit_learn import KerasClassifier

# class CNN():

#     def __init__(self, inputShape: tuple, layerFilters: list, layerTransferFunc: list):
#         if len(layerFilters) != len(layerTransferFunc):
#             self.cnn = None
#         else:
#             model = models.Sequential()
            
#             model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
#             model.add(layers.MaxPooling2D((2, 2)))
#             model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#             model.add(layers.MaxPooling2D((2, 2)))
#             model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            
#             model.add(layers.Flatten())
#             model.add(layers.Dense(64, activation='relu'))
#             model.add(layers.Dense(10))

#             model.compile(optimizer='adam',loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

#             self.cnn = KerasClassifier(model)
#         return self


# # Flying Dog Network

# fd_nn = models.Sequential()

# fd_nn.add(layers.Conv2D())
#         self.conv1 = nn.Conv2d(3, 6, 3)
#         self.pool = nn.AvgPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         self.conv3 = nn.Conv2d(16, 16, 3)
#         self.conv4 = nn.Conv2d(16, 16, 3)
#         self.conv5 = nn.Conv2d(16, 16, 3)

#         self.fc1 = nn.Linear(16 * 12 * 12, 1024)
#         self.fc2 = nn.Linear(1024, 64)
#         self.fc3 = nn.Linear(64, 5)
#         self.dropout = nn.Dropout(0.25)


# if __name__ == "__main__":
#     pass