from os import listdir
from os.path import isfile
from PIL import Image

# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
import numpy as np
from numpy.lib.npyio import load


'''
train_dir_path = "/Users/gabriel/Documents/UC/4-ano/1-semestre/IA/PVTL_dataset/train"
val_dir_path = "/Users/gabriel/Documents/UC/4-ano/1-semestre/IA/PVTL_dataset/val"

vehicle_red = []

labels_dict = {'vehicle-red': 0, 'vehicle-green': 1, 'pedestrian-red': 2, 'pedestrian-green': 3}

lbs = list(labels_dict.keys())
print(lbs)

### obter imagens
p = train_dir_path + "/" + lbs[0] + "/"
images = [p + f for f in listdir(p) if isfile(p + f)]
#print(images)


for img in images:
    #temp = np.asarray(image.imread(img))
    temp = np.asarray(Image.open(img))
    #print(temp)
    vehicle_red.append(temp)

label_values = [labels_dict[lbs[0]]] * len(images)
l = [labels_dict[lbs[0]] for _ in range(len(images))]

print(label_values)


#print(vehicle_red[0])

pyplot.imshow(Image.fromarray(vehicle_red[0]))
pyplot.show()




print(labels_dict)
#im = Image.open(images[0], 'r')
#im.show()


# # load image as pixel array
# data = image.imread(images[0])
# # summarize shape of the pixel array
# print(data.dtype)
# print(data.shape)
# # display the array of pixels as an image
# pyplot.imshow(data)
# pyplot.show()

'''

def load_images(train_dir_path, val_dir_path):
    ###
    #Loads the images in the pvtl subfolders
    ###

    # train_dir_path = "/Users/gabriel/Documents/UC/4-ano/1-semestre/IA/PVTL_dataset/train"
    # val_dir_path = "/Users/gabriel/Documents/UC/4-ano/1-semestre/IA/PVTL_dataset/val"
    dir_paths = [train_dir_path, val_dir_path]
    
    labels_dict = {'vehicle-red': 0, 'vehicle-green': 1, 'pedestrian-red': 2, 'pedestrian-green': 3}
    
    lbs = list(labels_dict.keys())

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    for e in range(len(dir_paths)):
        for l in lbs:
            if e == 0:
                p = dir_paths[e] + "/" + l + "/"
                image_paths = [p + f for f in listdir(p) if isfile(p + f)]

                for img_p in image_paths:
                    temp = np.asarray(Image.open(img_p))
                    X_train.append(temp)
                y_train = y_train + [labels_dict[l] for _ in range(len(image_paths))]
            
            elif e == 1:
                p = dir_paths[e] + "/" + l + "/"
                image_paths = [p + f for f in listdir(p) if isfile(p + f)]

                for img_p in image_paths:
                    temp = np.asarray(Image.open(img_p))
                    X_val.append(temp)
                y_val = y_val + [labels_dict[l] for _ in range(len(image_paths))]
                    
        
    

    return X_train, y_train, X_val, y_val



if __name__ == '__main__':
    train_dir_path = "/Users/gabriel/Documents/UC/4-ano/1-semestre/IA/PVTL_dataset/train"
    val_dir_path = "/Users/gabriel/Documents/UC/4-ano/1-semestre/IA/PVTL_dataset/val"

    X_train, y_train, X_val, y_val = load_images(train_dir_path, val_dir_path)

    print("n_elements of: X_train - {},  y_train - {}".format(len(X_train), len(y_train)))
    print("n_elements of: X_val - {},  y_val - {}".format(len(X_val), len(y_val)))

    print(y_val)