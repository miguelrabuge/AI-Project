import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile

def load_images(train_dir_path, val_dir_path):

    ###############################################
    ### Loads the Images in the PVTL Subfolders ###
    ###############################################

    dir_paths = [train_dir_path, val_dir_path]
    
    labels_dict = {'vehicle-red': 0, 'vehicle-green': 1, 'pedestrian-red': 2, 'pedestrian-green': 3, 'others': 4}
    
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

    return np.array(X_train), y_train, np.array(X_val), y_val



if __name__ == '__main__':
    DIR = "processed" 
    train_dir_path = f"../data/{DIR}/train"
    val_dir_path = f"../data/{DIR}/val"

    X_train, y_train, X_val, y_val = load_images(train_dir_path, val_dir_path)
    print("n_elements of: X_train - {},  y_train - {}".format(len(X_train), len(y_train)))
    print("n_elements of: X_val - {},  y_val - {}".format(len(X_val), len(y_val)))
    print(y_train)