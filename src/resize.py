import os
import cv2 # OpenCV lib for image manipulation

RAW_ROOT = "../data/PVTL_dataset/"
PCD_ROOT = "../data/processed/"
TRAIN_PATH = "train/"
VAL_PATH = "val/"

def get_image_names(path_type):
    return [f"{path_type}{img_class}/{image}" for img_class in os.listdir(path_type) for image in os.listdir(f"{path_type}{img_class}")]

def get_sizes(images):
    widths, heights = [], []
    for image in images:
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        widths.append(img.shape[1])
        heights.append(img.shape[0])
    return widths, heights

def resize_and_pad_images(images, w_target, h_target):
    for image in images:
        # Image
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        
        # Getting minimum ratio (to maintain image format)
        ratio_w = w_target / img.shape[1]
        ratio_h = h_target / img.shape[0]
        desired_ratio = min(ratio_w, ratio_h)
        
        # Resizing
        desired_size = [int(desired_ratio * img.shape[0]), int(desired_ratio * img.shape[1])]
        resized = cv2.resize(img, (desired_size[1], desired_size[0]))
        
        # Padding and Centering Image
        delta_w = w_target - desired_size[1]
        delta_h = h_target - desired_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
        
        # Writing Image to Processed Path
        cv2.imwrite(PCD_ROOT + image[len(RAW_ROOT):], padded)

if __name__ == "__main__":
    # Getting Training and Validation Images
    train_images = get_image_names(RAW_ROOT + TRAIN_PATH)
    validation_images = get_image_names(RAW_ROOT + VAL_PATH)
    total_images = train_images + validation_images

    # Getting Maximum sizes
    widths, heights = get_sizes(total_images)
    w_max, h_max = max(widths), max(heights)
    
    w_max, h_max = 224, 224 # Hardcoding

    # Resizing and Padding Images to maximum Width and Height, maintaining picture ratio
    resize_and_pad_images(total_images, w_max, h_max)
    print(f"Resized and Padded Images to w,h = ({w_max}, {h_max})") # (388, 884)
    
