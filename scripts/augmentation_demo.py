from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import numpy as np
import argparse
import os

# Parser
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=True, help="path to the input image")
ap.add_argument('-o', '--output', required=True, help="output directory")
ap.add_argument('-p', '--prefix', type=str, default="image", help="output filename prefix")

args = vars(ap.parse_args())

# if output dir doesn't exist, then create it

if not os.path.exists(args['output']):
    os.mkdir(args['output'])

# loading the image

print(f"[INFO] loading {args['image']}")
image = load_img(args['image']) # loading
image = img_to_array(image) # convertion to numpy array
image = np.expand_dims(image, axis=0) # adding an extra dimension

# construct augmentation generator for data augmentation 

aug  = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
     height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
     horizontal_flip = True, fill_mode="nearest")


total = 0 # then init the total nummber of imgs

print("[INFO] generating images...")
imageGen = aug.flow(image, batch_size=1, save_to_dir=args["output"],
        save_prefix=args['prefix'], save_format="jpg")

for image in imageGen:
    total +=1

    if total == 10:
        break