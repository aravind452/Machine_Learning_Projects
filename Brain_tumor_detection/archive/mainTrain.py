import cv2
import os

from PIL import Image

import numpy as np

from sklearn.model_selection import train_test_split 

image_directory = 'archive/datasets/'

no_tumor_images = os.listdir(image_directory+'no/')
yes_tumor_images = os.listdir(image_directory+'yes/') 

#print(no_tumor_images)
#print(yes_tumor_images)

#path = 'no0.jpg'
#print(path.split('.')[1])


dataset = []
label = []

for i,image_name in enumerate(no_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'/no'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(0)


for i,image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image=cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((64,64))
        dataset.append(np.array(image))
        label.append(1)

        
#print(dataset)
#print(label)

dataset  = np.array(dataset)
label = np.array(label)

