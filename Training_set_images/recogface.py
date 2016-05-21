# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:36:28 2016

@author: andresberejnoi
"""

import cv2
import os
import numpy as np
from AI.ANNs.NeuralNet import network

# The training images will be 100X100 pixels.
# I will be using my feedforward network and see how well it performs,
# however, it is not a good application, since the input layer has to be really big to 
# match the number of pixels. I will make it better later.

# get a list of files in the directory:
list_cats = os.listdir('cats')
list_dogs = os.listdir('dogs')
list_faces = os.listdir('faces')


print (list_cats)
print(list_dogs)
print(list_faces)

#net = network(topology=[10000,12000,12000,4])

# We need to read the images from the folders:
