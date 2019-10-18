import glob, os
import numpy as np
from sklearn.model_selection import train_test_split
import sys

image_paths_file = sys.argv[1] #traffic_lights/traffic_lights.txt

# Percentage of images to be used for the test set (float between 0-1)
percentage_test = float(sys.argv[2]);

img_paths = []
img_paths = open(image_paths_file).read().strip().split()

X_train, X_test= train_test_split(img_paths, test_size=percentage_test, random_state=31)

with open('train.txt', 'a') as train_file:
	for train in X_train:
		train_file.write(train + '\n')

with open('test.txt', 'a') as test_file:
	for test in X_test:
		test_file.write(test + '\n')